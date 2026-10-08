# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply PII detector decisions to persisted catalog columns."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from auto_ontology.dal.session import IN_QUERY_BATCH
from auto_ontology.dal.tags import (
    PII_TAG_NAME,
    TARGET_COLUMN,
    attach_tag,
    fetch_tags_map,
    get_or_create_tag,
    get_tag_by_name,
)
from auto_ontology.dal.pii import (
    mark_columns_pii_processed,
    tag_attributes_of_tagged_columns,
)
from auto_ontology.pii_detection.detector import LlmPiiClassifier, PiiDetector
from auto_ontology.pii_detection.models import ColumnInput, PiiStatus

logger = logging.getLogger(__name__)

AUTO_TAG_THRESHOLD = 0.9
# Same size as the DAL ``IN`` batches: classify, tag, and mark this many
# columns before the next chunk, so a later failure still keeps earlier
# ``pii_processed`` marks and the next ingest can continue.
PII_COLUMN_BATCH = IN_QUERY_BATCH


@dataclass(frozen=True, slots=True)
class PiiTaggingResult:
    """Operational counts from one catalog PII enrichment pass."""

    scanned: int = 0
    processed: int = 0
    rules_decided: int = 0
    llm_decided: int = 0
    review: int = 0
    tagged: int = 0
    already_tagged: int = 0


def _optional_text(value: Any) -> str | None:
    """Convert a DataFrame value to optional text without leaking ``pd.NA``."""

    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _is_processed(value: Any) -> bool:
    """Interpret a persisted PII marker, treating missing values as false."""

    return value is not None and not pd.isna(value) and bool(value)


def _add_results(left: PiiTaggingResult, right: PiiTaggingResult) -> PiiTaggingResult:
    """Sum two enrichment passes into one operational total."""

    return PiiTaggingResult(
        scanned=left.scanned + right.scanned,
        processed=left.processed + right.processed,
        rules_decided=left.rules_decided + right.rules_decided,
        llm_decided=left.llm_decided + right.llm_decided,
        review=left.review + right.review,
        tagged=left.tagged + right.tagged,
        already_tagged=left.already_tagged + right.already_tagged,
    )


def _apply_batch(
    records: list[tuple[str, ColumnInput]],
    classifier: PiiDetector,
    threshold: float,
) -> PiiTaggingResult:
    """Classify one batch, attach tags, then persist processed marks."""

    decisions = classifier.detect_many(column for _, column in records)
    rules_decided = sum(decision.source == "rules" for _, decision in decisions)
    llm_decided = sum(decision.source == "llm" for _, decision in decisions)
    review = sum(decision.status is PiiStatus.REVIEW for _, decision in decisions)
    processed_ids = [
        column_id
        for (column_id, _), (_, decision) in zip(records, decisions, strict=True)
        if decision.source != "fallback"
    ]
    candidates = [
        column_id
        for (column_id, _), (_, decision) in zip(records, decisions, strict=True)
        if decision.source != "fallback" and decision.should_auto_tag(threshold)
    ]

    tagged = 0
    already_tagged = 0
    if candidates:
        pii_tag = get_or_create_tag(name=PII_TAG_NAME)
        tag_id = str(pii_tag["id"])
        existing = fetch_tags_map(TARGET_COLUMN, candidates)
        for column_id in candidates:
            if any(tag["id"] == tag_id for tag in existing.get(column_id, [])):
                already_tagged += 1
                continue
            if (
                attach_tag(
                    tag_id=tag_id,
                    kind=TARGET_COLUMN,
                    item_id=column_id,
                    tagged_by=None,
                )
                is None
            ):
                raise RuntimeError(
                    f"Could not attach PII tag to catalog column {column_id!r}"
                )
            tagged += 1

    processed = mark_columns_pii_processed(processed_ids)
    return PiiTaggingResult(
        scanned=len(records),
        processed=processed,
        rules_decided=rules_decided,
        llm_decided=llm_decided,
        review=review,
        tagged=tagged,
        already_tagged=already_tagged,
    )


def detect_and_tag_pii(
    columns_df: pd.DataFrame,
    *,
    detector: PiiDetector | None = None,
    threshold: float = AUTO_TAG_THRESHOLD,
) -> PiiTaggingResult:
    """Classify persisted columns and attach the shared ``PII`` tag.

    ``columns_df`` is the frame returned by catalog ingestion, so every row is
    expected to carry its stable catalog-column id. Detection sees metadata
    only; sample values are deliberately not part of :class:`ColumnInput`.

    Work is applied in batches of :data:`PII_COLUMN_BATCH`: each chunk is
    classified, tagged, and marked processed before the next one starts. A
    failure in a later chunk therefore leaves earlier ``pii_processed`` marks
    in place, so the next ingest continues instead of re-classifying the
    whole catalog.
    """

    if columns_df is None or columns_df.empty:
        return PiiTaggingResult()

    records: list[tuple[str, ColumnInput]] = []
    for row in columns_df.to_dict(orient="records"):
        if _is_processed(row.get("pii_processed")):
            continue
        column_id = _optional_text(row.get("id"))
        column_name = _optional_text(row.get("column_name"))
        if column_id is None or column_name is None:
            logger.warning(
                "Skipping PII detection for a catalog column without id/name"
            )
            continue
        records.append(
            (
                column_id,
                ColumnInput(
                    column_name=column_name,
                    table_name=_optional_text(row.get("table_name")),
                    description=_optional_text(row.get("description")),
                ),
            )
        )

    if not records:
        return PiiTaggingResult()

    classifier = detector or PiiDetector(LlmPiiClassifier())
    total = PiiTaggingResult()
    batch_count = (len(records) + PII_COLUMN_BATCH - 1) // PII_COLUMN_BATCH
    try:
        for batch_index, offset in enumerate(
            range(0, len(records), PII_COLUMN_BATCH), start=1
        ):
            part = _apply_batch(
                records[offset : offset + PII_COLUMN_BATCH],
                classifier,
                threshold,
            )
            total = _add_results(total, part)
            if batch_count > 1:
                logger.info(
                    "PII detection batch %d/%d: %d scanned, %d processed, "
                    "%d rules, %d LLM, %d review, %d tagged, %d already tagged",
                    batch_index,
                    batch_count,
                    part.scanned,
                    part.processed,
                    part.rules_decided,
                    part.llm_decided,
                    part.review,
                    part.tagged,
                    part.already_tagged,
                )
    except Exception:
        logger.warning(
            "PII detection interrupted after %d column(s) processed of %d "
            "scanned; remaining unprocessed columns will retry on the next "
            "ingest",
            total.processed,
            len(records),
        )
        raise

    logger.info(
        "PII detection finished: %d scanned, %d processed, %d rules, %d LLM, "
        "%d review, %d tagged, %d already tagged",
        total.scanned,
        total.processed,
        total.rules_decided,
        total.llm_decided,
        total.review,
        total.tagged,
        total.already_tagged,
    )
    return total


@dataclass(frozen=True, slots=True)
class PiiPropagationResult:
    """Attributes newly labelled ``PII`` because a column they use is PII."""

    column_attributes: int = 0
    sql_attributes: int = 0


def propagate_pii_to_attributes() -> PiiPropagationResult:
    """Label the attributes built from PII columns with the shared ``PII`` tag.

    A ColumnAttribute is tagged when one of its columns is, and a SqlAttribute
    when its SQL reads one. Driven by the labels already on the columns rather
    than by detector output, so hand-applied PII tags propagate too and the
    pass is idempotent.

    Run it after PII detection **and** after semantic compilation: attributes
    are created by the latter, so on a first ingest there is nothing to label
    yet and a later pass has to pick them up.

    Does not create the tag: if no column was ever tagged, there is nothing to
    propagate.
    """

    pii_tag = get_tag_by_name(PII_TAG_NAME)
    if pii_tag is None:
        return PiiPropagationResult()

    column_attributes, sql_attributes = tag_attributes_of_tagged_columns(
        str(pii_tag["id"])
    )
    result = PiiPropagationResult(
        column_attributes=column_attributes, sql_attributes=sql_attributes
    )
    if column_attributes or sql_attributes:
        logger.info(
            "PII propagation: %d column attribute(s) and %d SQL attribute(s) tagged",
            column_attributes,
            sql_attributes,
        )
    return result
