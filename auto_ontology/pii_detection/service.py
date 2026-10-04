# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply PII detector decisions to persisted catalog columns."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from auto_ontology.dal.tags import (
    TARGET_COLUMN,
    attach_tag,
    fetch_tags_map,
    get_or_create_tag,
)
from auto_ontology.dal.pii import mark_columns_pii_processed
from auto_ontology.pii_detection.detector import LlmPiiClassifier, PiiDetector
from auto_ontology.pii_detection.models import ColumnInput, PiiStatus

logger = logging.getLogger(__name__)

PII_TAG_NAME = "PII"
AUTO_TAG_THRESHOLD = 0.9


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
    decisions = classifier.detect_many(column for _, column in records)
    rules_decided = sum(decision.source == "rules" for _, decision in decisions)
    llm_decided = sum(decision.source == "llm" for _, decision in decisions)
    review = sum(decision.status is PiiStatus.REVIEW for _, decision in decisions)
    # A PII decision below the auto-tag threshold stays unprocessed: data
    # movement treats processed-and-untagged columns as safe.
    processed_ids = [
        column_id
        for (column_id, _), (_, decision) in zip(records, decisions, strict=True)
        if decision.source != "fallback"
        and (not decision.is_pii or decision.should_auto_tag(threshold))
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
    result = PiiTaggingResult(
        scanned=len(records),
        processed=processed,
        rules_decided=rules_decided,
        llm_decided=llm_decided,
        review=review,
        tagged=tagged,
        already_tagged=already_tagged,
    )
    logger.info(
        "PII detection finished: %d scanned, %d processed, %d rules, %d LLM, "
        "%d review, %d tagged, %d already tagged",
        result.scanned,
        result.processed,
        result.rules_decided,
        result.llm_decided,
        result.review,
        result.tagged,
        result.already_tagged,
    )
    return result
