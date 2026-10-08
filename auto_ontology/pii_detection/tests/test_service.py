# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import pytest

from auto_ontology.pii_detection.detector import PiiDetector
from auto_ontology.pii_detection.models import ColumnInput, PiiDecision, PiiStatus
from auto_ontology.pii_detection import service


class _RecordingBackend:
    def __init__(self, confidence: float = 0.95) -> None:
        self.columns: list[ColumnInput] = []
        self.confidence = confidence

    def classify(self, column: ColumnInput) -> PiiDecision:
        self.columns.append(column)
        return PiiDecision(
            status=PiiStatus.PII,
            category="identifier",
            confidence=self.confidence,
            reason="The metadata identifies an individual.",
            source="llm",
        )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "email-id",
                "column_name": "email_address",
                "table_name": "customers",
                "description": "Customer email",
            },
            {
                "id": "reference-id",
                "column_name": "external_reference",
                "table_name": "events",
                "description": pd.NA,
            },
            {
                "id": "product-id",
                "column_name": "product_name",
                "table_name": "products",
                "description": "Display name",
            },
        ]
    )


@pytest.fixture(autouse=True)
def persist_processed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        service, "mark_columns_pii_processed", lambda column_ids: len(column_ids)
    )


def test_applies_one_shared_tag_and_reuses_existing_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _RecordingBackend()
    attached: list[dict[str, Any]] = []

    monkeypatch.setattr(
        service, "get_or_create_tag", lambda **_kwargs: {"id": "pii-tag"}
    )
    monkeypatch.setattr(
        service,
        "fetch_tags_map",
        lambda _kind, _ids: {"email-id": [{"id": "pii-tag", "name": "PII"}]},
    )

    def attach(**kwargs: Any) -> list[dict[str, str]]:
        attached.append(kwargs)
        return [{"id": "pii-tag", "name": "PII"}]

    monkeypatch.setattr(service, "attach_tag", attach)

    result = service.detect_and_tag_pii(_frame(), detector=PiiDetector(backend))

    assert result.scanned == 3
    assert result.processed == 3
    assert result.rules_decided == 2
    assert result.llm_decided == 1
    assert result.tagged == 1
    assert result.already_tagged == 1
    assert [call["item_id"] for call in attached] == ["reference-id"]
    assert backend.columns == [
        ColumnInput(
            column_name="external_reference",
            table_name="events",
            description=None,
        )
    ]


def test_below_threshold_does_not_create_a_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _RecordingBackend(confidence=0.89)

    def unexpected(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("tag vocabulary should not be touched")

    monkeypatch.setattr(service, "get_or_create_tag", unexpected)
    result = service.detect_and_tag_pii(
        _frame().iloc[[1]], detector=PiiDetector(backend)
    )

    assert result.tagged == 0
    assert result.processed == 1
    assert result.llm_decided == 1


def test_empty_catalog_does_not_construct_default_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        service,
        "LlmPiiClassifier",
        lambda: (_ for _ in ()).throw(AssertionError("must not construct backend")),
    )

    assert service.detect_and_tag_pii(pd.DataFrame()) == service.PiiTaggingResult()


def test_missing_column_identity_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        service, "get_or_create_tag", lambda **_kwargs: {"id": "pii-tag"}
    )
    monkeypatch.setattr(service, "fetch_tags_map", lambda _kind, _ids: {})
    monkeypatch.setattr(service, "attach_tag", lambda **_kwargs: [])

    frame = pd.DataFrame(
        [{"id": None, "column_name": "email"}, {"id": "x", "column_name": None}]
    )
    result = service.detect_and_tag_pii(frame, detector=PiiDetector())

    assert result.scanned == 0


def test_processed_column_is_not_classified_or_retagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("processed column must not touch the tag vocabulary")

    monkeypatch.setattr(service, "get_or_create_tag", unexpected)
    frame = _frame().iloc[[0]].assign(pii_processed=True)

    result = service.detect_and_tag_pii(frame)

    assert result == service.PiiTaggingResult()


def test_marks_negative_decision_as_processed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed: list[str] = []
    monkeypatch.setattr(
        service,
        "mark_columns_pii_processed",
        lambda column_ids: processed.extend(column_ids) or len(column_ids),
    )

    result = service.detect_and_tag_pii(_frame().iloc[[2]], detector=PiiDetector())

    assert result.rules_decided == 1
    assert result.tagged == 0
    assert processed == ["product-id"]


def test_classifier_failure_remains_unprocessed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingBackend:
        def classify(self, _column: ColumnInput) -> PiiDecision:
            raise RuntimeError("temporary failure")

    processed: list[str] = []
    monkeypatch.setattr(
        service,
        "mark_columns_pii_processed",
        lambda column_ids: processed.extend(column_ids) or len(column_ids),
    )

    result = service.detect_and_tag_pii(
        _frame().iloc[[1]], detector=PiiDetector(FailingBackend())
    )

    assert result.review == 1
    assert result.processed == 0
    assert processed == []


def test_propagation_labels_attributes_of_the_existing_pii_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(service, "get_tag_by_name", lambda _name: {"id": "pii-tag"})
    monkeypatch.setattr(
        service,
        "tag_attributes_of_tagged_columns",
        lambda tag_id: seen.append(tag_id) or (2, 3),
    )

    result = service.propagate_pii_to_attributes()

    assert seen == ["pii-tag"]
    assert result == service.PiiPropagationResult(column_attributes=2, sql_attributes=3)


def test_propagation_without_a_pii_tag_does_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service, "get_tag_by_name", lambda _name: None)

    def unexpected(_tag_id: str) -> tuple[int, int]:
        raise AssertionError("no tag means nothing to propagate")

    monkeypatch.setattr(service, "tag_attributes_of_tagged_columns", unexpected)

    assert service.propagate_pii_to_attributes() == service.PiiPropagationResult()


def _non_pii_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "p1",
                "column_name": "product_name",
                "table_name": "products",
                "description": "Display name",
            },
            {
                "id": "p2",
                "column_name": "account_type",
                "table_name": "accounts",
                "description": "Type",
            },
            {
                "id": "p3",
                "column_name": "created_at",
                "table_name": "events",
                "description": "Created",
            },
        ]
    )


def test_each_batch_is_marked_before_the_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service, "PII_COLUMN_BATCH", 1)
    processed: list[list[str]] = []
    monkeypatch.setattr(
        service,
        "mark_columns_pii_processed",
        lambda column_ids: processed.append(list(column_ids)) or len(column_ids),
    )

    result = service.detect_and_tag_pii(_non_pii_frame(), detector=PiiDetector())

    assert [ids[0] for ids in processed] == ["p1", "p2", "p3"]
    assert result.scanned == 3
    assert result.processed == 3
    assert result.rules_decided == 3


def test_a_failed_batch_keeps_earlier_processed_marks(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(service, "PII_COLUMN_BATCH", 1)
    processed: list[list[str]] = []

    def mark(column_ids: list[str]) -> int:
        if processed:
            raise RuntimeError("bind limit")
        processed.append(list(column_ids))
        return len(column_ids)

    monkeypatch.setattr(service, "mark_columns_pii_processed", mark)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match="bind limit"):
            service.detect_and_tag_pii(_non_pii_frame(), detector=PiiDetector())

    assert processed == [["p1"]]
    assert "interrupted after 1 column(s) processed of 3 scanned" in caplog.text


def test_a_tag_failure_does_not_mark_that_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service, "PII_COLUMN_BATCH", 1)
    processed: list[list[str]] = []
    monkeypatch.setattr(
        service,
        "mark_columns_pii_processed",
        lambda column_ids: processed.append(list(column_ids)) or len(column_ids),
    )
    monkeypatch.setattr(
        service, "get_or_create_tag", lambda **_kwargs: {"id": "pii-tag"}
    )
    monkeypatch.setattr(service, "fetch_tags_map", lambda _kind, _ids: {})

    def attach(**kwargs: Any) -> list[dict[str, str]]:
        if kwargs["item_id"] == "email-2":
            raise RuntimeError("attach failed")
        return [{"id": "pii-tag", "name": "PII"}]

    monkeypatch.setattr(service, "attach_tag", attach)
    frame = pd.DataFrame(
        [
            {
                "id": "email-1",
                "column_name": "email_address",
                "table_name": "customers",
                "description": "a",
            },
            {
                "id": "email-2",
                "column_name": "email_address",
                "table_name": "users",
                "description": "b",
            },
        ]
    )

    with pytest.raises(RuntimeError, match="attach failed"):
        service.detect_and_tag_pii(frame, detector=PiiDetector())

    assert processed == [["email-1"]]
