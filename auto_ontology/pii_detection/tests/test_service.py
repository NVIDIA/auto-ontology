# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
    assert result.processed == 0
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
