# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from threading import Barrier

from auto_ontology.pii_detection import (
    ColumnInput,
    LlmPiiClassifier,
    PiiDecision,
    PiiDetector,
    PiiStatus,
)
from auto_ontology.pii_detection import detector as detector_module
from auto_ontology.pii_detection.rules import evaluate_rules


class StubBackend:
    def __init__(self) -> None:
        self.calls = 0

    def classify(self, column: ColumnInput) -> PiiDecision:
        self.calls += 1
        return PiiDecision(
            status=PiiStatus.NOT_PII,
            category=None,
            confidence=0.8,
            reason="Stub classification.",
            source="stub",
        )


def test_direct_identifier_is_auto_tagged() -> None:
    decision = PiiDetector().detect(
        ColumnInput(column_name="customer_email_address", table_name="customers")
    )

    assert decision.status is PiiStatus.PII
    assert decision.category == "email_address"
    assert decision.should_auto_tag()


@pytest.mark.parametrize(
    ("column_name", "category"),
    [
        ("email", "email_address"),
        ("email_address", "email_address"),
        ("ssn", "government_id"),
        ("mobile", "phone_number"),
        ("phone", "phone_number"),
        ("password", "credential"),
        ("user_email", "email_address"),
        ("work_email", "email_address"),
        ("personal_email", "email_address"),
        ("home_phone", "phone_number"),
        ("cell_phone", "phone_number"),
        ("customer_ssn", "government_id"),
    ],
)
def test_direct_rules_still_match_the_field_itself(
    column_name: str, category: str
) -> None:
    decision = PiiDetector().detect(
        ColumnInput(column_name=column_name, table_name="events")
    )

    assert decision.status is PiiStatus.PII
    assert decision.category == category
    assert decision.source == "rules"


@pytest.mark.parametrize(
    "column_name",
    [
        "email_domain",
        "email_sent_at",
        "ssn_hash",
        "mobile_app_version",
        "password_reset_token",
        "no_email",
        "has_email",
        "is_mobile",
        "no_ssn",
        "email_address_hash",
        "phone_number_id",
        "cell_email",
        "has_first_name",
        "no_last_name",
        "first_name_hash",
        "no_address",
        "wallet_address",
        "contract_address",
        "has_age",
        "average_age",
        "gender_id",
        "has_salary",
        "salary_hash",
        "credit_score_model",
        "no_customer_id",
        "customer_id_hash",
    ],
)
def test_rules_do_not_match_qualified_lookalikes(column_name: str) -> None:
    """Rules must match the whole column name, not a suffix or subphrase.

    ``email_sent_at`` is a timestamp, ``no_email`` / ``has_first_name`` are
    flags, and ``email_address_hash`` / ``customer_id_hash`` are derived.
    Without a backend that becomes review, not ``not_pii`` -- an LLM can
    still decide.
    """
    column = ColumnInput(column_name=column_name, table_name="customers")

    assert evaluate_rules(column) is None

    decision = PiiDetector().detect(column)
    assert decision.status is PiiStatus.REVIEW
    assert decision.source == "fallback"


@pytest.mark.parametrize(
    ("column_name", "table_name", "category"),
    [
        ("emailaddress", "people", "email_address"),
        ("phonenumber", "customers", "phone_number"),
        ("fullname", "people", "person_name"),
        ("preferredname", "people", "person_name"),
        ("postaladdressline1", "customers", "postal_address"),
        ("home_address", "customers", "postal_address"),
        ("mailing_address", "customers", "postal_address"),
        ("customerid", "customers", "identifier"),
        ("id", "customers", "identifier"),
        ("customer_id", "customers", "identifier"),
        ("employee_id", "employees", "identifier"),
    ],
)
def test_compact_sql_identifiers_match_rules(
    column_name: str, table_name: str, category: str
) -> None:
    decision = PiiDetector().detect(
        ColumnInput(column_name=column_name, table_name=table_name)
    )

    assert decision.status is PiiStatus.PII
    assert decision.category == category
    assert decision.source == "rules"


def test_compact_id_suffix_does_not_match_unrelated_word() -> None:
    decision = PiiDetector().detect(
        ColumnInput(column_name="paid", table_name="customers")
    )

    assert decision.status is PiiStatus.REVIEW


@pytest.mark.parametrize(
    "column_name",
    ["store_id", "country_id", "last_update_id"],
)
def test_unqualified_id_suffix_is_not_a_person_identifier(column_name: str) -> None:
    """Bare ``id`` used to match any ``*_id`` on a person table and auto-tag it.

    The identifier rule must stay silent: ``last_update_id`` is an audit FK,
    not a person identifier. Without a backend that becomes review, not
    ``not_pii`` -- an LLM can still decide.
    """
    column = ColumnInput(column_name=column_name, table_name="customers")

    assert evaluate_rules(column) is None

    decision = PiiDetector().detect(column)
    assert decision.status is PiiStatus.REVIEW
    assert decision.source == "fallback"


@pytest.mark.parametrize(
    ("column_name", "category"),
    [
        ("full_name", "person_name"),
        ("first_name", "person_name"),
        ("address", "postal_address"),
        ("age", "personal_attribute"),
        ("gender", "personal_attribute"),
        ("salary", "financial"),
    ],
)
def test_conditional_rules_match_the_field_on_person_tables(
    column_name: str, category: str
) -> None:
    decision = PiiDetector().detect(
        ColumnInput(column_name=column_name, table_name="employees")
    )

    assert decision.status is PiiStatus.PII
    assert decision.category == category
    assert decision.source == "rules"


@pytest.mark.parametrize("table_name", ["customer_orders", "user_sessions"])
def test_person_token_in_table_name_is_not_enough(table_name: str) -> None:
    """``customer_orders`` contains customer but is not a person table."""
    column = ColumnInput(column_name="address", table_name=table_name)

    assert evaluate_rules(column) is None
    assert PiiDetector().detect(column).status is PiiStatus.REVIEW


def test_exact_name_on_a_person_table_is_pii() -> None:
    decision = PiiDetector().detect(
        ColumnInput(column_name="name", table_name="customers")
    )

    assert decision.status is PiiStatus.PII
    assert decision.category == "person_name"
    assert decision.source == "rules"


@pytest.mark.parametrize(
    "column_name",
    ["username", "filename", "category_name", "company_name"],
)
def test_generic_name_suffix_is_not_a_person_name(column_name: str) -> None:
    """Bare ``name`` used to match any column ending in name on a person table.

    The person-name rule must stay silent: ``filename`` is not a person name.
    Without a backend that becomes review, not ``not_pii`` -- an LLM can still
    decide.
    """
    column = ColumnInput(column_name=column_name, table_name="customers")

    assert evaluate_rules(column) is None

    decision = PiiDetector().detect(column)
    assert decision.status is PiiStatus.REVIEW
    assert decision.source == "fallback"


def test_product_name_is_not_pii() -> None:
    decision = PiiDetector().detect(
        ColumnInput(column_name="product_name", table_name="products")
    )

    assert decision.status is PiiStatus.NOT_PII


@pytest.mark.parametrize(
    "column_name",
    ["productname", "buyinggroupname", "cityname", "customercategoryname"],
)
def test_compact_business_name_is_not_pii(column_name: str) -> None:
    decision = PiiDetector().detect(
        ColumnInput(column_name=column_name, table_name="customers")
    )

    assert decision.status is PiiStatus.NOT_PII


def test_uncertain_column_requires_review_without_backend() -> None:
    decision = PiiDetector().detect(
        ColumnInput(column_name="external_reference", table_name="events")
    )

    assert decision.status is PiiStatus.REVIEW


def test_uncertain_column_uses_backend() -> None:
    backend = StubBackend()
    decision = PiiDetector(backend).detect(
        ColumnInput(column_name="external_reference", table_name="events")
    )

    assert decision.source == "stub"
    assert backend.calls == 1


def test_backend_failure_becomes_review() -> None:
    class FailingBackend:
        def classify(self, column: ColumnInput) -> PiiDecision:
            raise RuntimeError("unavailable")

    decision = PiiDetector(FailingBackend()).detect(
        ColumnInput(column_name="external_reference", table_name="events")
    )

    assert decision.status is PiiStatus.REVIEW
    assert decision.source == "fallback"


def test_batch_deduplicates_identical_requests() -> None:
    backend = StubBackend()
    column = ColumnInput(column_name="external_reference", table_name="events")

    results = PiiDetector(backend).detect_many([column, column])

    assert len(results) == 2
    assert backend.calls == 1


def test_empty_batch_returns_no_decisions() -> None:
    assert PiiDetector(StubBackend()).detect_many([]) == []


def test_batch_classifies_distinct_columns_concurrently() -> None:
    barrier = Barrier(3)

    class ConcurrentBackend:
        def classify(self, column: ColumnInput) -> PiiDecision:
            barrier.wait(timeout=2)
            return PiiDecision(
                status=PiiStatus.NOT_PII,
                category=None,
                confidence=0.8,
                reason=column.column_name,
                source="stub",
            )

    columns = [
        ColumnInput(column_name=f"external_reference_{index}", table_name="events")
        for index in range(3)
    ]
    results = PiiDetector(ConcurrentBackend(), max_workers=3).detect_many(columns)

    assert [column for column, _ in results] == columns


def test_llm_classifier_uses_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invoke(_llm: object, messages: list, schema: type) -> object:
        assert "reference" in str(messages[-1].content)
        return schema.model_validate(
            {
                "status": "pii",
                "category": "identifier",
                "confidence": 0.93,
                "reason": "Identifies a customer.",
            }
        )

    monkeypatch.setattr(detector_module, "invoke_with_structured_output", invoke)
    backend = LlmPiiClassifier(llm=object())  # type: ignore[arg-type]
    decision = backend.classify(ColumnInput(column_name="reference"))

    assert decision.status is PiiStatus.PII
    assert decision.should_auto_tag()


def test_llm_classifier_accepts_missing_optional_explanation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invoke(_llm: object, _messages: list, schema: type) -> object:
        return schema.model_validate({"status": "not_pii", "confidence": 0.78})

    monkeypatch.setattr(detector_module, "invoke_with_structured_output", invoke)
    classifier = LlmPiiClassifier(llm=object())  # type: ignore[arg-type]

    decision = classifier.classify(ColumnInput(column_name="cityname"))

    assert decision.status is PiiStatus.NOT_PII
    assert decision.confidence == 0.78
    assert decision.reason == "LLM classified the column as not_pii."


def test_llm_uncertainty_is_conservatively_treated_as_pii(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invoke(_llm: object, _messages: list, schema: type) -> object:
        return schema.model_validate(
            {
                "status": "review",
                "confidence": 0.2,
                "reason": "The available metadata is ambiguous.",
            }
        )

    monkeypatch.setattr(detector_module, "invoke_with_structured_output", invoke)
    classifier = LlmPiiClassifier(llm=object())  # type: ignore[arg-type]

    decision = classifier.classify(ColumnInput(column_name="id", table_name="case"))

    assert decision.status is PiiStatus.PII
    assert decision.confidence == 1.0
    assert decision.should_auto_tag()
