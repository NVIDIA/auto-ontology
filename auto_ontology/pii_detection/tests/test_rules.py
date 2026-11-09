# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from auto_ontology.pii_detection.models import ColumnInput
from auto_ontology.pii_detection.rules import evaluate_rules, normalize


@pytest.mark.parametrize(
    ("column_name", "expected"),
    [
        ("주민등록번호", "주민등록번호"),
        ("전화번호", "전화번호"),
        ("телефон", "телефон"),
        ("البريد_الإلكتروني", "البريد الإلكتروني"),
        ("téléphone", "téléphone"),
    ],
)
def test_non_latin_column_names_survive_normalization(
    column_name: str, expected: str
) -> None:
    """Non-Latin names used to collapse to an empty string under the old
    ASCII-only normalizer, which the rules then misread as a missing name."""

    assert normalize(column_name) == expected


@pytest.mark.parametrize(
    ("column_name", "expected"),
    [
        ("Customer_Email", "customer email"),
        ("SSN", "ssn"),
        ("home-phone", "home phone"),
        ("AddressLine1", "addressline1"),
        ("first.name", "first name"),
        ("e_mail", "e mail"),
    ],
)
def test_ascii_names_normalize_as_before(column_name: str, expected: str) -> None:
    """The Unicode-aware split must not change any ASCII normalization."""

    assert normalize(column_name) == expected


@pytest.mark.parametrize(
    ("column_name", "table_name"),
    [
        ("주민등록번호", "고객"),
        ("전화번호", "顧客"),
        ("телефон", "клиенты"),
        ("البريد_الإلكتروني", "العملاء"),
        ("téléphone", "clients"),
    ],
)
def test_non_latin_columns_fall_through_to_classifier(
    column_name: str, table_name: str
) -> None:
    """No English rule matches a non-Latin name, so the rules must stay
    silent and let the LLM classifier see the column instead of returning
    a REVIEW decision that marks it processed."""

    assert (
        evaluate_rules(ColumnInput(column_name=column_name, table_name=table_name))
        is None
    )


@pytest.mark.parametrize("column_name", ["", "   ", "___", "---"])
def test_empty_or_separator_only_name_is_not_a_rules_decision(
    column_name: str,
) -> None:
    """A name with no signal must not produce a rules REVIEW: the service
    layer marks every non-fallback decision processed, which would hide
    the column from the classifier and from later ingests."""

    assert evaluate_rules(ColumnInput(column_name=column_name)) is None
