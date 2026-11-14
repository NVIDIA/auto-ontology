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
    ("column_name", "expected"),
    [
        ("e\u0301mail", "émail"),
        ("product_name\u0301", "product namé"),
        ("na\u0301me", "náme"),
        ("te\u0301le\u0301phone", "téléphone"),
    ],
)
def test_decomposed_latin_names_normalize_like_composed_form(
    column_name: str, expected: str
) -> None:
    """A decomposed name must normalize exactly like its composed form.

    The combining mark is part of the word, not a separator: under the old
    split, ``e`` plus combining acute plus ``mail`` became ``e mail`` and
    matched the email rule, while composed ``émail`` fell through."""

    assert normalize(column_name) == expected
    assert normalize(expected) == expected


@pytest.mark.parametrize(
    ("column_name", "expected"),
    [
        ("नाम", "नाम"),
        ("ईमेल", "ईमेल"),
        ("ชื่อ", "ชื่อ"),
        ("الْبَرِيد", "الْبَرِيد"),
    ],
)
def test_mark_bearing_names_keep_combining_marks(
    column_name: str, expected: str
) -> None:
    """Hindi, Thai, and Arabic vowel signs and diacritics are combining
    marks, not separators: splitting on them shredded the name into
    fragments. They must stay attached to their base characters."""

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


@pytest.mark.parametrize(
    "column_name",
    [
        "product_name\u0301",
        "e\u0301mail",
        "na\u0301me",
    ],
)
def test_decomposed_names_do_not_match_ascii_rules(column_name: str) -> None:
    """Regression: a decomposed name must not match the rule its ASCII
    lookalike would match. Decomposed ``product_name`` plus combining acute
    matched the ``product name`` non-PII rule and was marked processed,
    while composed ``product_namé`` correctly fell through."""

    assert evaluate_rules(ColumnInput(column_name=column_name)) is None


@pytest.mark.parametrize(
    ("column_name", "table_name"),
    [
        ("नाम", "ग्राहक"),
        ("ईमेल", "ग्राहक"),
        ("ชื่อ", "ลูกค้า"),
        ("الْبَرِيد", "العملاء"),
    ],
)
def test_mark_bearing_names_fall_through_to_classifier(
    column_name: str, table_name: str
) -> None:
    """No English rule matches a mark-bearing Hindi, Thai, or Arabic name,
    so the rules must stay silent and let the LLM classifier see it."""

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
