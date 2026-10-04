# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""High-confidence, metadata-only PII classification rules."""

import re
from dataclasses import dataclass

from auto_ontology.pii_detection.models import ColumnInput, PiiDecision, PiiStatus


def normalize(value: str | None) -> str:
    """Normalize SQL identifiers into words suitable for rule matching."""

    if not value:
        return ""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


@dataclass(frozen=True, slots=True)
class _Pattern:
    expression: re.Pattern[str]
    category: str
    compact_expression: re.Pattern[str] | None = None


def _pattern(words: str, category: str, *, compact: str | None = None) -> _Pattern:
    return _Pattern(
        re.compile(rf"\b(?:{words})\b"),
        category,
        re.compile(compact) if compact else None,
    )


def _matches(pattern: _Pattern, normalized: str, compact: str) -> bool:
    return bool(
        pattern.expression.search(normalized)
        or (
            pattern.compact_expression is not None
            and pattern.compact_expression.search(compact)
        )
    )


DIRECT_PII_PATTERNS = (
    _pattern(
        r"ssn|social security(?: number)?",
        "government_id",
        compact=r"(?:ssn|socialsecurity(?:number)?)$",
    ),
    _pattern(
        r"passport(?: number)?|driver(?:s)? license",
        "government_id",
        compact=r"(?:passport(?:number)?|drivers?license)$",
    ),
    _pattern(
        r"taxpayer id|tax id|national id",
        "government_id",
        compact=r"(?:taxpayerid|taxid|nationalid)$",
    ),
    _pattern(
        r"e mail|email(?: address)?",
        "email_address",
        compact=r"(?:email|emailaddress)$",
    ),
    _pattern(
        r"phone|mobile|telephone|fax",
        "phone_number",
        compact=r"(?:phone|phonenumber|mobile|telephone|fax|faxnumber)$",
    ),
    _pattern(
        r"date of birth|birth date|birthdate|dob",
        "date_of_birth",
        compact=r"(?:dateofbirth|birthdate|dob)$",
    ),
    _pattern(
        r"credit card|debit card|card number",
        "payment_card",
        compact=r"(?:creditcard|debitcard|cardnumber)$",
    ),
    _pattern(
        r"bank account(?: number)?|iban|swift code",
        "banking",
        compact=r"(?:bankaccount(?:number)?|iban|swiftcode)$",
    ),
    _pattern(
        r"password|passcode|pin code",
        "credential",
        compact=r"(?:password|passcode|pincode)$",
    ),
    _pattern(
        r"fingerprint|biometric|retina scan|face print",
        "biometric",
        compact=r"(?:fingerprint|biometric|retinascan|faceprint)$",
    ),
    _pattern(
        r"ip address|mac address",
        "online_identifier",
        compact=r"(?:ipaddress|macaddress)$",
    ),
)

CONDITIONAL_PII_PATTERNS = (
    _pattern(
        r"first name|last name|full name|middle name|maiden name",
        "person_name",
        compact=r"(?:firstname|lastname|fullname|middlename|maidenname|preferredname)$",
    ),
    _pattern(r"name", "person_name", compact=r"name$"),
    _pattern(
        r"address|street|postal code|zip code",
        "postal_address",
        compact=r"(?:address(?:line\d*)?|street|postalcode|zipcode)$",
    ),
    _pattern(
        r"age|gender|sex|marital status",
        "personal_attribute",
        compact=r"^(?:age|gender|sex|maritalstatus)$",
    ),
    _pattern(
        r"salary|income|credit score",
        "financial",
        compact=r"(?:salary|income|creditscore)$",
    ),
    _pattern(
        r"id|employee id|customer id|user id|contact id|account number",
        "identifier",
        compact=(
            r"(?:person|employee|customer|user|contact|account)id$"
            r"|accountnumber$|^id$"
        ),
    ),
)

PERSON_TABLE_PATTERN = re.compile(
    r"\b(?:person|people|customer|client|contact|user|employee|patient|"
    r"applicant|member|student|advisor|representative|household|account holder)s?\b"
)

NON_PERSON_TABLE_PATTERN = re.compile(
    r"\b(?:product|asset|strategy|platform|transaction type|business line|"
    r"configuration|audit|metric|aggregate|inventory|catalog)s?\b"
)

NON_PII_COLUMN_PATTERN = re.compile(
    r"\b(?:product name|account type|account status|employee count|"
    r"bank name|firm id|buying group name|city name|customer category name|"
    r"last password change|created at|updated at)\b"
)
NON_PII_COMPACT_COLUMN_PATTERN = re.compile(
    r"(?:productname|accounttype|accountstatus|employeecount|bankname|firmid|"
    r"buyinggroupname|cityname|customercategoryname|lastpasswordchange|"
    r"createdat|updatedat)$"
)


def evaluate_rules(column: ColumnInput) -> PiiDecision | None:
    """Return a high-confidence decision, or ``None`` when an LLM is needed."""

    column_name = normalize(column.column_name)
    compact_column_name = column_name.replace(" ", "")
    table_name = normalize(column.table_name)

    if not column_name:
        return PiiDecision(
            PiiStatus.REVIEW,
            None,
            0.0,
            "Column name is empty.",
            "rules",
        )

    if NON_PII_COLUMN_PATTERN.search(
        column_name
    ) or NON_PII_COMPACT_COLUMN_PATTERN.search(compact_column_name):
        return PiiDecision(
            PiiStatus.NOT_PII,
            None,
            0.96,
            "Column matches a known non-personal business field.",
            "rules",
        )

    for pattern in DIRECT_PII_PATTERNS:
        if _matches(pattern, column_name, compact_column_name):
            return PiiDecision(
                PiiStatus.PII,
                pattern.category,
                0.99,
                f"Column name matches the {pattern.category} rule.",
                "rules",
            )

    for pattern in CONDITIONAL_PII_PATTERNS:
        if not _matches(pattern, column_name, compact_column_name):
            continue
        if PERSON_TABLE_PATTERN.search(table_name):
            return PiiDecision(
                PiiStatus.PII,
                pattern.category,
                0.94,
                "Sensitive field appears in a table representing individuals.",
                "rules",
            )
        if NON_PERSON_TABLE_PATTERN.search(table_name):
            return PiiDecision(
                PiiStatus.NOT_PII,
                None,
                0.9,
                "Ambiguous field appears in a non-person table.",
                "rules",
            )

    return None
