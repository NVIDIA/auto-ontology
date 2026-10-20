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


# Role or channel tokens that may precede a PII field, e.g. customer_email.
# Longer alternatives come first so ``personal`` is not consumed as ``person``.
_FIELD_PREFIXES = (
    "preferred|personal|primary|secondary|alternate|business|billing|"
    "shipping|mailing|customer|employee|contact|patient|member|person|"
    "client|user|household|home|work|postal"
)

# Person-role tokens that may precede an identifier, e.g. customer_id.
# Home/work/billing are omitted so ``home_id`` is not treated as a person key.
_IDENTIFIER_PREFIXES = (
    "employee|customer|contact|patient|member|person|client|user|account"
)


def _pattern(
    words: str,
    category: str,
    *,
    compact: str,
    prefixes: str | None = _FIELD_PREFIXES,
) -> _Pattern:
    """Compile a rule that must describe the entire column name.

    Suffix and subphrase search auto-tagged flags such as ``no_email`` and
    derived fields such as ``email_address_hash``. Optional role or channel
    prefixes keep ``customer_email`` matching. Short tokens pass
    ``prefixes=None`` so ``user`` + ``name`` does not become a person name.
    """

    if prefixes:
        word_expr = rf"^(?:(?:{prefixes}) )*(?:{words})$"
        compact_expr = rf"^(?:{prefixes})*(?:{compact})$"
    else:
        word_expr = rf"^(?:{words})$"
        compact_expr = rf"^(?:{compact})$"
    return _Pattern(re.compile(word_expr), category, re.compile(compact_expr))


def _matches(pattern: _Pattern, normalized: str, compact: str) -> bool:
    return bool(
        pattern.expression.fullmatch(normalized)
        or (
            pattern.compact_expression is not None
            and pattern.compact_expression.fullmatch(compact)
        )
    )


DIRECT_PII_PATTERNS = (
    _pattern(
        r"social security(?: number)?",
        "government_id",
        compact=r"ssn|socialsecurity(?:number)?",
    ),
    _pattern(
        r"passport(?: number)?|driver(?:s)? license",
        "government_id",
        compact=r"passport(?:number)?|drivers?license",
    ),
    _pattern(
        r"taxpayer id|tax id|national id",
        "government_id",
        compact=r"taxpayerid|taxid|nationalid",
    ),
    _pattern(
        r"e mail|email address",
        "email_address",
        compact=r"emailaddress|email",
    ),
    _pattern(
        r"phone number|mobile phone|mobile number|cell phone|fax number",
        "phone_number",
        compact=(
            r"phonenumber|faxnumber|cellphone|mobilephone|telephone|"
            r"mobile|phone|fax|cell"
        ),
    ),
    _pattern(
        r"date of birth|birth date|birthdate|dob",
        "date_of_birth",
        compact=r"dateofbirth|birthdate|dob",
    ),
    _pattern(
        r"credit card|debit card|card number",
        "payment_card",
        compact=r"creditcard|debitcard|cardnumber",
    ),
    _pattern(
        r"bank account(?: number)?|iban|swift code",
        "banking",
        compact=r"bankaccount(?:number)?|iban|swiftcode",
    ),
    _pattern(
        r"passcode|pin code",
        "credential",
        compact=r"password|passcode|pincode",
    ),
    _pattern(
        r"fingerprint|biometric|retina scan|face print",
        "biometric",
        compact=r"fingerprint|biometric|retinascan|faceprint",
    ),
    _pattern(
        r"ip address|mac address",
        "online_identifier",
        compact=r"ipaddress|macaddress",
    ),
)

CONDITIONAL_PII_PATTERNS = (
    _pattern(
        r"first name|last name|full name|middle name|maiden name|preferred name",
        "person_name",
        compact=(r"firstname|lastname|fullname|middlename|maidenname|preferredname"),
    ),
    _pattern(
        r"name",
        "person_name",
        compact=r"name",
        prefixes=None,
    ),
    _pattern(
        r"address|street|postal code|zip code",
        "postal_address",
        compact=r"address(?:line\d*)?|street|postalcode|zipcode",
    ),
    _pattern(
        r"age|gender|sex|marital status",
        "personal_attribute",
        compact=r"age|gender|sex|maritalstatus",
        prefixes=None,
    ),
    _pattern(
        r"salary|income|credit score",
        "financial",
        compact=r"salary|income|creditscore",
        prefixes=None,
    ),
    _pattern(
        r"id|account number",
        "identifier",
        compact=r"id|accountnumber",
        prefixes=_IDENTIFIER_PREFIXES,
    ),
)

PERSON_TABLE_PATTERN = re.compile(
    r"^(?:persons?|people|customers?|clients?|contacts?|users?|employees?|"
    r"patients?|applicants?|members?|students?|advisors?|"
    r"representatives?|households?|account holders?)$"
)

NON_PERSON_TABLE_PATTERN = re.compile(
    r"^(?:products?|assets?|strateg(?:y|ies)|platforms?|transaction types?|"
    r"business lines?|configurations?|audits?|metrics?|aggregates?|"
    r"inventor(?:y|ies)|catalogs?)$"
)

NON_PII_COLUMN_PATTERN = re.compile(
    r"^(?:product name|account type|account status|employee count|"
    r"bank name|firm id|buying group name|city name|customer category name|"
    r"last password change|created at|updated at)$"
)
NON_PII_COMPACT_COLUMN_PATTERN = re.compile(
    r"^(?:productname|accounttype|accountstatus|employeecount|bankname|firmid|"
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

    if NON_PII_COLUMN_PATTERN.fullmatch(
        column_name
    ) or NON_PII_COMPACT_COLUMN_PATTERN.fullmatch(compact_column_name):
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
        if PERSON_TABLE_PATTERN.fullmatch(table_name):
            return PiiDecision(
                PiiStatus.PII,
                pattern.category,
                0.94,
                "Sensitive field appears in a table representing individuals.",
                "rules",
            )
        if NON_PERSON_TABLE_PATTERN.fullmatch(table_name):
            return PiiDecision(
                PiiStatus.NOT_PII,
                None,
                0.9,
                "Ambiguous field appears in a non-person table.",
                "rules",
            )

    return None
