# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""When two SQL expressions count as the same expression."""

from __future__ import annotations

import sqlglot

# Tokenizing (e.g. an unterminated string) fails separately from parsing.
SQL_ERRORS = (sqlglot.errors.ParseError, sqlglot.errors.TokenError)


def sql_equivalence_key(expression: str) -> str:
    """A key equal for expressions that differ only in formatting or case.

    Whitespace, keyword case and unquoted identifier case are ignored; string
    literals and quoted identifiers are kept, since ``status = 'Open'`` and
    ``status = 'OPEN'`` can select different rows. SQL that does not parse
    falls back to collapsing whitespace and case over the whole text.
    """
    try:
        return sqlglot.parse_one(expression).sql(normalize=True)
    except SQL_ERRORS:
        return " ".join(expression.split()).lower()
