# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dialect-aware quoting of SQL identifiers.

Shared by the connectors (which build fully-qualified references for metadata
queries) and by the semantic layer's profiling probes, so both quote names the
same way.
"""

from __future__ import annotations

from functools import lru_cache

import sqlglot
from sqlglot import exp
from sqlglot.dialects.dialect import Dialect


def quoted_identifier(name: str, dialect: str | None = None) -> str:
    """Quote a catalog, schema, table or column name for *dialect*.

    Names carrying a space or a reserved word have to be quoted or the probe
    silently loses the table: ``SELECT * FROM main.Sales Orders`` parses as
    table ``main.Sales``, raises "no such table", and the caller drops every
    column of that table from profiling. The quote character is dialect-specific
    (backticks on MySQL and Spark, double quotes elsewhere), so the naive
    hard-coded ``"`` would trade a SQLite bug for a MySQL one.
    """
    try:
        return exp.to_identifier(name, quoted=True).sql(dialect=dialect or None)
    except Exception:
        # Unknown dialect: fall back to the SQL-standard quote rather than
        # emitting a bare identifier, since bare is what breaks on spaces.
        return '"' + name.replace('"', '""') + '"'


def qualified_name(*parts: str | None, dialect: str | None = None) -> str:
    """Join *parts* into a dotted, quoted reference, skipping empty parts.

    ``qualified_name("nvapp", "raw", "t", dialect="spark")`` ->
    ```nvapp`.`raw`.`t```.
    """
    return ".".join(quoted_identifier(p, dialect) for p in parts if p)


@lru_cache(maxsize=4096)
def needs_quoting(name: str, dialect: str | None = None) -> bool:
    """Whether *name* only means itself in *dialect* when quoted.

    True when sqlglot would quote it (a hyphen, a space, a leading digit), when
    the dialect would fold its case, or when it does not parse back as a plain
    column name (a reserved word).
    """
    try:
        target = Dialect.get_or_raise(dialect)
    except ValueError:
        target = Dialect()
    if exp.to_identifier(name).sql(dialect=target) != name:
        return True
    if target.normalize_identifier(exp.to_identifier(name)).name != name:
        return True
    try:
        column = sqlglot.parse_one(f"SELECT {name}", read=target).selects[0]
    except sqlglot.errors.SqlglotError:
        return True
    return not (
        isinstance(column, exp.Column) and column.name == name and not column.table
    )


def quoted_if_needed(name: str, dialect: str | None = None) -> str:
    """*name* as it has to be written in SQL: bare when that is safe."""
    return quoted_identifier(name, dialect) if needs_quoting(name, dialect) else name
