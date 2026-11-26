# SPDX-FileCopyrightText: Copyright (c) 2026 Jeremy Daly
# SPDX-License-Identifier: Apache-2.0

"""Dialect-aware row limits for SQL the backend writes itself.

Profiling and join probes are built as strings and need a row cap. Most engines
spell it ``LIMIT n``, but not all: Oracle writes ``FETCH FIRST n ROWS ONLY`` and
SQL Server ``SELECT TOP n``. A hard-coded ``LIMIT`` on those engines fails, and
the probes treat a failure as "no profile available", so the damage is a quietly
thinner ontology rather than an error.

sqlglot decides which spelling a dialect uses. For every dialect that spells it
``LIMIT``, the SQL is returned exactly as it was always built, so those
connectors see no change. Only the others go through a sqlglot round trip.
"""

from __future__ import annotations

from functools import lru_cache

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError


@lru_cache(maxsize=None)
def _spells_limit(dialect: str | None) -> bool:
    """Whether *dialect* caps rows with a trailing ``LIMIT n``.

    An unknown dialect name counts as ``LIMIT``: that is what callers emitted
    before this module existed, and sqlglot could not rewrite it anyway.
    """
    try:
        probe = exp.select("x").from_("t").limit(5).sql(dialect=dialect)
    except (SqlglotError, ValueError):
        return True
    return probe.rstrip().upper().endswith("LIMIT 5")


def with_row_limit(sql: str, limit: int, dialect: object = None) -> str:
    """Return *sql* capped at *limit* rows in *dialect*'s syntax.

    *sql* is a single ``SELECT`` with no row cap of its own, with identifiers
    already quoted for *dialect*. *dialect* is typed loosely because callers
    pass ``getattr(connector, "dialect", None)``, and a non-string (a test
    double's attribute, say) is treated as no dialect.

    Raises:
        ValueError: *dialect* needs a rewrite and sqlglot cannot parse *sql*.
            Appending ``LIMIT`` instead would send SQL the engine rejects.
    """
    name = dialect if isinstance(dialect, str) and dialect else None
    if _spells_limit(name):
        return f"{sql} LIMIT {int(limit)}"
    try:
        return sqlglot.parse_one(sql, read=name).limit(int(limit)).sql(dialect=name)
    except SqlglotError as exc:
        raise ValueError(f"Cannot add a {name} row limit to the query: {exc}") from exc
