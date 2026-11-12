# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Talking to Apache AGE inside the Auto Ontology Postgres database.

AGE is a Postgres extension, so the graph lives in the same database as the
catalog it is projected from, reached through the same pooled engine. Three
things about it shape every call here:

* **It is loaded per session, not preloaded.** ``LOAD 'age'`` is issued on every
  transaction that touches the graph; it is a no-op once the library is in the
  backend, and it means no ``shared_preload_libraries`` change is required.
* **Its functions resolve through ``search_path``.** ``UNWIND`` compiles to an
  unqualified ``age_unnest`` call, so qualifying ``ag_catalog.cypher`` is not
  enough. The path is set with ``SET LOCAL`` so it reverts when the transaction
  ends and the ``public`` pin in :mod:`auto_ontology.dal.session` is intact for
  the next borrower of the pooled connection.
* **Parameters are one agtype map**, bound as a real server-side parameter.
  Values never become Cypher text, so quoting in names and descriptions cannot
  change a query.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import Connection
from sqlalchemy.engine import Engine

ENABLED_ENV = "AGE_GRAPH_ENABLED"
GRAPH_NAME_ENV = "AGE_GRAPH_NAME"
DEFAULT_GRAPH_NAME = "auto_ontology"

_GRAPH_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
_DOLLAR_TAG = "$cypher$"


class AgeUnavailable(RuntimeError):
    """The database has no usable Apache AGE extension."""


class GraphNotBuilt(RuntimeError):
    """AGE is available but the graph has not been synced yet."""


def enabled() -> bool:
    """Whether the AGE projection is switched on for this deployment."""
    return os.environ.get(ENABLED_ENV, "").strip().lower() in {"1", "true", "yes"}


def graph_name() -> str:
    """The configured graph name, validated because it is spliced into SQL."""
    name = os.environ.get(GRAPH_NAME_ENV, DEFAULT_GRAPH_NAME).strip()
    if not _GRAPH_NAME.match(name):
        raise ValueError(
            f"{GRAPH_NAME_ENV}={name!r} is not a valid graph name "
            "(lowercase letters, digits and underscores, starting with a letter)"
        )
    return name


def identifier(value: str) -> str:
    """A label or column alias, checked before it is written into a statement."""
    if not _IDENTIFIER.match(value):
        raise ValueError(f"{value!r} is not a valid identifier")
    return value


def prepare(conn: Connection) -> None:
    """Load AGE and put ``ag_catalog`` on the path for this transaction only."""
    try:
        conn.exec_driver_sql("LOAD 'age'")
    except Exception as exc:  # noqa: BLE001 — re-raised with the fix attached
        raise AgeUnavailable(
            "Apache AGE is not installed in this Postgres server. Use an image "
            "that ships it (for example apache/age) or install the extension, "
            "then run CREATE EXTENSION age."
        ) from exc
    conn.exec_driver_sql('SET LOCAL search_path = ag_catalog, "$user", public')


def ensure_extension(conn: Connection) -> None:
    """``CREATE EXTENSION age`` if it is installable but not yet created."""
    conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS age")


def graph_exists(conn: Connection, name: str) -> bool:
    row = conn.exec_driver_sql(
        "SELECT 1 FROM ag_catalog.ag_graph WHERE name = %s", (name,)
    ).first()
    return row is not None


@contextmanager
def read_transaction(engine: Engine) -> Iterator[Connection]:
    """A read-only transaction that is always rolled back.

    ``READ ONLY`` makes Postgres refuse a Cypher ``CREATE``/``SET``/``DELETE``
    (AGE writes through the executor, which the flag guards), and the rollback
    means nothing survives even if a future AGE release stops honouring it.
    """
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            conn.exec_driver_sql("SET TRANSACTION READ ONLY")
            prepare(conn)
            yield conn
        finally:
            trans.rollback()


def to_agtype_value(value: Any) -> Any:
    """Coerce a Postgres value into something JSON (and so agtype) can carry."""
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, list | tuple):
        return [to_agtype_value(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): to_agtype_value(v) for k, v in value.items()}
    return value


def cypher(
    conn: Connection,
    graph: str,
    query: str,
    params: Mapping[str, Any] | None = None,
    columns: Sequence[str] = ("v",),
) -> list[tuple[Any, ...]]:
    """Run one Cypher statement and return its rows, each value parsed."""
    if _DOLLAR_TAG in query:
        raise ValueError(f"Cypher text may not contain {_DOLLAR_TAG}")
    if not _GRAPH_NAME.match(graph):
        raise ValueError(f"{graph!r} is not a valid graph name")
    projection = ", ".join(f"{identifier(c)} agtype" for c in columns)
    if params is None:
        sql = f"SELECT * FROM cypher('{graph}', {_DOLLAR_TAG} {query} {_DOLLAR_TAG}) AS ({projection})"
        result = conn.exec_driver_sql(sql)
    else:
        # With bind parameters the driver treats `%` as a placeholder marker,
        # so a literal one in the Cypher text has to be doubled.
        body = query.replace("%", "%%")
        sql = f"SELECT * FROM cypher('{graph}', {_DOLLAR_TAG} {body} {_DOLLAR_TAG}, %s) AS ({projection})"
        payload = json.dumps(to_agtype_value(dict(params)))
        result = conn.exec_driver_sql(sql, (payload,))
    return [tuple(parse_agtype(v) for v in row) for row in result]


_TYPE_SUFFIXES = ("::vertex", "::edge", "::path", "::numeric")


def parse_agtype(text: str | None) -> Any:
    """Parse agtype's text form into Python values.

    agtype is JSON plus a ``::vertex``/``::edge``/``::path``/``::numeric``
    annotation after a value. The annotations are removed by a scanner that
    skips string literals, so a description containing ``}::vertex`` is left as
    written.
    """
    if text is None:
        return None
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == '"':
                    break
                j += 1
            out.append(text[i : j + 1])
            i = j + 1
            continue
        if ch == ":":
            suffix = next((s for s in _TYPE_SUFFIXES if text.startswith(s, i)), None)
            if suffix is not None:
                i += len(suffix)
                continue
        out.append(ch)
        i += 1
    return json.loads("".join(out))
