# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classify each table's role and grain, and note near-identical tables.

SQL generation sees a flat list of tables. Two things it cannot read off that
list decide many wrong queries:

* **Role.** A dimension (one row per customer, product or code) is where a
  name or label lives; a fact (orders, payments, shipments, summaries)
  references dimensions. Joining a fact to a dimension is how a name is
  reached. Joining two facts to each other on a shared dimension key repeats
  rows, because neither row identifies the other.
* **Siblings.** A warehouse often holds several tables with near-identical
  columns — a detail table and its summary, a current and an all-history
  copy, two feeds of one source. Both join the same way, so nothing in the
  join graph says which one a question means.

Algorithm
---------
1. **Role and grain.** One LLM call per non-legacy table, given its columns,
   the tables it references and the tables that reference it. A table the
   compiler already marked as a junction stays a junction.
2. **Sibling pairs.** Tables of one schema pair up when they share at least
   ``_MIN_SHARED_COLUMNS`` columns and a column-name Jaccard similarity of
   ``_MIN_SIBLING_SIMILARITY`` — or ``_MIN_NAMED_SIBLING_SIMILARITY`` when the
   table names also share a token. Columns present on a large share of the
   database's tables (load dates and the like) are ignored. Each table keeps at
   most ``_MAX_SIBLINGS_PER_TABLE`` siblings, the most similar first.
3. **Sibling notes.** One LLM call per pair writes, for each side, when to use
   it rather than the other. A pair the model says cannot be confused is
   dropped.

Every compile replaces the database's previous roles and notes, except where a
call failed outright: that table or pair keeps what an earlier compile stored.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from auto_ontology.dal.datasources import (
    fetch_tables_for_role_classification,
    set_table_roles,
    set_table_siblings,
)
from auto_ontology.semantic.models import SiblingTableNotes, TableRoleJudgement
from auto_ontology.utils.llm_invoke import (
    get_llm_client,
    invoke_with_structured_output,
)

logger = logging.getLogger(__name__)

_MIN_SHARED_COLUMNS = 3
_MIN_SIBLING_SIMILARITY = 0.3
_MIN_NAMED_SIBLING_SIMILARITY = 0.15
_MIN_NAME_TOKEN_OVERLAP = 0.5
_MAX_SIBLINGS_PER_TABLE = 4
#: A column on more than this share of tables says nothing about identity.
_UBIQUITOUS_COLUMN_SHARE = 0.3
#: ...but only past this many tables, so a family of near-identical copies in
#: a small database is not mistaken for housekeeping.
_MIN_UBIQUITOUS_TABLES = 8
_MAX_PROMPT_COLUMNS = 60
_MAX_SAMPLE_VALUES = 5
#: Below the compile-wide inflight bound: a call that exhausts its retries on
#: a saturated endpoint leaves its table unclassified until the next compile.
_WORKERS = int(os.environ.get("TABLE_ROLE_WORKERS", "3"))

_ROLE_SYSTEM_PROMPT = """\
You classify one table of a data warehouse so a SQL writer knows how to use it.

Return its role:
- dimension: one row per entity or code (a customer, a product, a region, a
  calendar date, a status code). Other tables reference it to get a name,
  label or attribute. Mostly descriptive columns.
- fact: rows record events, transactions, orders, payments, assignments,
  measurements or aggregates. It usually references several other tables and
  often carries counts, amounts or dates of what happened. A summary or
  aggregated table is a fact.
- junction: each row only associates two or more entities, with at most a few
  properties of the association.
Return null only when the evidence does not support any of them.

Also return the grain: one short sentence naming what a single row represents,
naming the columns that identify a row when they are clear.
"""

_SIBLING_SYSTEM_PROMPT = """\
You compare two tables of one data warehouse that hold near-identical columns,
so a SQL writer can choose between them.

For each table write one short sentence saying when to use it rather than the
other: what its rows cover that the other's do not (detail vs summary, all
history vs current only, a different source or audience, a different grain).
Use only the evidence given. When the two tables answer different questions
and cannot be confused, return null for both.
"""


@dataclass(frozen=True)
class SiblingPair:
    """Two tables of one schema with near-identical columns."""

    first: dict[str, Any]
    second: dict[str, Any]
    similarity: float
    shared: tuple[str, ...]
    first_only: tuple[str, ...]
    second_only: tuple[str, ...]


def classify_table_roles(database_name: str) -> tuple[int, int]:
    """Classify *database_name*'s tables and note their siblings.

    Returns ``(tables classified, sibling pairs noted)``.
    """
    tables = fetch_tables_for_role_classification(database_name)
    if not tables:
        set_table_roles(database_name, {})
        set_table_siblings(database_name, [])
        return 0, 0

    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        judged = list(pool.map(_classify_one, tables))
    roles = {
        table["id"]: decision
        for table, (_, decision) in zip(tables, judged)
        if decision is not None
    }
    failed_tables = [table["id"] for table, (ok, _) in zip(tables, judged) if not ok]
    set_table_roles(database_name, roles, keep=failed_tables)
    logger.info(
        "classify_table_roles: %d of %d table(s) classified in %s (%s); "
        "%d call(s) failed and keep their stored role",
        len(roles),
        len(tables),
        database_name,
        dict(Counter(d["role"] or "none" for d in roles.values())),
        len(failed_tables),
    )

    pairs = find_sibling_pairs(tables)
    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        noted = list(pool.map(lambda pair: _note_pair(pair, roles), pairs))
    siblings: list[dict[str, str]] = []
    failed_pairs: list[tuple[str, str]] = []
    noted_pairs = 0
    for pair, (ok, notes) in zip(pairs, noted):
        if not ok:
            failed_pairs.append((pair.first["id"], pair.second["id"]))
            continue
        if notes is None:
            continue
        noted_pairs += 1
        first_note, second_note = notes
        if first_note:
            siblings.append(
                {
                    "table_id": pair.first["id"],
                    "sibling_table_id": pair.second["id"],
                    "note": first_note,
                }
            )
        if second_note:
            siblings.append(
                {
                    "table_id": pair.second["id"],
                    "sibling_table_id": pair.first["id"],
                    "note": second_note,
                }
            )
        logger.info(
            "classify_table_roles: %s / %s are siblings",
            pair.first["name"],
            pair.second["name"],
        )
    set_table_siblings(database_name, siblings, keep=failed_pairs)
    logger.info(
        "classify_table_roles: %d of %d candidate sibling pair(s) noted in %s; "
        "%d call(s) failed and keep their stored notes",
        noted_pairs,
        len(pairs),
        database_name,
        len(failed_pairs),
    )
    return len(roles), noted_pairs


# ---------------------------------------------------------------------------
# Role and grain
# ---------------------------------------------------------------------------


def _column_line(column: dict[str, Any]) -> str:
    line = f"  - {column['name']} ({column.get('data_type') or ''})"
    if column.get("description"):
        line += f" — {column['description']}"
    samples = [str(v) for v in column.get("sample_values") or []][:_MAX_SAMPLE_VALUES]
    if samples:
        line += f" [samples: {', '.join(samples)}]"
    return line


def _describe_table(table: dict[str, Any], role: dict[str, Any] | None = None) -> str:
    columns = table.get("columns") or []
    lines = [
        f"Table: {table['name']}",
        f"Description: {table.get('description') or '(none)'}",
        f"Primary key: {', '.join(table.get('pk') or []) or '(none)'}",
        f"References: {', '.join(table.get('references') or []) or '(none)'}",
        f"Referenced by: {', '.join(table.get('referenced_by') or []) or '(none)'}",
    ]
    if role is not None:
        lines.append(f"Role: {role.get('role') or '(unknown)'}")
        lines.append(f"Grain: {role.get('grain') or '(unknown)'}")
    lines.append(f"Columns ({len(columns)}):")
    lines.extend(_column_line(c) for c in columns[:_MAX_PROMPT_COLUMNS])
    if len(columns) > _MAX_PROMPT_COLUMNS:
        lines.append(f"  … {len(columns) - _MAX_PROMPT_COLUMNS} more")
    return "\n".join(lines)


def _classify_one(table: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
    """``(answered, {"role", "grain"})`` for one table.

    The decision is ``None`` when nothing is known. ``answered`` is false when
    the call itself failed, so the caller can keep what an earlier compile
    stored.
    """
    try:
        result = invoke_with_structured_output(
            get_llm_client(temperature=0.0, max_tokens=512),
            [
                SystemMessage(content=_ROLE_SYSTEM_PROMPT),
                HumanMessage(content=_describe_table(table)),
            ],
            TableRoleJudgement,
        )
    except Exception:
        logger.warning(
            "classify_table_roles: role call failed for %s",
            table["name"],
            exc_info=True,
        )
        result = None
    if result is None:
        return False, None
    role = result.role
    grain = result.grain.strip()
    if table.get("is_junction_table"):
        role = "junction"
    if role is None and not grain:
        return True, None
    return True, {"role": role, "grain": grain}


# ---------------------------------------------------------------------------
# Sibling tables
# ---------------------------------------------------------------------------


def _name_tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[^A-Za-z0-9]+", name.upper()) if t}


def _ubiquitous_columns(tables: list[dict[str, Any]]) -> set[str]:
    counts = Counter(
        name
        for table in tables
        for name in {str(c["name"]).upper() for c in table.get("columns") or []}
    )
    limit = max(_MIN_UBIQUITOUS_TABLES, _UBIQUITOUS_COLUMN_SHARE * len(tables))
    return {name for name, count in counts.items() if count > limit}


def find_sibling_pairs(tables: list[dict[str, Any]]) -> list[SiblingPair]:
    """Pairs of tables in one schema whose columns are near-identical."""
    ignored = _ubiquitous_columns(tables)
    columns = {
        table["id"]: {str(c["name"]).upper() for c in table.get("columns") or []}
        - ignored
        for table in tables
    }
    candidates: list[SiblingPair] = []
    for first, second in combinations(tables, 2):
        if (first.get("schema_name") or "") != (second.get("schema_name") or ""):
            continue
        a, b = columns[first["id"]], columns[second["id"]]
        shared = a & b
        if len(shared) < _MIN_SHARED_COLUMNS:
            continue
        similarity = len(shared) / len(a | b)
        ta, tb = _name_tokens(first["name"]), _name_tokens(second["name"])
        named = (
            bool(ta and tb)
            and len(ta & tb) / min(len(ta), len(tb)) >= _MIN_NAME_TOKEN_OVERLAP
        )
        threshold = _MIN_NAMED_SIBLING_SIMILARITY if named else _MIN_SIBLING_SIMILARITY
        if similarity < threshold:
            continue
        candidates.append(
            SiblingPair(
                first=first,
                second=second,
                similarity=similarity,
                shared=tuple(sorted(shared)),
                first_only=tuple(sorted(a - b)),
                second_only=tuple(sorted(b - a)),
            )
        )

    candidates.sort(key=lambda p: (-p.similarity, p.first["name"], p.second["name"]))
    kept: list[SiblingPair] = []
    per_table: Counter[str] = Counter()
    for pair in candidates:
        ids = (pair.first["id"], pair.second["id"])
        if any(per_table[i] >= _MAX_SIBLINGS_PER_TABLE for i in ids):
            continue
        per_table.update(ids)
        kept.append(pair)
    return kept


def _only_line(label: str, names: tuple[str, ...]) -> str:
    shown = ", ".join(names[:_MAX_PROMPT_COLUMNS]) or "(none)"
    more = len(names) - _MAX_PROMPT_COLUMNS
    return f"{label}: {shown}" + (f" … {more} more" if more > 0 else "")


def _note_pair(
    pair: SiblingPair, roles: dict[str, dict[str, Any]]
) -> tuple[bool, tuple[str | None, str | None] | None]:
    """``(answered, (first note, second note))`` for one pair.

    The notes are ``None`` when the pair is not confusable. ``answered`` is
    false when the call itself failed.
    """
    prompt = "\n\n".join(
        [
            "FIRST TABLE\n"
            + _describe_table(pair.first, roles.get(pair.first["id"]) or {}),
            "SECOND TABLE\n"
            + _describe_table(pair.second, roles.get(pair.second["id"]) or {}),
            f"Shared columns: {len(pair.shared)} "
            f"(name similarity {pair.similarity:.0%})\n"
            + _only_line("Columns only in the first", pair.first_only)
            + "\n"
            + _only_line("Columns only in the second", pair.second_only),
        ]
    )
    try:
        result = invoke_with_structured_output(
            get_llm_client(temperature=0.0, max_tokens=512),
            [
                SystemMessage(content=_SIBLING_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ],
            SiblingTableNotes,
        )
    except Exception:
        logger.warning(
            "classify_table_roles: sibling call failed for %s / %s",
            pair.first["name"],
            pair.second["name"],
            exc_info=True,
        )
        return False, None
    if result is None:
        return False, None
    first = (result.first_note or "").strip() or None
    second = (result.second_note or "").strip() or None
    if first is None and second is None:
        return True, None
    return True, (first, second)
