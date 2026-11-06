# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Link key columns that hold the same identifier.

One code can key several tables: a table of active depots and a table of every
depot ever opened are both keyed by depot code. Each key owns its own
attribute, so a column referencing one of them reaches only that table, and the
two keys are never joined to each other. A query that needs the other table has
no join to follow.

Algorithm
---------
1. **Keys.** The key-like attribute owners found by value-overlap profiling,
   with their distinct values from a bounded sample.
2. **Pairs.** Two keys on different tables, of the same type family, whose
   shared values make up at least ``_MIN_CONTAINED_SHARE`` of the smaller key.
3. **Decision.** The LLM confirms that a pair holds the same identifier; dates,
   periods and counters that only coincide are rejected.
4. **Key links.** For each pair of tables one confirmed pair is kept, and the
   key whose values sit inside the other gets a semantic FK to the other's
   attribute.
5. **Reference links.** A column referencing one key of a confirmed pair also
   references the other table's key when ``_MIN_REFERENCE_SHARE`` of its
   values are in it, unless it already joins that table. A column then has a join to each table keyed by
   its codes; :func:`~auto_ontology.semantic.semantic_fk.select_semantic_fk_join_defaults`
   still keeps one default per source table and target.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from auto_ontology.connectors.base import SQLDatabase
from auto_ontology.dal.attributes import (
    fetch_semantic_fk_join_groups,
    merge_semantic_fk,
)
from auto_ontology.dal.datasources import fetch_columns_for_value_overlap
from auto_ontology.semantic.models import KeyEquivalenceJudgement
from auto_ontology.semantic.value_overlap_fk import (
    ProfiledColumn,
    describe_column,
    overlap_accepted,
    profile_tables,
    sampled_values,
    shared_value_counts,
)
from auto_ontology.utils.llm_invoke import (
    get_llm_client,
    invoke_with_structured_output,
)

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get(
    "SEMANTIC_FK_EQUIVALENT_KEYS", "true"
).strip().lower() not in {
    "0",
    "false",
    "no",
}
_MIN_CONTAINED_SHARE = float(os.environ.get("SEMANTIC_FK_EQUIVALENT_KEY_SHARE", "0.6"))
_MIN_SHARED_VALUES = 5
_MIN_REFERENCE_SHARE = float(
    os.environ.get("SEMANTIC_FK_EQUIVALENT_REFERENCE_SHARE", "0.5")
)
_KEY_VALUE_ROWS = 50_000
_WORKERS = 2
_INTEGER_TYPES = frozenset(
    {
        "int",
        "integer",
        "tinyint",
        "smallint",
        "mediumint",
        "bigint",
        "int2",
        "int4",
        "int8",
    }
)

_SYSTEM_PROMPT = """\
You are a database schema expert. Two key columns on different tables share \
many values. Decide whether they hold the same identifier: the same kind of \
thing named by the same codes, so that a column holding those codes could join \
to either table.

Rules:
- True for one code kept by two tables, such as a depot code in a table of \
active depots and in a table of every depot ever opened.
- False for dates, times, periods, amounts, counts, sequence numbers and other \
values that only coincide.
- False when the columns identify different things whose codes look alike, \
such as a depot code and a region code.
- Do NOT guess. If unsure, return false.
"""


@dataclass(frozen=True)
class KeyPair:
    """Two keys sharing values; *inner*'s values sit more fully inside *outer*'s."""

    inner: ProfiledColumn
    outer: ProfiledColumn
    shared: int
    inner_values: int
    outer_values: int

    @property
    def contained_share(self) -> float:
        return self.shared / self.inner_values


def link_equivalent_keys(database_name: str, connector: SQLDatabase | None) -> int:
    """Create semantic FKs between keys holding the same identifier, and from
    the columns referencing them.

    Returns the number of edges written.
    """
    if not _ENABLED:
        logger.info("equivalent-key linking disabled (SEMANTIC_FK_EQUIVALENT_KEYS)")
        return 0
    if connector is None:
        logger.info(
            "equivalent-key linking skipped: no connector for %s", database_name
        )
        return 0

    tables = fetch_columns_for_value_overlap(database_name)
    keys, _ = profile_tables(tables, connector)
    values: dict[str, set[str]] = {}
    for key in keys:
        try:
            values[key.id] = sampled_values(
                connector,
                key.schema_name,
                key.table_name,
                key.name,
                rows=_KEY_VALUE_ROWS,
            )
        except Exception:
            logger.info(
                "equivalent keys: could not read %s.%s",
                key.table_name,
                key.name,
                exc_info=True,
            )

    candidates = candidate_pairs(keys, values)
    if not candidates:
        logger.info("equivalent-key linking: no key pairs share enough values")
        return 0
    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        verdicts = list(pool.map(_confirmed, candidates))
    confirmed = [pair for pair, ok in zip(candidates, verdicts) if ok]
    logger.info(
        "equivalent-key linking: %d of %d candidate pair(s) confirmed",
        len(confirmed),
        len(candidates),
    )
    if not confirmed:
        return 0

    existing = fetch_semantic_fk_join_groups(database_name)
    joined: dict[str, set[str]] = {}
    for row in existing:
        joined.setdefault(row["source_column_id"], set()).add(row["target_table_id"])

    pk = {(table["id"], name.casefold()) for table in tables for name in table["pk"]}
    canonical = canonical_pairs(confirmed, pk)
    written = 0
    for pair in canonical.values():
        written += _link_keys(pair, joined)
    attribute_tables = {
        table["id"]
        for table in tables
        if any(col["attribute_id"] for col in table["columns"])
    }
    written += _link_references(
        existing, confirmed, canonical, attribute_tables, joined, connector
    )
    logger.info("equivalent-key linking: %d semantic FK edge(s) written", written)
    return written


def is_integer_type(data_type: str) -> bool:
    return (data_type or "").casefold().split("(")[0].strip() in _INTEGER_TYPES


def candidate_pairs(
    keys: list[ProfiledColumn], values: dict[str, set[str]]
) -> list[KeyPair]:
    """Keys on different tables whose shared values cover most of the smaller one."""
    pairs: list[KeyPair] = []
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            if a.table_id == b.table_id or a.attribute_id == b.attribute_id:
                continue
            if is_integer_type(a.data_type) != is_integer_type(b.data_type):
                continue
            va, vb = values.get(a.id), values.get(b.id)
            if not va or not vb:
                continue
            shared = len(va & vb)
            if shared < _MIN_SHARED_VALUES:
                continue
            if shared < _MIN_CONTAINED_SHARE * min(len(va), len(vb)):
                continue
            inner, outer = (a, b) if (len(va), a.id) <= (len(vb), b.id) else (b, a)
            pairs.append(
                KeyPair(
                    inner=inner,
                    outer=outer,
                    shared=shared,
                    inner_values=len(values[inner.id]),
                    outer_values=len(values[outer.id]),
                )
            )
    return pairs


def canonical_pairs(
    confirmed: list[KeyPair], pk: set[tuple[str, str]]
) -> dict[frozenset[str], KeyPair]:
    """The one confirmed pair that links each pair of tables.

    Preferred in order: the higher contained share, columns sharing a name,
    columns in the tables' primary keys.
    """

    def rank(pair: KeyPair) -> tuple[Any, ...]:
        same_name = pair.inner.name.casefold() == pair.outer.name.casefold()
        in_pk = sum(
            (col.table_id, col.name.casefold()) in pk
            for col in (pair.inner, pair.outer)
        )
        return (
            -pair.contained_share,
            not same_name,
            -in_pk,
            pair.inner.id,
            pair.outer.id,
        )

    best: dict[frozenset[str], KeyPair] = {}
    for pair in sorted(confirmed, key=rank):
        best.setdefault(frozenset((pair.inner.table_id, pair.outer.table_id)), pair)
    return best


def _confirmed(pair: KeyPair) -> bool:
    try:
        return _llm_confirms(pair)
    except Exception:
        logger.exception(
            "equivalent keys: LLM failed for %s.%s ~ %s.%s",
            pair.inner.table_name,
            pair.inner.name,
            pair.outer.table_name,
            pair.outer.name,
        )
        return False


def _llm_confirms(pair: KeyPair) -> bool:
    human = (
        f"Column A: {describe_column(pair.inner)} | {pair.inner_values} distinct values\n"
        f"Column B: {describe_column(pair.outer)} | {pair.outer_values} distinct values\n"
        f"They share {pair.shared} values: "
        f"{100 * pair.shared / pair.inner_values:.0f}% of A's and "
        f"{100 * pair.shared / pair.outer_values:.0f}% of B's.\n\n"
        "Do A and B hold the same identifier?"
    )
    result = invoke_with_structured_output(
        get_llm_client(max_tokens=4096),
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human)],
        KeyEquivalenceJudgement,
    )
    return bool(result is not None and result.equivalent)


def _link_keys(pair: KeyPair, joined: dict[str, set[str]]) -> int:
    if pair.outer.table_id in joined.get(pair.inner.id, set()):
        return 0
    if not pair.outer.attribute_id or not merge_semantic_fk(
        pair.inner.id, pair.outer.attribute_id
    ):
        return 0
    joined.setdefault(pair.inner.id, set()).add(pair.outer.table_id)
    logger.info(
        "resolve_semantic_fks [equivalent keys]: %s.%s -> %s.%s "
        "(%d of %d values shared)",
        pair.inner.table_name,
        pair.inner.name,
        pair.outer.table_name,
        pair.outer.name,
        pair.shared,
        pair.inner_values,
    )
    return 1


def _link_references(
    existing: list[dict[str, Any]],
    confirmed: list[KeyPair],
    canonical: dict[frozenset[str], KeyPair],
    attribute_tables: set[str],
    joined: dict[str, set[str]],
    connector: SQLDatabase,
) -> int:
    """Point columns referencing one key of a pair at the other table's key too.

    A table none of whose columns owns an attribute is a pure association of
    references; it keeps exactly one target per column.
    """
    partner_tables: dict[str, set[str]] = {}
    for pair in confirmed:
        partner_tables.setdefault(pair.inner.id, set()).add(pair.outer.table_id)
        partner_tables.setdefault(pair.outer.id, set()).add(pair.inner.table_id)

    written = 0
    seen: set[tuple[str, str]] = set()
    for row in existing:
        source_id, key_id = row["source_column_id"], row["target_column_id"]
        if (source_id, key_id) in seen:
            continue
        seen.add((source_id, key_id))
        if row["source_table_id"] not in attribute_tables:
            continue
        for table_id in sorted(partner_tables.get(key_id, ())):
            if table_id == row["source_table_id"] or table_id in joined.get(
                source_id, set()
            ):
                continue
            pair = canonical[frozenset((table_id, row["target_table_id"]))]
            target = pair.inner if pair.inner.table_id == table_id else pair.outer
            source_values, shared = _overlap(row, target, connector)
            if not _references(source_values, shared) or not target.attribute_id:
                continue
            if merge_semantic_fk(source_id, target.attribute_id):
                joined.setdefault(source_id, set()).add(table_id)
                written += 1
                logger.info(
                    "resolve_semantic_fks [equivalent key reference]: "
                    "%s.%s -> %s.%s (%d of %d values shared; also references %s.%s)",
                    row["source_table_name"],
                    row["source_column_name"],
                    target.table_name,
                    target.name,
                    shared,
                    source_values,
                    row["target_table_name"],
                    row["target_column_name"],
                )
    return written


def _references(source_values: int, shared: int) -> bool:
    """Whether a column holding *source_values* codes, *shared* of them in a
    key, references that key.

    Stricter than a first link: the column already has a target, and a key
    holding only some of its codes names a narrower set of things.
    """
    return (
        overlap_accepted(source_values, shared)
        and shared >= _MIN_REFERENCE_SHARE * source_values
    )


def _overlap(
    row: dict[str, Any], target: ProfiledColumn, connector: SQLDatabase
) -> tuple[int, int]:
    try:
        return shared_value_counts(
            connector,
            (
                row.get("source_schema_name"),
                row["source_table_name"],
                row["source_column_name"],
            ),
            (target.schema_name, target.table_name, target.name),
        )
    except Exception:
        logger.info(
            "equivalent keys: overlap check skipped for %s.%s",
            row["source_table_name"],
            row["source_column_name"],
            exc_info=True,
        )
        return 0, 0
