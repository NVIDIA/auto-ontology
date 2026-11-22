# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merge the Concepts of tables that are parts of one entity.

Runs after term families and before semantic FKs. A Term has a one-member
Concept once its table has a key (``dal.concepts.ensure_term_concept``), and
this stage creates the ones it needs; it folds two of them into one when the
tables are a vertical split of the same entity — ``customer`` and
``customer_ext``, one row per customer in each, sharing ``customer_id``.

Two tables are merged only when every check passes:

1. They are in the same term family, or both in none — a merged Concept never
   mixes ``OLD_`` and ``NEW_`` copies of a table.
2. Each has a single-column key (a declared primary key or a unique column),
   and the two keys are related: a declared FK from one to the other, or the
   same normalized name. A bare ``id`` matches every table's surrogate key, so
   generic names count only with a declared FK.
3. Each key's lowest distinct values hold at least ``MIN_CONTAINMENT`` of the
   other's, in **both** directions — a one-to-many reference holds in one
   direction only. A name-only match whose values any surrogate key would hold
   (``fk_values.is_value_search_ambiguous``) is refused.
4. Joining the two keys returns a row.
5. The LLM confirms the tables are one entity rather than two that share an id
   (an order and its shipment), and names it.

Confirmed pairs merge as they are found, so a chain of pairs forms one Concept
— the union of its pairs. Check 1 is an equivalence, so no chain can cross
families. ``PREVENT_SEMANTIC_FK_DB_QUERIES`` limits checks 3 and 4 to values
already stored and skips the join.
"""

from __future__ import annotations

from collections import defaultdict
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from auto_ontology.connectors.base import SQLDatabase
from auto_ontology.dal.concepts import (
    ensure_concept_key,
    ensure_term_concept,
    fetch_concept_fields,
    fetch_concept_merge_candidates,
    fetch_table_concept_id,
    merge_concepts,
)
from auto_ontology.dal.datasources import store_column_distinct_values
from auto_ontology.infra.feature_flags import is_semantic_fk_db_query_prevented
from auto_ontology.semantic.fk_values import (
    MIN_CONTAINMENT,
    containment,
    is_value_search_ambiguous,
)
from auto_ontology.semantic.models import ConceptMergeDecision
from auto_ontology.semantic.semantic_fk import (
    _resolve_connector,
    join_probe_returns_rows,
)
from auto_ontology.semantic.visit_enter import (
    KEY_DISTINCT_VALUES_LIMIT,
    probe_ascending_distinct_values,
)
from auto_ontology.utils.llm_invoke import (
    get_llm_client,
    invoke_with_structured_output,
)
from auto_ontology.utils.sample_values import (
    parse_sample_values,
    stringify_sample_values,
)

logger = logging.getLogger(__name__)

_GENERIC_KEY_NAMES = frozenset({"id", "pk", "key", "code", "uuid", "guid", "rowid"})
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_FIELDS_IN_PROMPT = 25

_SYSTEM_PROMPT = """\
You are a database schema expert. Two tables share a key: each row of one has \
a matching row in the other. Decide whether they are parts of ONE real-world \
entity whose fields were split across two tables (a customer and its extended \
profile, a product and its specification sheet), or TWO entities that merely \
share an identifier (an order and its shipment, a person and their account, \
a school and one year of its statistics).

Rules:
- same_entity is true only when each table holds at most one row per entity \
and both describe that entity itself.
- When same_entity is true, name the entity: short, singular, business wording.
- Do NOT guess. If unsure, return same_entity false.
"""


def resolve_concepts(database_name: str) -> int:
    """Merge the Concepts of split entities in *database_name*.

    Returns the number of merges applied.
    """
    rows = fetch_concept_merge_candidates(database_name)
    pairs = _candidate_pairs(rows)
    if not pairs:
        logger.info("resolve_concepts: no candidate table pairs")
        return 0
    logger.info("resolve_concepts: %d candidate table pair(s)", len(pairs))

    connector: SQLDatabase | None = None
    if is_semantic_fk_db_query_prevented():
        logger.info(
            "resolve_concepts: PREVENT_SEMANTIC_FK_DB_QUERIES is set — comparing "
            "stored values only, without a join check"
        )
    else:
        connector = _resolve_connector(database_name)
    values = _Values(connector)

    try:
        llm = get_llm_client()
    except Exception:
        logger.warning(
            "resolve_concepts: no LLM available — no Concept is merged without "
            "confirmation"
        )
        return 0

    merged = 0
    renamed: set[str] = set()
    for base, part in pairs:
        survivor = fetch_table_concept_id(base["table_id"]) or ensure_term_concept(
            base["term_id"]
        )
        absorbed = fetch_table_concept_id(part["table_id"]) or ensure_term_concept(
            part["term_id"]
        )
        if survivor is None or absorbed is None or survivor == absorbed:
            continue
        if not _values_match(base, part, values):
            continue
        if connector is not None and not join_probe_returns_rows(
            connector,
            {**part, "name": part["column_name"]},
            base,
            stage="resolve_concepts",
        ):
            continue
        decision = _confirm(llm, base, part, survivor, absorbed)
        if decision is None or not decision.same_entity:
            logger.info(
                "resolve_concepts: %s and %s are not one entity",
                base["table_name"],
                part["table_name"],
            )
            continue

        survivor_key = ensure_concept_key(survivor, base["column_id"])
        absorbed_key = ensure_concept_key(absorbed, part["column_id"])
        if survivor_key is None or absorbed_key is None:
            logger.info(
                "resolve_concepts: %s.%s or %s.%s cannot be a key; not merged",
                base["table_name"],
                base["column_name"],
                part["table_name"],
                part["column_name"],
            )
            continue
        name = decision.name if survivor not in renamed else None
        if merge_concepts(
            survivor, absorbed, key_pairs={absorbed_key: survivor_key}, name=name
        ):
            merged += 1
            renamed.add(survivor)
            logger.info(
                "resolve_concepts: %s.%s + %s.%s -> Concept %r",
                base["table_name"],
                base["column_name"],
                part["table_name"],
                part["column_name"],
                decision.name or base["concept_name"] or base["term_name"],
            )
    logger.info("resolve_concepts: %d merge(s) applied", merged)
    return merged


def _normalized(name: str | None) -> str:
    return _NON_ALNUM.sub("", str(name or "").lower())


def _declared_between(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("fk_target_column_id") == right["column_id"]
        or right.get("fk_target_column_id") == left["column_id"]
    )


def _candidate_pairs(
    rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """One ``(base, part)`` key pair per related table pair, declared first.

    A declared FK pairs any two candidate columns. A shared name pairs only a
    table's primary key when it has one, and never another unique column of
    that table. Within each, pairs on primary keys come first.

    The base is the side a declared FK points at — the satellite references
    the core table — and otherwise the alphabetically first table, so the
    surviving Concept is the same on every compile.
    """
    by_table: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_table[row["table_id"]].append(row)
    table_ids = sorted(by_table, key=lambda t: (by_table[t][0]["table_name"], t))
    has_pk = {t for t, columns in by_table.items() if any(c["is_pk"] for c in columns)}

    def by_name(row: dict[str, Any]) -> bool:
        return bool(row["is_pk"]) or row["table_id"] not in has_pk

    pairs: list[tuple[tuple[int, int], dict[str, Any], dict[str, Any]]] = []
    for i, left_table in enumerate(table_ids):
        for right_table in table_ids[i + 1 :]:
            best: tuple[tuple[int, int], dict[str, Any], dict[str, Any]] | None = None
            for left in by_table[left_table]:
                for right in by_table[right_table]:
                    if left["family_id"] != right["family_id"]:
                        continue
                    if _declared_between(left, right):
                        tier = 0
                    elif (
                        by_name(left)
                        and by_name(right)
                        and _normalized(left["column_name"])
                        == _normalized(right["column_name"])
                        and _normalized(left["column_name"]) not in _GENERIC_KEY_NAMES
                    ):
                        tier = 1
                    else:
                        continue
                    rank = (tier, 2 - bool(left["is_pk"]) - bool(right["is_pk"]))
                    if left.get("fk_target_column_id") == right["column_id"]:
                        base, part = right, left
                    else:
                        base, part = left, right
                    if best is None or rank < best[0]:
                        best = (rank, base, part)
            if best is not None:
                pairs.append(best)
    pairs.sort(key=lambda p: (p[0], p[1]["table_name"], p[2]["table_name"]))
    return [(base, part) for _, base, part in pairs]


class _Values:
    """Lowest distinct values per key column: stored, else probed once."""

    def __init__(self, connector: SQLDatabase | None) -> None:
        self._connector = connector
        self._cache: dict[str, list[Any]] = {}

    def of(self, row: dict[str, Any]) -> list[Any]:
        column_id = row["column_id"]
        if column_id in self._cache:
            return self._cache[column_id]
        values = parse_sample_values(row.get("distinct_values")) or []
        if not values and self._connector is not None:
            values = (
                probe_ascending_distinct_values(
                    self._connector,
                    row.get("schema_name"),
                    row["table_name"],
                    row["column_name"],
                )
                or []
            )
            if values:
                store_column_distinct_values(
                    row["table_id"], {row["column_name"]: values}
                )
        self._cache[column_id] = values
        return values


def _values_match(base: dict[str, Any], part: dict[str, Any], values: _Values) -> bool:
    """Two-way containment of the keys' lowest distinct values."""
    base_values, part_values = values.of(base), values.of(part)
    if not _declared_between(base, part) and (
        is_value_search_ambiguous(str(base["column_name"]), base_values)
        or is_value_search_ambiguous(str(part["column_name"]), part_values)
    ):
        return False
    forward = containment(part_values, base_values, limit=KEY_DISTINCT_VALUES_LIMIT)
    backward = containment(base_values, part_values, limit=KEY_DISTINCT_VALUES_LIMIT)
    matched = (
        forward is not None
        and backward is not None
        and forward >= MIN_CONTAINMENT
        and backward >= MIN_CONTAINMENT
    )
    if not matched:
        logger.info(
            "resolve_concepts [values]: %s.%s / %s.%s containment %s / %s",
            base["table_name"],
            base["column_name"],
            part["table_name"],
            part["column_name"],
            forward,
            backward,
        )
    return matched


def _describe(
    row: dict[str, Any], fields: dict[str, list[str]], values: list[Any]
) -> str:
    names = fields.get(row["table_id"], [])[:_FIELDS_IN_PROMPT]
    lowest = stringify_sample_values(values, max_len=30) or []
    return (
        f"Table: {row.get('schema_name') or ''}.{row['table_name']}\n"
        f"Term: {row.get('term_name') or '(none)'}\n"
        f"Description: {row.get('term_description') or row.get('table_description') or '(none)'}\n"
        f"Key column: {row['column_name']}"
        f"{' (primary key)' if row.get('is_pk') else ''}\n"
        f"Lowest key values: {', '.join(lowest) or '(none)'}\n"
        f"Fields: {', '.join(names) or '(none)'}"
    )


def _confirm(
    llm: Any,
    base: dict[str, Any],
    part: dict[str, Any],
    survivor: str,
    absorbed: str,
) -> ConceptMergeDecision | None:
    fields: dict[str, list[str]] = defaultdict(list)
    for field in fetch_concept_fields([survivor, absorbed]):
        fields[field["table_id"]].append(field["column_name"])
    prompt = (
        f"{_describe(base, fields, parse_sample_values(base.get('distinct_values')) or [])}"
        "\n\n"
        f"{_describe(part, fields, parse_sample_values(part.get('distinct_values')) or [])}"
        "\n\nAre these two tables parts of one entity?"
    )
    try:
        return invoke_with_structured_output(
            llm,
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)],
            ConceptMergeDecision,
        )
    except Exception:
        logger.exception(
            "resolve_concepts: LLM failed for %s / %s",
            base["table_name"],
            part["table_name"],
        )
        return None
