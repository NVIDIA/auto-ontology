# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile concepts: give every Term one, key it, and merge the ones that match.

Algorithm
---------
1. **Singletons** -- every Term describing a table in the database gets a
   concept (``ensure_term_concept``), keyed on that table by its declared
   primary key or, failing that, a unique column.
2. **Candidates** -- two concepts are a candidate pair when one of each's
   keys has the same column names (normalized) with compatible types. Keys
   made only of generic names (``id``, ``code``, ...) are skipped: every
   auto-increment table would pair with every other one.
3. **Value overlap** -- the two keys must share at least one value in the
   live database, checked with a bounded read-only probe.
4. **LLM confirmation** -- the model says whether the two describe the same
   thing, and names the merged concept. A confirmed pair is merged: terms,
   keys and references move to the surviving concept.
"""

from __future__ import annotations

from collections import defaultdict
import itertools
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import select

from auto_ontology.connectors.base import SQLDatabase
from auto_ontology.dal import schema as s
from auto_ontology.dal.concepts import (
    ensure_term_concept,
    fetch_concept_keys,
    fetch_concepts,
    merge_concepts,
    set_concept_key,
)
from auto_ontology.dal.session import store
from auto_ontology.retrieval.text_to_sql.db_probe.executor import ProbeExecutor
from auto_ontology.semantic.models import ConceptMergeDecision
from auto_ontology.semantic.semantic_fk import _resolve_connector
from auto_ontology.utils.llm_invoke import (
    get_llm_client,
    invoke_with_structured_output,
)
from auto_ontology.utils.sql_identifiers import quoted_identifier
from auto_ontology.vdb import get_semantic_vdb

logger = logging.getLogger(__name__)

#: Upper bound on candidate pairs probed and sent to the LLM in one pass.
MAX_CANDIDATE_PAIRS = 200

_GENERIC_KEY_NAMES = frozenset({"id", "key", "pk", "code", "uuid", "guid", "no"})

_TYPE_FAMILIES = (
    ("number", ("int", "serial", "numeric", "decimal", "number", "long", "short")),
    ("text", ("char", "text", "string", "clob")),
    ("time", ("date", "time")),
    ("uuid", ("uuid", "uniqueidentifier")),
    ("bool", ("bool", "bit")),
)

_SYSTEM_PROMPT = """\
You are a data modeling expert. Two business concepts were found whose tables \
share a key: the same key column names, and at least one shared key value.

Decide whether both tables describe the SAME real-world thing -- one row per \
instance, with its columns split across tables (e.g. customer and \
customer_details). Answer false when one is a parent and the other its \
children or events (customer and order), or when they only share an id \
format. When true, give a short business name and a one-sentence \
description for the merged concept.
"""


def compile_concepts(database_name: str) -> int:
    """Create, key and merge the concepts of *database_name*.

    Returns the number of merges made.
    """
    keyed = _ensure_singleton_concepts(database_name)
    logger.info("compile_concepts: %d table key(s) assigned", keyed)
    return _merge_shared_key_concepts(database_name)


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------


def _ensure_singleton_concepts(database_name: str) -> int:
    """Give every Term on a table here a concept and key it on that table."""
    rows = store().query_read(
        select(
            s.table__term.c.term_id,
            s.table__term.c.table_id,
            s.catalog_table.c.pk,
        )
        .select_from(
            s.table__term.join(
                s.catalog_table, s.catalog_table.c.id == s.table__term.c.table_id
            )
            .join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            )
            .join(
                s.catalog_database,
                s.catalog_database.c.id == s.catalog_schema.c.database_id,
            )
        )
        .where(
            s.catalog_database.c.name == database_name,
            ~s.catalog_table.c.is_legacy,
        )
        .order_by(s.table__term.c.table_id, s.table__term.c.term_id)
    )
    keyed = 0
    for row in rows:
        concept_id = ensure_term_concept(row["term_id"])
        if not concept_id or row["table_id"] in fetch_concept_keys(concept_id):
            continue
        key = _default_key(row["table_id"], list(row["pk"] or []))
        if key and set_concept_key(concept_id, row["table_id"], key):
            keyed += 1
    return keyed


def _default_key(table_id: str, pk_names: list[str]) -> list[str]:
    """The declared primary key's column ids, else one unique column's."""
    columns = store().query_read(
        select(
            s.catalog_column.c.id,
            s.catalog_column.c.name,
            s.catalog_column.c.is_unique,
        )
        .where(s.catalog_column.c.table_id == table_id)
        .order_by(s.catalog_column.c.name)
    )
    by_name = {c["name"].casefold(): c["id"] for c in columns}
    if pk_names:
        key = [by_name.get(name.casefold()) for name in pk_names]
        if all(key):
            return [column_id for column_id in key if column_id]
    unique = [c for c in columns if c["is_unique"]]
    if not unique:
        return []
    named_as_key = [
        c for c in unique if re.search(r"(id|key|code)$", c["name"].casefold())
    ]
    return [(named_as_key or unique)[0]["id"]]


# ---------------------------------------------------------------------------
# Candidates and merging
# ---------------------------------------------------------------------------


def _normalize(name: str) -> str:
    return re.sub(r"[^0-9a-z]", "", name.casefold())


def _type_family(data_type: str | None) -> str | None:
    lowered = (data_type or "").casefold()
    if not lowered:
        return None
    for family, markers in _TYPE_FAMILIES:
        if any(marker in lowered for marker in markers):
            return family
    return lowered


def _compatible(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Same family, or unknown on either side."""
    fa, fb = _type_family(a.get("data_type")), _type_family(b.get("data_type"))
    return fa is None or fb is None or fa == fb


def _key_entries(concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per (concept, member table) key, with a name signature."""
    entries = []
    for concept in concepts:
        for table_id, table in concept["keys"].items():
            columns = table["columns"]
            names = [_normalize(c["name"]) for c in columns]
            if not names or all(name in _GENERIC_KEY_NAMES for name in names):
                continue
            entries.append(
                {
                    "concept": concept,
                    "table_id": table_id,
                    "table_name": table["table_name"],
                    "schema_name": table["schema_name"],
                    "columns": columns,
                    "by_name": dict(zip(names, columns)),
                    "signature": tuple(sorted(names)),
                }
            )
    return entries


def _candidate_pairs(
    entries: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        groups[entry["signature"]].append(entry)
    pairs = []
    for signature, group in sorted(groups.items()):
        if len(signature) != len(set(signature)):
            continue
        for a, b in itertools.combinations(group, 2):
            if a["concept"]["id"] == b["concept"]["id"]:
                continue
            if all(
                _compatible(a["by_name"][name], b["by_name"][name])
                for name in signature
            ):
                pairs.append((a, b))
    return pairs


def _merge_shared_key_concepts(database_name: str) -> int:
    entries = _key_entries(fetch_concepts(database_name))
    pairs = _candidate_pairs(entries)
    if not pairs:
        logger.info("compile_concepts: no shared-key candidates")
        return 0
    if len(pairs) > MAX_CANDIDATE_PAIRS:
        logger.warning(
            "compile_concepts: %d candidate pair(s), checking the first %d",
            len(pairs),
            MAX_CANDIDATE_PAIRS,
        )
        pairs = pairs[:MAX_CANDIDATE_PAIRS]

    connector = _resolve_connector(database_name)
    if connector is None:
        logger.info("compile_concepts: no connector for %s, no merges", database_name)
        return 0
    try:
        llm = get_llm_client()
    except Exception:
        logger.warning("compile_concepts: no LLM configured, no merges")
        return 0

    # Merges are applied as they are confirmed, so a later pair may name a
    # concept that was folded into another; `merged_into` follows it.
    merged_into: dict[str, str] = {}

    def current(concept_id: str) -> str:
        while concept_id in merged_into:
            concept_id = merged_into[concept_id]
        return concept_id

    merges = 0
    with ProbeExecutor(connector, max_calls=len(pairs)) as executor:
        for a, b in pairs:
            keep_id, drop_id = current(a["concept"]["id"]), current(b["concept"]["id"])
            if keep_id == drop_id:
                continue
            if not _shares_a_value(a, b, connector, executor):
                continue
            decision = _confirm(a, b, llm)
            if decision is None or not decision.same_thing:
                continue
            if not _align_keys(drop_id, a):
                continue
            if merge_concepts(
                keep_id, drop_id, name=decision.name, description=decision.description
            ):
                merged_into[drop_id] = keep_id
                merges += 1
                _forget_vector(drop_id)
                logger.info(
                    "compile_concepts: merged %s (%s) into %s (%s)",
                    b["concept"]["name"],
                    b["table_name"],
                    a["concept"]["name"],
                    a["table_name"],
                )
    return merges


def _forget_vector(concept_id: str) -> None:
    """Drop a merged-away concept's embedding; a stale row is only dead weight."""
    try:
        get_semantic_vdb().delete_by_id(concept_id)
    except Exception:
        logger.info("compile_concepts: could not delete vector %s", concept_id)


def _align_keys(drop_id: str, reference: dict[str, Any]) -> bool:
    """Reorder every key of *drop_id* to *reference*'s column order.

    Member keys join position by position, so after a merge the same name has
    to sit at the same ordinal on every table.
    """
    order = [_normalize(c["name"]) for c in reference["columns"]]
    names = {
        r["id"]: _normalize(r["name"])
        for r in store().query_read(
            select(s.catalog_column.c.id, s.catalog_column.c.name).where(
                s.catalog_column.c.id.in_(
                    [c for cols in fetch_concept_keys(drop_id).values() for c in cols]
                )
            )
        )
    }
    for table_id, columns in fetch_concept_keys(drop_id).items():
        by_name = {names.get(column_id): column_id for column_id in columns}
        if set(by_name) != set(order):
            return False
        aligned = [by_name[name] for name in order]
        if aligned != columns and not set_concept_key(drop_id, table_id, aligned):
            return False
    return True


def _shares_a_value(
    a: dict[str, Any],
    b: dict[str, Any],
    connector: SQLDatabase,
    executor: ProbeExecutor,
) -> bool:
    dialect = getattr(connector, "dialect", None)
    table_a = connector.qualify(a["schema_name"], a["table_name"])
    table_b = connector.qualify(b["schema_name"], b["table_name"])
    names = a["signature"]
    a_cols = [quoted_identifier(a["by_name"][n]["name"], dialect) for n in names]
    b_cols = [quoted_identifier(b["by_name"][n]["name"], dialect) for n in names]
    match = " AND ".join(f"kb.{bc} = ka.{ac}" for ac, bc in zip(a_cols, b_cols))
    sql = (
        f"SELECT 1 AS hit FROM {table_a} ka "
        f"WHERE EXISTS (SELECT 1 FROM {table_b} kb WHERE {match}) LIMIT 1"
    )
    result = executor.run(sql, purpose="concept key overlap")
    return bool(result.get("ok") and result.get("rows"))


def _describe(entry: dict[str, Any]) -> str:
    concept = entry["concept"]
    terms = ", ".join(
        f"{t['name']} ({t.get('description') or 'no description'})"
        for t in concept["terms"]
    )
    key = ", ".join(c["name"] for c in entry["columns"])
    return (
        f"Concept: {concept['name']}\n"
        f"Description: {concept.get('description') or '(none)'}\n"
        f"Terms: {terms}\n"
        f"Table: {entry['schema_name']}.{entry['table_name']} key ({key})"
    )


def _confirm(
    a: dict[str, Any], b: dict[str, Any], llm: object
) -> ConceptMergeDecision | None:
    prompt = f"First:\n{_describe(a)}\n\nSecond:\n{_describe(b)}\n"
    try:
        return invoke_with_structured_output(
            llm,
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)],
            ConceptMergeDecision,
        )
    except Exception:
        logger.exception(
            "compile_concepts: LLM failed for %s / %s",
            a["concept"]["name"],
            b["concept"]["name"],
        )
        return None
