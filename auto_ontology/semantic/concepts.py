# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile concepts: give every Term one, key it, and merge the ones that match.

Algorithm
---------
1. **Singletons** -- every Term describing a table in the database gets a
   concept (``ensure_term_concept``), keyed on that table by its declared
   primary key or, failing that, the unique column other tables reference
   most -- by name or declared foreign key, then by shared sampled values.
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
from auto_ontology.utils.sample_values import parse_sample_values
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
    index: _ReferenceIndex | None = None
    keyed = 0
    for row in rows:
        concept_id = ensure_term_concept(row["term_id"])
        if not concept_id or row["table_id"] in fetch_concept_keys(concept_id):
            continue
        if index is None:
            index = _ReferenceIndex.load(database_name)
        key = _default_key(row["table_id"], list(row["pk"] or []), index)
        if key and set_concept_key(concept_id, row["table_id"], key):
            keyed += 1
    return keyed


_KEY_LIKE_NAME = re.compile(r"(id|key|code|number|_num|_no|_nbr)$")
#: Names of columns that describe a row rather than identify it. Many tables
#: share them, which says nothing about references.
_DESCRIPTIVE_NAME = re.compile(
    r"(desc|description|name|title|text|comment|comments|note|notes|flag|ind"
    r"|indicator|attribute|attr|label)$"
)
_KEY_SUFFIX = re.compile(r"(id|key|code|number|num|no|nbr)$")


class _ReferenceIndex:
    """The database's columns, indexed by what other tables could reference.

    Built once per compile: ``columns`` per table, the tables holding a column
    of each normalized name, the tables holding each sampled value, and the
    tables with a declared foreign key into each column.
    """

    def __init__(self, columns: list[dict[str, Any]], declared: list[dict[str, Any]]):
        self.table_names: dict[str, str] = {}
        self.columns: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.by_name: dict[str, set[str]] = defaultdict(set)
        self.by_value: dict[str, set[str]] = defaultdict(set)
        for column in columns:
            table_id = column["table_id"]
            self.table_names[table_id] = column["table_name"]
            self.columns[table_id].append(column)
            self.by_name[_normalize(column["name"])].add(table_id)
            for value in _comparable_values(column):
                self.by_value[value].add(table_id)
        self.declared: dict[str, set[str]] = defaultdict(set)
        for row in declared:
            self.declared[row["target_column_id"]].add(row["source_table_id"])

    @classmethod
    def load(cls, database_name: str) -> _ReferenceIndex:
        scope = (
            s.catalog_column.join(
                s.catalog_table, s.catalog_table.c.id == s.catalog_column.c.table_id
            )
            .join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            )
            .join(
                s.catalog_database,
                s.catalog_database.c.id == s.catalog_schema.c.database_id,
            )
        )
        in_database = (
            s.catalog_database.c.name == database_name,
            ~s.catalog_table.c.is_legacy,
        )
        columns = store().query_read(
            select(
                s.catalog_column.c.id,
                s.catalog_column.c.table_id,
                s.catalog_column.c.name,
                s.catalog_column.c.data_type,
                s.catalog_column.c.is_unique,
                s.catalog_column.c.sample_values,
                s.catalog_table.c.name.label("table_name"),
            )
            .select_from(scope)
            .where(*in_database)
            .order_by(s.catalog_column.c.table_id, s.catalog_column.c.name)
        )
        declared = store().query_read(
            select(
                s.column__foreign_key.c.target_column_id,
                s.catalog_column.c.table_id.label("source_table_id"),
            )
            .select_from(
                scope.join(
                    s.column__foreign_key,
                    s.column__foreign_key.c.source_column_id == s.catalog_column.c.id,
                )
            )
            .where(*in_database)
        )
        return cls(
            [
                {**dict(c), "sample_values": parse_sample_values(c["sample_values"])}
                for c in columns
            ],
            [dict(r) for r in declared],
        )

    def name_references(self, table_id: str, column: dict[str, Any]) -> int:
        """Other tables that declare a foreign key into *column* or name it."""
        name = _normalize(column["name"])
        names = {name}
        if name in _GENERIC_KEY_NAMES:
            table = _normalize(self.table_names.get(table_id, ""))
            names = {f"{table}{name}"}
            if table.endswith("s"):
                names.add(f"{table[:-1]}{name}")
        tables = set(self.declared.get(column["id"], set()))
        for name in names:
            tables |= self.by_name.get(name, set())
        return len(tables - {table_id})

    def value_references(self, table_id: str, column: dict[str, Any]) -> int:
        """Other tables with a column sharing one of *column*'s sampled values."""
        tables: set[str] = set()
        for value in _comparable_values(column):
            tables |= self.by_value.get(value, set())
        return len(tables - {table_id})


def _comparable_values(column: dict[str, Any]) -> set[str]:
    """Sampled values worth matching across tables: not dates or flags, and
    long enough not to collide by chance."""
    if _type_family(column.get("data_type")) in ("time", "bool"):
        return set()
    values = column.get("sample_values") or []
    return {
        text
        for value in values
        if len(text := str(value).strip().casefold()) >= 3
        and not isinstance(value, (dict, list))
    }


def _default_key(
    table_id: str, pk_names: list[str], index: _ReferenceIndex
) -> list[str]:
    """The declared primary key's column ids, else the unique column other
    tables reference most; ``[]`` when no unique column is a plausible key."""
    columns = index.columns.get(table_id, [])
    by_name = {c["name"].casefold(): c["id"] for c in columns}
    if pk_names:
        key = [by_name.get(name.casefold()) for name in pk_names]
        if all(key):
            return [column_id for column_id in key if column_id]
    column_id = _choose_key(table_id, [c for c in columns if c["is_unique"]], index)
    return [column_id] if column_id else []


def _choose_key(
    table_id: str, unique: list[dict[str, Any]], index: _ReferenceIndex
) -> str | None:
    """The unique column other tables most plausibly reference.

    Columns named after another table's key (``<other table>_id``) are that
    table's, unique here only by chance -- unless this table's own name starts
    with it, as a copy or variant of that table's does -- and descriptive columns (names,
    descriptions, flags) never identify a row. Of the rest, a key-like name
    other tables reference wins, most references first; then any other
    referenced column; then an unreferenced key-like name. Ties go to a name
    built from this table's own, then to shared sampled values. A column
    nothing references is taken only when its name looks like a key:
    otherwise a value that happens to be unique in a small table would become
    the table's identity.
    """
    own = _normalize(index.table_names.get(table_id, ""))
    others = {
        _normalize(name) for tid, name in index.table_names.items() if tid != table_id
    }
    others |= {name[:-1] for name in others if name.endswith("s")}
    ranked = []
    for column in unique:
        lowered = column["name"].casefold()
        stem = _KEY_SUFFIX.sub("", _normalize(column["name"]))
        own_named = bool(stem) and own.startswith(stem)
        if _DESCRIPTIVE_NAME.search(lowered) or (stem in others and not own_named):
            continue
        references = index.name_references(table_id, column)
        key_like = bool(_KEY_LIKE_NAME.search(lowered))
        tier = 0 if key_like and references else 1 if references else 2
        if tier == 2 and not key_like:
            continue
        ranked.append(
            (
                tier,
                -references,
                not own_named,
                -index.value_references(table_id, column),
                lowered,
                column["id"],
            )
        )
    return min(ranked)[-1] if ranked else None


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
