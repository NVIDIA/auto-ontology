# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Join suspected FK columns that found no target directly to each other.

A suspected FK column that semantic FK resolution could not link usually
references a key whose values it does not share: in a sampled warehouse each
table kept a different slice of rows, so the referenced key's table has few of
the FK's values, while the tables referencing the same key still join one
another. Before such a column becomes a plain ColumnAttribute, it is matched
against the other FK columns of the database — the unlinked ones and those
already linked — on the same normalized name and a join probe that returns
rows. Columns that join are written as one SEMANTIC_FK_DIRECT group, which
joins them to each other through no Concept, and the unmatched-FK stage then
skips them.

An unlinked column with fewer than :data:`MIN_DISTINCT_VALUES` distinct values
is never grouped: a handful of codes shared by two tables join whether or not
one references the other.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from auto_ontology.dal.attributes import (
    fetch_direct_fk_peers,
    find_unlinked_fk_columns,
    link_direct_fk_columns,
)
from auto_ontology.dal.datasources import store_column_distinct_values
from auto_ontology.dal.term_families import fetch_table_families
from auto_ontology.infra.feature_flags import is_semantic_fk_db_query_prevented
from auto_ontology.semantic.cancellation import is_cancelled
from auto_ontology.semantic.concepts import _GENERIC_KEY_NAMES, _normalized
from auto_ontology.semantic.semantic_fk import (
    _Families,
    _resolve_connector,
    join_probe_returns_rows,
)
from auto_ontology.semantic.visit_enter import probe_ascending_distinct_values
from auto_ontology.utils.sample_values import parse_sample_values

logger = logging.getLogger(__name__)

_STAGE = "link_direct_fks"

#: Below :data:`~auto_ontology.semantic.visit_enter.KEY_DISTINCT_VALUES_LIMIT`,
#: which caps how many distinct values a column stores.
MIN_DISTINCT_VALUES = 5


def link_direct_fks(database_name: str) -> int:
    """Group the unlinked FK columns of *database_name* that join same-name FKs.

    Needs the warehouse for its join probes, so it does nothing when warehouse
    queries are disabled or no connector serves the database. Returns the
    number of unlinked columns put in a group.
    """
    candidates = find_unlinked_fk_columns(database_name)
    if not candidates:
        logger.info("%s: no unlinked FK columns", _STAGE)
        return 0
    if is_semantic_fk_db_query_prevented():
        logger.info(
            "%s: PREVENT_SEMANTIC_FK_DB_QUERIES is set — skipping %d column(s)",
            _STAGE,
            len(candidates),
        )
        return 0
    connector = _resolve_connector(database_name)
    if connector is None:
        logger.info(
            "%s: no connector for %r — skipping %d column(s)",
            _STAGE,
            database_name,
            len(candidates),
        )
        return 0

    few = [col for col in candidates if not _has_enough_values(connector, col)]
    if few:
        logger.info(
            "%s: %d column(s) with fewer than %d distinct values not grouped: %s",
            _STAGE,
            len(few),
            MIN_DISTINCT_VALUES,
            sorted(f"{col['table_name']}.{col['name']}" for col in few),
        )
        skipped = {col["id"] for col in few}
        candidates = [col for col in candidates if col["id"] not in skipped]

    families = _Families(fetch_table_families(database_name))
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for peer in fetch_direct_fk_peers(database_name):
        by_name[_normalized(peer["name"])].append({**peer, "linked": True})
    for col in candidates:
        by_name[_normalized(col["name"])].append(
            {**col, "group_id": None, "linked": False}
        )

    linked = 0
    for name, columns in sorted(by_name.items()):
        if is_cancelled():
            logger.info("%s: cancelled", _STAGE)
            break
        if not name or name in _GENERIC_KEY_NAMES:
            continue
        if all(col["linked"] for col in columns) or len(columns) < 2:
            continue
        try:
            linked += _link_same_name(connector, columns, families)
        except Exception:
            logger.exception("%s: failed for columns named %r", _STAGE, name)
    logger.info("%s: %d column(s) linked directly", _STAGE, linked)
    return linked


def _has_enough_values(connector: Any, col: dict[str, Any]) -> bool:
    """Whether *col* holds at least :data:`MIN_DISTINCT_VALUES` distinct values.

    Counted from its stored lowest distinct values, probed and stored first
    when it has none. A column whose values cannot be read is not grouped.
    """
    values = parse_sample_values(col.get("distinct_values"))
    if values is None:
        values = probe_ascending_distinct_values(
            connector, col.get("schema_name"), col["table_name"], col["name"]
        )
        if values:
            store_column_distinct_values(col["table_id"], {col["name"]: values})
    return len(values or []) >= MIN_DISTINCT_VALUES


def _link_same_name(
    connector: Any,
    columns: list[dict[str, Any]],
    families: _Families,
) -> int:
    """Probe each unlinked column against its same-name peers; write the groups.

    Peers are probed linked ones first, and a pair already in one component is
    not probed again. Returns the number of unlinked columns written.
    """
    parent = {col["id"]: col["id"] for col in columns}

    def find(column_id: str) -> str:
        while parent[column_id] != column_id:
            parent[column_id] = parent[parent[column_id]]
            column_id = parent[column_id]
        return column_id

    def join(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    first_in_group: dict[str, str] = {}
    for col in columns:
        if col["group_id"]:
            join(col["id"], first_in_group.setdefault(col["group_id"], col["id"]))

    peers = sorted(columns, key=lambda col: (not col["linked"], col["table_name"]))
    for col in columns:
        if col["linked"]:
            continue
        for peer in peers:
            if peer["table_id"] == col["table_id"] or find(peer["id"]) == find(
                col["id"]
            ):
                continue
            if families.crosses(col["table_id"], peer["table_id"]) or families.crosses(
                peer["table_id"], col["table_id"]
            ):
                continue
            target = {
                "column_name": peer["name"],
                "table_name": peer["table_name"],
                "schema_name": peer.get("schema_name"),
            }
            if join_probe_returns_rows(connector, col, target, stage=_STAGE):
                join(col["id"], peer["id"])

    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for col in columns:
        components[find(col["id"])].append(col)
    written = 0
    for members in components.values():
        new = [col for col in members if not col["linked"]]
        if len(members) < 2 or not new:
            continue
        if link_direct_fk_columns([col["id"] for col in members]) is None:
            continue
        written += len(new)
        logger.info(
            "%s: %s joined directly",
            _STAGE,
            sorted(f"{col['table_name']}.{col['name']}" for col in members),
        )
    return written
