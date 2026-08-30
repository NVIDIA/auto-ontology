# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate enrichment: turn vector hits into the rows a prompt needs.

Retrieval returns ``(label, id)`` pairs and nothing else. This resolves each one
into the properties and surrounding context the generator actually reads —
notably ``relevant_tables``, the tables and columns a candidate implies.

Each label gets its own function, dispatched by a dict. The output is a mapping
of id to a properties dict, with per-label extras merged in.

**The branches disagree with each other, deliberately.** A candidate with no
statement behind it vanishes if it is a SqlAttribute and comes back blank if it
is a CustomAnalysis. Which is right is a product question — an attribute with
nothing to say contributes nothing to a prompt, but a caller that asked about a
specific id is better served by a visibly blank entry than an absent one. Both
arms are pinned by tests, so settling it is a deliberate change that fails a
test rather than a silent one.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import literal, select

from gsf.catalog.constants import Labels
from gsf.dal import schema as s
from gsf.dal.session import store
from gsf.dal.sql_fragments import column_description_expr

logger = logging.getLogger(__name__)

#: Labels that resolve to a table here. The others in ``Labels.LIST_OF_ALL``
#: are accepted and fall through to the plain-properties branch.
_NODE_TABLES = {
    Labels.DB: s.catalog_database,
    Labels.SCHEMA: s.catalog_schema,
    Labels.TABLE: s.catalog_table,
    Labels.COLUMN: s.catalog_column,
    Labels.SQL: s.sql_query,
    Labels.CUSTOM_ANALYSIS: s.custom_analysis,
}


def _table_payloads(table_ids: list[str]) -> dict[str, dict[str, Any]]:
    """``{table_id: table dict with nested columns}``.

    The shape ``relevant_tables`` entries take, built once for every table any
    candidate in this batch needs rather than per candidate — several hits
    routinely land on the same table.

    ``sample_values`` is emitted only when non-empty, and that matters: an
    empty list rendered into a prompt reads as "this column has no values",
    which is a different claim from "we did not profile it".
    """
    if not table_ids:
        return {}

    tables: dict[str, dict[str, Any]] = {}
    for row in store().query_read(
        select(
            s.catalog_table,
            s.catalog_schema.c.name.label("schema_name"),
            s.catalog_database.c.name.label("database_name"),
        )
        .select_from(
            s.catalog_table.join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            ).join(
                s.catalog_database,
                s.catalog_database.c.id == s.catalog_schema.c.database_id,
            )
        )
        .where(s.catalog_table.c.id.in_(table_ids))
    ):
        payload = dict(row)
        payload["label"] = Labels.TABLE
        payload["columns"] = []
        tables[row["id"]] = payload

    for row in store().query_read(
        select(
            s.catalog_column.c.table_id,
            s.catalog_column.c.name,
            s.catalog_column.c.data_type,
            s.catalog_column.c.sample_values,
            s.catalog_column.c.format,
            column_description_expr().label("description"),
        )
        .where(s.catalog_column.c.table_id.in_(table_ids))
        .order_by(s.catalog_column.c.table_id, s.catalog_column.c.ordinal_position)
    ):
        table = tables.get(row["table_id"])
        if table is None:
            continue
        sample_values = row["sample_values"]
        table["columns"].append(
            {
                "name": row["name"],
                # `toString(coalesce(c.data_type, ""))` -- callers concatenate
                # this, so a null must become an empty string, not "None".
                "data_type": str(row["data_type"] or ""),
                "description": row["description"],
                "format": row["format"],
                "sample_values": sample_values if sample_values else None,
            }
        )
    return tables


def _sql_owned_tables(link_table, owner_column, owner_ids: list[str]):
    """``{owner_id: [table_id, ...]}`` for owners that reach tables via their SQL."""
    owned: dict[str, list[str]] = {}
    for row in store().query_read(
        select(owner_column, s.sql_query_table.c.table_id)
        .select_from(
            link_table.join(
                s.sql_query_table,
                s.sql_query_table.c.sql_query_id == link_table.c.sql_query_id,
            )
        )
        .where(owner_column.in_(owner_ids))
        .distinct()
    ):
        owned.setdefault(row[owner_column.name], []).append(row["table_id"])
    return owned


def _sql_text(link_table, owner_column, owner_ids: list[str]) -> dict[str, str]:
    """``{owner_id: sql}``, taking the lowest-id statement as every read does."""
    text: dict[str, str] = {}
    for row in store().query_read(
        select(owner_column, s.sql_query.c.id, s.sql_query.c.sql_full_query)
        .select_from(
            link_table.join(s.sql_query, s.sql_query.c.id == link_table.c.sql_query_id)
        )
        .where(owner_column.in_(owner_ids))
        .order_by(owner_column, s.sql_query.c.id)
    ):
        text.setdefault(row[owner_column.name], row["sql_full_query"])
    return text


def _expand_custom_analyses(ids: list[str]) -> dict[str, dict[str, Any]]:
    """Analysis properties plus its SQL and the tables that SQL references.

    Both are optional here, so an analysis with no statement still comes back,
    with ``sql: ""`` and an empty ``relevant_tables``.
    """
    rows = store().query_read(
        select(s.custom_analysis).where(s.custom_analysis.c.id.in_(ids))
    )
    sql = _sql_text(s.custom_analysis_sql, s.custom_analysis_sql.c.analysis_id, ids)
    owned = _sql_owned_tables(
        s.custom_analysis_sql, s.custom_analysis_sql.c.analysis_id, ids
    )
    tables = _table_payloads(sorted({t for v in owned.values() for t in v}))

    return {
        row["id"]: {
            **dict(row),
            "sql": sql.get(row["id"], ""),
            "relevant_tables": [
                tables[t] for t in owned.get(row["id"], []) if t in tables
            ],
        }
        for row in rows
    }


def _expand_sql_attributes(ids: list[str]) -> dict[str, dict[str, Any]]:
    """Attribute properties plus its SQL, its tables, and its Term.

    Unlike the analysis branch these matches were **not** optional, so an
    attribute with no statement or no term does not appear at all. Preserved:
    such an attribute has nothing to contribute to a prompt, and the caller
    treats a missing id as "no context available".
    """
    rows = store().query_read(
        select(
            s.sql_attribute,
            s.term.c.name.label("term_name"),
            s.term.c.id.label("term_id"),
        )
        .select_from(
            s.sql_attribute.join(
                s.sql_attribute_term,
                s.sql_attribute_term.c.attribute_id == s.sql_attribute.c.id,
            ).join(s.term, s.term.c.id == s.sql_attribute_term.c.term_id)
        )
        .where(
            s.sql_attribute.c.id.in_(ids),
            select(literal(1))
            .where(s.sql_attribute_sql.c.attribute_id == s.sql_attribute.c.id)
            .correlate(s.sql_attribute)
            .exists(),
        )
        .order_by(s.sql_attribute.c.id, s.term.c.id)
    )
    sql = _sql_text(s.sql_attribute_sql, s.sql_attribute_sql.c.attribute_id, ids)
    owned = _sql_owned_tables(
        s.sql_attribute_sql, s.sql_attribute_sql.c.attribute_id, ids
    )
    tables = _table_payloads(sorted({t for v in owned.values() for t in v}))

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        # An attribute linked to two Terms produced two rows and the map kept
        # one; the ORDER BY makes that the lowest-id term, every time.
        result.setdefault(
            row["id"],
            {
                **{k: v for k, v in row.items() if k not in {"term_name", "term_id"}},
                "sql": sql.get(row["id"], ""),
                "relevant_tables": [
                    tables[t] for t in owned.get(row["id"], []) if t in tables
                ],
                "term_name": row["term_name"],
                "term_id": row["term_id"],
            },
        )
    return result


def _expand_columns(ids: list[str]) -> dict[str, dict[str, Any]]:
    """Column properties plus its parent table, as the single relevant table.

    A column hit is really a table hit — the generator needs the whole table to
    write a query against it — so the parent is merged in with *all* its
    columns, not just the one that matched.
    """
    rows = store().query_read(
        select(
            s.catalog_column,
            s.catalog_table.c.name.label("table_name"),
            s.catalog_table.c.table_type,
            s.catalog_table.c.id.label("parent_id"),
        )
        .select_from(
            s.catalog_column.join(
                s.catalog_table, s.catalog_table.c.id == s.catalog_column.c.table_id
            )
        )
        .where(s.catalog_column.c.id.in_(ids))
    )
    tables = _table_payloads(sorted({row["parent_id"] for row in rows}))

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        payload = {k: v for k, v in row.items() if k != "parent_id"}
        payload["parent_id"] = row["parent_id"]
        parent = tables.get(row["parent_id"])
        result[row["id"]] = {
            **payload,
            "relevant_tables": [parent] if parent else [],
        }
    return result


def _expand_plain(table):
    """The default branch: the row's properties and nothing else."""

    def expand(ids: list[str]) -> dict[str, dict[str, Any]]:
        return {
            row["id"]: dict(row)
            for row in store().query_read(select(table).where(table.c.id.in_(ids)))
        }

    return expand


_EXPANDERS = {
    Labels.CUSTOM_ANALYSIS: _expand_custom_analyses,
    "SqlAttribute": _expand_sql_attributes,
    Labels.COLUMN: _expand_columns,
}


def expand_info(ids_and_labels: list | None) -> dict:
    """``{id: properties}`` for each ``{"id", "label"}`` pair given.

    Malformed entries are dropped rather than raising — these arrive from
    vector-store metadata, where one bad row should not cost the whole
    retrieval. An unknown label is logged and skipped for the same reason.
    """
    by_label: dict[str, list[str]] = {}
    for item in ids_and_labels or []:
        if not isinstance(item, dict) or item.get("id") is None:
            continue
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        by_label.setdefault(label, []).append(item["id"])

    results: dict[str, Any] = {}
    allowed = set(Labels.LIST_OF_ALL) | set(_EXPANDERS)
    for label, ids in by_label.items():
        if label not in allowed:
            logger.warning("Skipping unknown label %r in expand_info", label)
            continue
        expander = _EXPANDERS.get(label)
        if expander is None:
            table = _NODE_TABLES.get(label)
            if table is None:
                logger.warning("No table for label %r in expand_info", label)
                continue
            expander = _expand_plain(table)
        results |= expander(sorted(set(ids)))
    return results
