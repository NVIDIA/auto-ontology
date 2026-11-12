# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Give every table and column a description, then re-embed the catalog.

Runs last in a compilation, once Terms, attributes, foreign keys and bridges
exist. A table or column with no description of its own gets one copied from
the semantic layer: a table its Term's, a column its attribute's or the one
its foreign key points at. The copy is the node's own from then on, so a later
change to the Term or attribute does not change it. What has no source -- a
table no Term represents, such as a bridge, and a column with neither an
attribute nor a resolved foreign key -- is written by the LLM, one call per
table. A description the node already has is never replaced.

The Table and Column rows were embedded at ingest, before any of this existed,
so they are re-embedded with their new descriptions.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from auto_ontology.dal.datasources import (
    fetch_catalog_node_ids,
    fetch_description_context,
    fill_empty_descriptions,
)
from auto_ontology.utils.llm_invoke import get_llm_client, invoke_with_structured_output

logger = logging.getLogger(__name__)

_WORKERS = 4
_MAX_SAMPLES = 5

_SYSTEM_PROMPT = """\
You write short descriptions of database tables and columns for a data catalog.
Each description is one plain sentence saying what the table's rows or the
column's values are, in business terms. Use the column names, data types,
sample values, the tables foreign keys point to, and the descriptions already
known. Do not invent facts the metadata does not support. When a column's
meaning is unclear, describe what its name and values show.
"""


class ColumnDescription(BaseModel):
    column_name: str = Field(description="The column's exact name.")
    description: str = Field(description="One sentence describing the column.")


class TableDescriptions(BaseModel):
    table_description: str = Field(
        default="", description="One sentence describing the table's rows."
    )
    columns: list[ColumnDescription] = Field(
        default_factory=list,
        description="A description for every column listed as needing one.",
    )


def describe_catalog(database_name: str) -> int:
    """Describe what has no description, then re-embed every Table and Column.

    Returns the number of tables and columns given a description.
    """
    tables = fetch_description_context(database_name)
    table_descriptions = _copied(tables)
    column_descriptions = _copied([col for table in tables for col in table["columns"]])
    gaps = [table for table in tables if _needs_description(table)]
    logger.info(
        "Copying %d semantic-layer description(s); describing %d of %d table(s) "
        "with no description source",
        len(table_descriptions) + len(column_descriptions),
        len(gaps),
        len(tables),
    )

    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        for table, result in zip(gaps, pool.map(_describe_table, gaps)):
            if result is None:
                continue
            if not _text(table.get("description")) and result.table_description:
                table_descriptions[str(table["id"])] = result.table_description
            wanted = {
                str(col["name"]): str(col["id"])
                for col in table["columns"]
                if not _text(col.get("description"))
            }
            for col in result.columns:
                if col.column_name in wanted and col.description.strip():
                    column_descriptions[wanted[col.column_name]] = col.description

    written = fill_empty_descriptions(table_descriptions, column_descriptions)
    logger.info("Descriptions written: %d", len(written))

    from auto_ontology.server.datasources.service import refresh_catalog_embeddings

    node_ids = fetch_catalog_node_ids(database_name)
    if node_ids:
        refresh_catalog_embeddings(node_ids)
        logger.info("Re-embedded %d Table/Column row(s)", len(node_ids))
    return len(written)


def _copied(nodes: list[dict[str, Any]]) -> dict[str, str]:
    """``{id: description}`` for each node whose description is borrowed."""
    return {
        str(node["id"]): _text(node.get("description"))
        for node in nodes
        if _text(node.get("description")) and not _text(node.get("own_description"))
    }


def _needs_description(table: dict[str, Any]) -> bool:
    return not _text(table.get("description")) or any(
        not _text(col.get("description")) for col in table["columns"]
    )


def _describe_table(table: dict[str, Any]) -> TableDescriptions | None:
    try:
        return invoke_with_structured_output(
            get_llm_client(max_tokens=4096),
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=_table_prompt(table)),
            ],
            TableDescriptions,
        )
    except Exception:
        logger.warning(
            "Describing table %s failed — leaving it undescribed",
            table.get("name"),
            exc_info=True,
        )
        return None


def _table_prompt(table: dict[str, Any]) -> str:
    lines = [f"Table: {table.get('schema_name')}.{table.get('name')}"]
    if description := _text(table.get("description")):
        lines.append(f"Description: {description}")
    else:
        lines.append("Description: (needs one)")
    if bridge := _text(table.get("bridge_description")):
        lines.append(f"Join pattern this table implements: {bridge}")
    lines.append("Columns:")
    for col in table["columns"]:
        line = f"- {col.get('name')} ({_text(col.get('data_type')) or 'unknown type'})"
        if references := _text(col.get("references")):
            line += f", foreign key to {references}"
        samples = [str(v) for v in (col.get("sample_values") or [])[:_MAX_SAMPLES]]
        if samples:
            line += f", sample values: {', '.join(samples)}"
        description = _text(col.get("description"))
        line += f": {description}" if description else ": (needs a description)"
        lines.append(line)
    lines.append(
        "Return a table_description if the table needs one, and a description "
        "for every column marked as needing one."
    )
    return "\n".join(lines)


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""
