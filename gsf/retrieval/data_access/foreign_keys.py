# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Foreign-key / join discovery against Neo4j.

* :func:`gsf.neo4j.foreign_keys.get_relevant_fks` expands outward up to 3
  levels through ``fk`` / ``join`` edges to gather all FK relationships among
  the connected tables.
* :func:`_apply_foreign_key_hints` decorates relevant-table dicts in place.
* :func:`get_relevant_fks_from_candidates_tables` and
  :func:`get_relevant_tables_with_fks` are the two public combinators used
  by the deep_agent and text-to-SQL pipelines respectively.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gsf.neo4j.foreign_keys import get_relevant_fks
from gsf.retrieval.data_access.relevant_tables import (
    get_relevant_tables,
    get_relevant_tables_from_candidates,
)

if TYPE_CHECKING:
    from nemo_retriever.graph.retriever import Retriever

logger = logging.getLogger(__name__)


def _apply_foreign_key_hints(tables: list[dict], relevant_fks: list) -> None:
    """Set ``foreign_key`` on tables when name matches FK side."""
    for table in tables:
        for fk in relevant_fks:
            if table["name"] == fk["table1"]:
                table["foreign_key"] = (
                    f"'{table['name']}.{fk['column1']}' = '{fk['table2']}.{fk['column2']}'"
                )


def get_relevant_fks_from_candidates_tables(
    candidates: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Extract tables and foreign keys from flat candidate dicts.

    Wraps :func:`~gsf.retrieval.data_access.relevant_tables.get_relevant_tables_from_candidates`
    and additionally fetches FK / join relationships for the resulting tables
    via :func:`gsf.neo4j.foreign_keys.get_relevant_fks`, then applies FK hints in-place.

    Returns:
        ``(relevant_tables, relevant_fks)``.
    """
    relevant_tables = get_relevant_tables_from_candidates(candidates)
    if not relevant_tables:
        return [], []

    try:
        relevant_fks = get_relevant_fks([x["id"] for x in relevant_tables])
    except Exception:
        logger.exception("get_relevant_fks failed for candidate tables")
        relevant_fks = []

    _apply_foreign_key_hints(relevant_tables, relevant_fks)
    return relevant_tables, relevant_fks


def get_relevant_tables_with_fks(
    retriever: "Retriever",
    initial_question,
    k=15,
    database_name: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Like :func:`get_relevant_tables` but also returns FK relationships, with FK hints applied in-place."""
    relevant_tables_list = get_relevant_tables(
        retriever, initial_question, k=k, database_name=database_name
    )

    relevant_fks: list = []
    if relevant_tables_list:
        try:
            relevant_fks = get_relevant_fks([x["id"] for x in relevant_tables_list])
        except Exception:
            logger.exception("get_relevant_fks failed in get_relevant_tables_with_fks")
            relevant_fks = []
    _apply_foreign_key_hints(relevant_tables_list, relevant_fks)

    return relevant_tables_list, relevant_fks
