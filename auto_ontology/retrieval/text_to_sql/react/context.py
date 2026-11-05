# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a ReAct episode's tools share.

Built once per question and passed to every tool call. It is the only state
the tools keep, so an episode can be replayed by rebuilding it and resending
the same calls — which is what an RL environment needs from them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from auto_ontology.connectors.base import SQLDatabase


@dataclass
class ToolContext:
    """Per-question inputs and memory for the ReAct tools.

    ``seen_tables`` is the memory that lets a later ``retrieve`` connect what it
    finds to what earlier calls found: a join is only worth showing between
    tables the model already has in view. ``schemas`` is the catalog the SQL
    checks resolve names against, loaded on first use because a question that
    never drafts SQL should not pay for it.

    ``retriever`` searches the semantic collection (definitions, analyses);
    ``data_retriever`` the one tables and columns are embedded in.

    ``cost_status`` is for callers that charge for queries: it returns what the
    model has left to spend and what each action costs, read fresh each time
    because every query changes it. ``None`` when queries are free.
    """

    retriever: Any
    connectors: list[SQLDatabase]
    database_name: str | None = None
    data_retriever: Any = None
    numeric_type_alignment: bool = False
    seen_tables: dict[str, dict[str, Any]] = field(default_factory=dict)
    joined_pairs: set[frozenset[str]] = field(default_factory=set)
    anchor_column_ids: dict[str, str | None] = field(default_factory=dict)
    sibling_pairs: set[frozenset[str]] = field(default_factory=set)
    catalog_tables: list[dict[str, Any]] | None = None
    row_counts: dict[str, int | None] = field(default_factory=dict)
    schemas: dict[str, Any] | None = None
    sql_runs: int = 0
    cost_status: Callable[[], str | None] | None = None
