# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a ReAct episode's tools share.

Built once per question and passed to every tool call. It is the only state
the tools keep, so an episode can be replayed by rebuilding it and resending
the same calls — which is what an RL environment needs from them.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import Any

from auto_ontology.connectors.base import SQLDatabase


@dataclass
class ToolContext:
    """Per-question inputs and memory for the ReAct tools.

    ``seen_tables`` is the memory that lets a later ``retrieve`` connect what it
    finds to what earlier calls found: a join is only worth showing between
    tables the model already has in view. ``schemas`` holds the catalog the SQL
    checks resolve names against, per database, loaded on first use because a
    question that never drafts SQL should not pay for it.

    ``join_paths`` keeps the catalog joins ``retrieve`` found between tables in
    view, in the shape KumoRFM links its graph with. ``question``,
    ``prediction_llm`` and ``custom_prompts`` are what ``predict`` hands
    KumoRFM: the question as the user asked it, not as the model rephrased it,
    and the organization's instructions and glossary the episode's model sees.

    ``row_counts`` caches the SIMILAR-line counts for the episode;
    ``counts_left`` is how many more the current ``retrieve`` call may run.

    ``pages`` holds the ``run_sql`` results too long to show, for ``grep`` to
    search; page *n* is ``pages[n - 1]``.

    ``shown_tables`` is every table a ``retrieve`` result named — in a table
    block, a JOIN route, a SIMILAR line or a definition's SQL — as lowercased
    ``(schema, name)``. It is wider than ``seen_tables`` on purpose: a bridge
    table on a join route or the other copy on a SIMILAR line is one the model
    is told to use, though its columns were never listed.

    ``retriever`` searches the semantic collection; ``data_retriever`` the one
    tables and columns are embedded in, for what the semantic layer leaves out.

    ``information_answers`` lets ``submit`` take an information question's answer
    with no SQL; it stays off wherever a grader scores the SQL.
    ``information_retrieved`` records that ``retrieve`` showed the information
    view, which such an answer has to come from; a SQL-view search does not count.

    ``submit_permission_denied`` tracks the outcome of the last executed final
    query so the episode can explain a denial if it ends without an answer.

    ``cost_status`` is for callers that charge for queries: it returns what the
    model has left to spend and what each action costs, read fresh each time
    because every query changes it. ``None`` when queries are free.
    ``can_run_queries`` comes with it: False once the next query would be
    refused, so the episode can still submit instead of trying to run one.
    """

    retriever: Any
    connectors: list[SQLDatabase]
    database_name: str | None = None
    data_retriever: Any = None
    numeric_type_alignment: bool = False
    information_answers: bool = False
    information_retrieved: bool = False
    question: str = ""
    prediction_llm: Any = None
    custom_prompts: str = ""
    seen_tables: dict[str, dict[str, Any]] = field(default_factory=dict)
    joined_pairs: set[frozenset[str]] = field(default_factory=set)
    join_paths: list[dict[str, Any]] = field(default_factory=list)
    anchor_column_ids: dict[str, str | None] = field(default_factory=dict)
    sibling_pairs: set[frozenset[str]] = field(default_factory=set)
    shown_concepts: set[str] = field(default_factory=set)
    catalog_tables: list[dict[str, Any]] | None = None
    row_counts: dict[str, int | None] = field(default_factory=dict)
    counts_left: int = 0
    schemas: dict[str | None, dict[str, Any]] = field(default_factory=dict)
    sql_runs: int = 0
    predict_calls: int = 0
    prediction_refusal: str = ""
    pages: list[Any] = field(default_factory=list)
    shown_tables: set[tuple[str, str]] = field(default_factory=set)
    cost_status: Callable[[], str | None] | None = None
    can_run_queries: Callable[[], bool] | None = None
    submit_permission_denied: bool = False

    def show_table(self, schema: str | None, name: str | None) -> None:
        if name:
            self.shown_tables.add(((schema or "").lower(), name.lower()))

    def was_shown(
        self, schema: str | None, name: str, owners: Collection[str] = ()
    ) -> bool:
        """Whether a ``retrieve`` result named the table *schema*.*name* is.

        *owners* are the schemas the catalog has a table *name* in. A name
        without a schema, on either side, stands for its one owner. With more
        than one owner it matches only another name without a schema: the
        database's search path picks between them, not this check.
        """
        wanted = name.lower()
        owner_set = {o.lower() for o in owners}
        only = next(iter(owner_set)) if len(owner_set) == 1 else ""
        target = (schema or "").lower() or only
        return any(
            shown == wanted and (s or only) == target for s, shown in self.shown_tables
        )
