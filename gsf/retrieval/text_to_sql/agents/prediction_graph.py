# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
KumoRFM graph-preparation node.

The prediction path is split into two nodes so the (potentially slow) infra
step streams its own progress:

  1. ``prepare_prediction_graph`` (this agent) — load the relevant tables, build
     the KumoRFM ``LocalGraph`` + model, and stash the resulting
     :class:`~gsf.retrieval.kumo.predictor.PredictionContext` in ``path_state``.
  2. ``kumo_predict`` — generate/repair the PQL, predict, and format.

On failure (no relevant tables, KumoRFM unavailable on this platform, etc.) it
writes a graceful ``final_response`` and signals ``predict_failed`` so the graph
routes straight to END without attempting a prediction.
"""

import logging
from typing import Any, Dict

from langchain_core.messages import AIMessage
from gsf.catalog.constants import Labels

from gsf.dal.datasources import fetch_table_by_name
from gsf.retrieval.kumo import PredictionContext, build_prediction_context
from gsf.retrieval.kumo.pql_gen import _TABLE_COL
from gsf.retrieval.kumo.rag import fetch_pql_examples
from gsf.retrieval.text_to_sql.base import BaseAgent
from gsf.retrieval.text_to_sql.state import AgentState, get_standalone_question

logger = logging.getLogger(__name__)


def _error_response(message: str) -> Dict[str, Any]:
    return {
        "response": message,
        "sql_code": "",
        "sql_columns": [],
        "custom_analyses_used": [],
        "sql_response_from_db": None,
    }


def _pql_tables(pql: str) -> set[str]:
    """Table names referenced in a PQL (the left side of every ``TABLE.COLUMN``)."""
    return {m.group(1) for m in _TABLE_COL.finditer(pql or "")}


def _enrich_relevant_tables(
    relevant_tables: list[dict[str, Any]],
    examples: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Add any table referenced by a retrieved PQL example but missing from
    ``relevant_tables``, resolved from the catalog by name.

    Done BEFORE the graph is built so the added tables are loaded into the graph
    (and auto-linked by metadata inference) — the LLM is guided by these examples,
    so every table they reference must exist in the graph or the PQL won't parse.
    """
    existing = {str(t.get("name") or "").upper() for t in relevant_tables}
    referenced: set[str] = set()
    for ex in examples:
        referenced |= _pql_tables(ex.get("query") or "")

    enriched = list(relevant_tables)
    added: list[str] = []
    for name in sorted(referenced):
        if name.upper() in existing:
            continue
        row = fetch_table_by_name(name)
        if not row or not row.get("name"):
            continue
        enriched.append(
            {
                "id": row.get("id"),
                "name": row.get("name"),
                "schema_name": row.get("schema_name") or "",
                "description": row.get("description") or "",
                "pk": row.get("pk"),
                "label": Labels.TABLE,
            }
        )
        existing.add(name.upper())
        added.append(row["name"])
    if added:
        logger.info(
            "kumo: enriched graph with %d table(s) from PQL examples: %s",
            len(added),
            added,
        )
    return enriched


class PredictionGraphAgent(BaseAgent):
    """Build the KumoRFM graph/model for the relevant tables (prediction phase 1)."""

    def __init__(self):
        super().__init__("prediction_graph")

    def execute(self, state: AgentState) -> Dict[str, Any]:
        path_state = state.get("path_state", {})
        connectors = state.get("connectors", []) or []
        # Scope the KumoRFM graph to the tables candidate-preparation found relevant,
        # and use the catalog-derived join paths as the graph's table relationships.
        relevant_tables = path_state.get("relevant_tables") or []
        join_paths = path_state.get("attribute_join_paths") or []
        # Few-shot PQL examples retrieved from the verified PqlAnalysis corpus.
        examples = fetch_pql_examples(
            state.get("semantic_retriever"), get_standalone_question(state)
        )
        # Enrich the table set with any table the examples reference before the
        # graph is built, so the LLM can never cite a table absent from the graph.
        relevant_tables = _enrich_relevant_tables(relevant_tables, examples)

        try:
            context = build_prediction_context(
                connectors,
                relevant_tables,
                join_paths=join_paths,
                examples=examples,
            )
        except Exception as exc:
            self.logger.exception("KumoRFM graph preparation failed")
            context = _error_response(
                "I couldn't prepare a prediction for this question. "
                f"({type(exc).__name__}: {exc})"
            )

        if isinstance(context, PredictionContext):
            return {
                "decision": "predict_ready",
                "path_state": {**path_state, "prediction_context": context},
            }

        # build_prediction_context returned a graceful error response dict —
        # terminate the prediction path with it.
        markdown = context.get("response", "")
        return {
            "decision": "predict_failed",
            "messages": state["messages"] + [AIMessage(content=markdown)],
            "path_state": {
                **path_state,
                "formatted_response": markdown,
                "final_response": context,
            },
        }
