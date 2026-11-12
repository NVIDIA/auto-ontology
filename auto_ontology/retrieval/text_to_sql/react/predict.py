# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``predict`` tool: answer a forecasting question with KumoRFM.

The graph is built over the tables ``retrieve`` has shown and linked with the
catalog joins it found, so the model scopes a prediction the same way it scopes
a query: by searching. A build that is refused (no usable tables, over budget)
comes back as an observation the model can act on; a prediction that ran is
the answer, already in the shape the SQL response takes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from auto_ontology.catalog.constants import Labels
from auto_ontology.dal.datasources import fetch_table_by_name
from auto_ontology.retrieval.text_to_sql.react import config
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
from auto_ontology.retrieval.text_to_sql.react.run_sql import connector_for

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """A KumoRFM answer, in the final-response shape the client renders."""

    response: dict[str, Any]


def _pql_tables(pql: str) -> set[str]:
    """Table names a PQL references: the left side of every ``TABLE.COLUMN``."""
    from auto_ontology.retrieval.kumo.pql_gen import _TABLE_COL

    return {m.group(1) for m in _TABLE_COL.finditer(pql or "")}


def _with_example_tables(
    tables: list[dict[str, Any]], examples: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """*tables* plus any table a PQL example references that is not among them.

    PQL generation is guided by the examples, so a table one of them names must
    be in the graph or the generated PQL cannot parse.
    """
    existing = {str(t.get("name") or "").upper() for t in tables}
    referenced: set[str] = set()
    for example in examples:
        referenced |= _pql_tables(example.get("query") or "")

    enriched = list(tables)
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
    return enriched


def _refused(ctx: ToolContext, message: str) -> tuple[str, None]:
    ctx.prediction_refusal = message
    return (
        f"Prediction not run: {message}\nRetrieve the tables it is missing "
        "and call predict again.",
        None,
    )


def predict(ctx: ToolContext) -> tuple[str, Prediction | None]:
    """Build the KumoRFM graph over the tables in view and predict."""
    if ctx.predict_calls >= config.max_predict_calls():
        return "No predict calls left in this episode.", None
    ctx.predict_calls += 1
    if not ctx.seen_tables:
        return (
            "Error: no tables to predict from. Retrieve the entity the question "
            "is about and the tables that hold its history first.",
            None,
        )

    from auto_ontology.retrieval.kumo import (
        PredictionContext,
        build_prediction_context,
        run_prediction,
    )
    from auto_ontology.retrieval.kumo.rag import fetch_pql_examples

    examples = fetch_pql_examples(ctx.retriever, ctx.question)
    tables = _with_example_tables(list(ctx.seen_tables.values()), examples)
    connector = connector_for(ctx)
    try:
        context = build_prediction_context(
            [connector] if connector is not None else ctx.connectors,
            tables,
            join_paths=ctx.join_paths,
            examples=examples,
        )
    except Exception as error:  # noqa: BLE001 — a failed build is an observation
        logger.exception("KumoRFM graph preparation failed")
        return _refused(
            ctx,
            "I couldn't prepare a prediction for this question. "
            f"({type(error).__name__}: {error})",
        )
    if not isinstance(context, PredictionContext):
        return _refused(ctx, str(context.get("response") or "the graph was refused"))

    try:
        response = run_prediction(ctx.question, ctx.prediction_llm, context)
    except Exception as error:  # noqa: BLE001
        logger.exception("KumoRFM prediction failed")
        response = {
            "response": "I couldn't produce a prediction for this question. "
            f"({type(error).__name__}: {error})",
            "sql_code": "",
            "sql_columns": [],
            "custom_analyses_used": [],
            "sql_response_from_db": None,
        }
    return "Prediction complete.", Prediction(response)
