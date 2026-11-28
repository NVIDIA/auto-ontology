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
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from auto_ontology.catalog.constants import Labels
from auto_ontology.dal.datasources import fetch_table_by_name
from auto_ontology.retrieval.text_to_sql.react import config
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
from auto_ontology.retrieval.text_to_sql.react.run_sql import connector_for
from auto_ontology.utils.retriever import get_semantic_objects_retriever

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


# Recognize explicit labels independently of the planner. For less specific
# phrases, require a date-like continuation so "forecast from sales data" stays undated.
_DATE_LIKE = (
    r"(?:[0-9]|today\b|tomorrow\b|yesterday\b|now\b|(?:last|next|this)\s+"
    r"(?:day|week|month|year|quarter)\b|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+[0-9])"
)
_ORIGIN_MARKER = re.compile(
    r"\b(?:(?P<context>context\s+(?:origin|cutoff|as[- ]of)|data\s+(?:available|known)\s+as[- ]of)"
    r"|(?:forecast|prediction)\s+origin"
    rf"|(?:starting\s+from|(?:forecast|predict(?:ion)?)\s+from)(?=\s+{_DATE_LIKE})"
    rf"|(?P<as_of>as[- ]of)(?=\s+{_DATE_LIKE}))\b"
    r"\s*(?:(?:is|of|on)\b\s*|[:=]\s*)?",
    re.IGNORECASE,
)
_ORIGIN_DATE = r"(?<![\w-])[0-9]{4}-[0-9]{2}-[0-9]{2}(?![\w/-]| +[0-9]{1,2}:)"
_FILTER_CLAUSE = re.compile(r"\b(?:where|whose|who|which|that)\b", re.IGNORECASE)
_SECOND_DATE = re.compile(
    rf"\s*(?:,|/|to\s|or\s|and\s|through\s)\s*{_DATE_LIKE}", re.IGNORECASE
)


def _prediction_origins(
    question: str, forecast_origin: str | None, context_origin: str | None
) -> dict[str, pd.Timestamp]:
    candidates: dict[str, set[str]] = {"forecast": set(), "context": set()}
    for marker in _ORIGIN_MARKER.finditer(question):
        if marker.group("as_of"):
            clause = question[: marker.start()].rsplit(",", 1)[-1]
            # A relative clause can describe the input population as of a date.
            # Prefer an explicit origin label when a date could serve either role.
            if _FILTER_CLAUSE.search(clause.lstrip(), 1):
                continue
        kind = "context" if marker.group("context") else "forecast"
        tail = question[marker.end() :]
        date = re.match(_ORIGIN_DATE, tail)
        if date is None:
            raise ValueError(f"Specify the {kind} origin as a YYYY-MM-DD date.")
        if _SECOND_DATE.match(tail[date.end() :]):
            raise ValueError(f"Specify just one {kind} origin for this prediction.")
        candidates[kind].add(date[0])

    origins = {}
    for kind, value, key in (
        ("forecast", forecast_origin, "anchor_time"),
        ("context", context_origin, "context_anchor_time"),
    ):
        if value is not None:
            if not isinstance(value, str) or not re.fullmatch(_ORIGIN_DATE, value):
                raise ValueError(f"Specify the {kind} origin as a YYYY-MM-DD date.")
            if value not in re.findall(_ORIGIN_DATE, question):
                raise ValueError(f"The {kind} origin must be stated in the question.")
            candidates[kind].add(value)
        if len(candidates[kind]) > 1:
            raise ValueError(f"Specify just one {kind} origin for this prediction.")
        if candidates[kind]:
            try:
                origins[key] = pd.Timestamp(candidates[kind].pop()).as_unit("ns")
            except (ValueError, OverflowError) as exc:
                raise ValueError(
                    f"The {kind} origin is invalid or outside the supported date range."
                ) from exc
    if "context_anchor_time" in origins:
        if "anchor_time" not in origins:
            raise ValueError("Specify a forecast origin alongside the context cutoff.")
        if origins["context_anchor_time"] > origins["anchor_time"]:
            raise ValueError(
                "The context cutoff cannot be later than the forecast origin."
            )
    return origins


def predict(
    ctx: ToolContext,
    *,
    forecast_origin: str | None = None,
    context_origin: str | None = None,
) -> tuple[str, Prediction | None]:
    """Build the KumoRFM graph over the tables in view and predict."""
    if ctx.predict_calls >= config.max_predict_calls():
        return "No predict calls left in this episode.", None
    ctx.predict_calls += 1
    try:
        origins = _prediction_origins(ctx.question, forecast_origin, context_origin)
    except ValueError as exc:
        ctx.prediction_refusal = str(exc)
        return f"Prediction not run: {exc}", None
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

    examples = fetch_pql_examples(get_semantic_objects_retriever(), ctx.question)
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
        response = run_prediction(
            ctx.question,
            ctx.prediction_llm,
            context,
            organization=ctx.custom_prompts,
            **origins,
        )
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
