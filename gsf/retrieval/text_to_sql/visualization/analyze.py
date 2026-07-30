# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestrate LLM chart recommendations → ResultChart specs (no image rendering)."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from gsf.retrieval.text_to_sql.visualization.chart_specs import generate_chart_specs
from gsf.retrieval.text_to_sql.visualization.helpers import (
    parse_sql_response_to_dataframe,
)
from gsf.retrieval.text_to_sql.visualization.prompts import (
    IS_TABLE_REQUEST_PROMPT,
    VISUALIZATION_PROMPT,
    IsTableRequestModel,
    VisualizationRecommendation,
)
from gsf.utils.llm_invoke import invoke_with_structured_output

logger = logging.getLogger(__name__)


def analyze_and_visualize(
    llm: BaseChatModel,
    question: str,
    sql: str,
    sql_response_from_db: Any,
) -> list[dict[str, Any]] | None:
    """Recommend charts for an SQL result and return ResultChart JSON specs.

    Returns ``None`` when visualization should be skipped (empty data, user
    asked for a table, LLM failure, etc.). Never raises.
    """
    try:
        df = parse_sql_response_to_dataframe(sql_response_from_db)
        if df is None or df.empty:
            logger.info("analyze_and_visualize: no tabular data — skip")
            return None

        # Explicit table request → no charts.
        table_messages = [
            SystemMessage(content="You are a data visualization expert."),
            HumanMessage(
                content=IS_TABLE_REQUEST_PROMPT.format(question=question),
            ),
        ]
        table_check = invoke_with_structured_output(
            llm, table_messages, IsTableRequestModel
        )
        if table_check is not None and table_check.is_table_request:
            logger.info("analyze_and_visualize: user requested table — skip")
            return None

        viz_messages = [
            SystemMessage(content="You are a data visualization expert."),
            HumanMessage(
                content=VISUALIZATION_PROMPT.format(
                    sql=sql,
                    question=question,
                    df_size=len(df),
                    df_columns=list(df.columns),
                ),
            ),
        ]
        recommendation = invoke_with_structured_output(
            llm, viz_messages, VisualizationRecommendation
        )
        if recommendation is None or not recommendation.plots:
            logger.info("analyze_and_visualize: no plot recommendations — skip")
            return None

        specs = generate_chart_specs(df, recommendation.plots)
        if not specs:
            logger.info("analyze_and_visualize: no valid chart specs built — skip")
            return None
        # One primary chart only (avoid bar+line duplicates for the same data).
        return specs[:1]
    except Exception as exc:  # noqa: BLE001 — soft-fail the whole viz step
        logger.error("analyze_and_visualize failed: %s", exc)
        return None
