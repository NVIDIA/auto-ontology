# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One ReAct episode: the model calls tools until it submits a query.

Plain Python, so the same function serves a request in
the pipeline and a rollout in an RL environment. Every message the model sees is
built here and recorded in the trajectory, so a trajectory is exactly what was
sent — which is what training on it needs.

The model alone decides what to search, what to run and when it is done. The
loop only enforces the budget, refuses repeated calls, and turns a submission
into an answer once it runs cleanly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from auto_ontology.retrieval.text_to_sql.agents.sql_execution import (
    QueryResponse,
    _run_sql,
)
from auto_ontology.retrieval.text_to_sql.prompts import format_dialect_rules
from auto_ontology.retrieval.text_to_sql.react import config, prompts
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
from auto_ontology.retrieval.text_to_sql.react.retrieve import retrieve
from auto_ontology.utils.llm_invoke import invoke_with_retry
from auto_ontology.retrieval.text_to_sql.react.run_sql import (
    catalog_for,
    check_sql,
    connector_for,
    run_sql,
)

logger = logging.getLogger(__name__)

_MAX_CONSECUTIVE_NUDGES = 2
_TRAJECTORY_TEXT_CHARS = 2000


@dataclass
class Submission:
    """A query that passed the checks and ran."""

    sql: str
    response: str
    assumptions: str
    db_response: QueryResponse


@dataclass
class EpisodeResult:
    """How an episode ended, and everything the model saw and did."""

    submission: Submission | None
    reason: str
    tool_calls: int
    trajectory: list[dict[str, Any]] = field(default_factory=list)


def submit(
    ctx: ToolContext, sql: str, response: str = "", assumptions: str = ""
) -> tuple[str, Submission | None]:
    """Check and run a final query; the submission when it ran cleanly.

    A failed submission is an observation, not the end of the episode: the
    model sees why and can fix it.
    """
    sql = (sql or "").strip().rstrip(";")
    if not sql:
        return "Error: sql is empty.", None
    connector = connector_for(ctx)
    if connector is None:
        return "Error: no database to run on. Retrieve the tables you need first.", None
    check = check_sql(sql, getattr(connector, "dialect", None), catalog_for(ctx))
    if check.problems:
        return "Not accepted. Fix these first:\n- " + "\n- ".join(check.problems), None
    db_response = _run_sql(
        sql, connector, record_numeric_kinds=ctx.numeric_type_alignment
    )
    if db_response.error:
        return (
            f"Not accepted. The database returned an error: {db_response.error}",
            None,
        )
    if db_response.sql_correction:
        return (
            "Not accepted. The query only runs as rewritten here; submit this "
            f"version instead:\n{db_response.sql_correction}"
        ), None
    submission = Submission(sql, (response or "").strip(), assumptions, db_response)
    return "Accepted.", submission


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _call_key(name: str, args: dict) -> str:
    return f"{name}:{json.dumps(args, sort_keys=True, default=str)}"


def _dispatch(ctx: ToolContext, name: str, args: dict) -> tuple[str, Submission | None]:
    handlers: dict[str, Callable[..., str]] = {
        "retrieve": lambda query="", kinds=None: retrieve(ctx, query, kinds),
        "run_sql": lambda sql="": run_sql(ctx, sql),
    }
    try:
        if name == "submit":
            return submit(ctx, **args)
        handler = handlers.get(name)
        if handler is None:
            return f"Unknown tool {name!r}. Use retrieve, run_sql or submit.", None
        return handler(**args), None
    except TypeError as error:
        return f"Wrong arguments for {name}: {error}", None
    except Exception as error:  # noqa: BLE001 — a failing tool must not end the question
        logger.warning("ReAct tool %s raised", name, exc_info=True)
        return f"{name} failed: {error}", None


def run_episode(
    llm: BaseChatModel,
    ctx: ToolContext,
    question: str,
    evidence: str | None = None,
    dialect: str | None = None,
) -> EpisodeResult:
    """Run the model with the three tools until it submits or the budget ends."""
    # Dialect rules, e.g. for MySQL: do not CAST or ROUND computed values,
    # which changes the digits an exact grader compares.
    system_prompt = prompts.SYSTEM_PROMPT.format(dialect=dialect or "SQL")
    dialect_rules = format_dialect_rules(dialect).strip()
    if dialect_rules:
        system_prompt = f"{system_prompt}\n\n{dialect_rules}"
    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompts.user_message(question, evidence)),
    ]
    trajectory: list[dict[str, Any]] = [
        {"role": m.type, "content": m.content} for m in messages
    ]
    bound = llm.bind_tools(prompts.TOOLS)
    budget = config.max_tool_calls()
    calls = 0
    nudges = 0
    made: set[str] = set()

    while True:
        last_turn = calls >= budget
        if last_turn:
            messages.append(HumanMessage(content=prompts.LAST_CALL))
            trajectory.append({"role": "human", "content": prompts.LAST_CALL})
            bound = llm.bind_tools([prompts.SUBMIT_TOOL])
        try:
            reply = invoke_with_retry(
                bound, messages, max_attempts=config.model_retries()
            )
        except Exception as error:  # noqa: BLE001
            logger.error("ReAct turn failed: %s", error, exc_info=True)
            trajectory.append({"role": "error", "content": str(error)})
            return EpisodeResult(None, "model_error", calls, trajectory)

        messages.append(reply)
        tool_calls = list(getattr(reply, "tool_calls", None) or [])
        trajectory.append(
            {
                "role": "ai",
                "content": _clip(str(reply.content or ""), _TRAJECTORY_TEXT_CHARS),
                "tool_calls": [
                    {"name": c["name"], "args": c["args"]} for c in tool_calls
                ],
            }
        )

        if not tool_calls:
            nudges += 1
            if last_turn or nudges > _MAX_CONSECUTIVE_NUDGES:
                return EpisodeResult(None, "no_submission", calls, trajectory)
            messages.append(HumanMessage(content=prompts.NUDGE))
            trajectory.append({"role": "human", "content": prompts.NUDGE})
            continue
        nudges = 0

        for call in tool_calls:
            name = call.get("name") or ""
            args = call.get("args") or {}
            key = _call_key(name, args)
            if key in made and name != "submit":
                observation, submission = (
                    "You already made this exact call; its result is above.",
                    None,
                )
            elif calls >= budget and name != "submit":
                observation, submission = ("No tool calls left; submit.", None)
            else:
                made.add(key)
                calls += 1
                observation, submission = _dispatch(ctx, name, args)
            messages.append(
                ToolMessage(content=observation, tool_call_id=call.get("id") or name)
            )
            trajectory.append({"role": "tool", "name": name, "content": observation})
            if submission is not None:
                return EpisodeResult(submission, "submitted", calls, trajectory)

        if last_turn:
            return EpisodeResult(None, "budget_exhausted", calls, trajectory)
