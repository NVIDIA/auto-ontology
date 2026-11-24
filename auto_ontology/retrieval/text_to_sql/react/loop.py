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
from auto_ontology.retrieval.text_to_sql.react.grep import grep
from auto_ontology.retrieval.text_to_sql.react.information import information
from auto_ontology.retrieval.text_to_sql.react.mode import EpisodeMode, ToolName
from auto_ontology.retrieval.text_to_sql.react.predict import Prediction, predict
from auto_ontology.retrieval.text_to_sql.react.retrieve import retrieve
from auto_ontology.utils.llm_invoke import invoke_with_retry
from auto_ontology.retrieval.text_to_sql.react.run_sql import (
    check_query,
    connector_for,
    no_database_error,
    run_sql,
)

logger = logging.getLogger(__name__)

_MAX_CONSECUTIVE_NUDGES = 2
_TRAJECTORY_TEXT_CHARS = 2000


@dataclass
class Submission:
    """A query that passed the checks and ran, or an answer to an information question.

    An information answer has no ``sql`` and no ``db_response``.
    """

    sql: str
    response: str
    assumptions: str
    db_response: QueryResponse | None


@dataclass
class EpisodeResult:
    """How an episode ended, and everything the model saw and did."""

    submission: Submission | None
    reason: str
    tool_calls: int
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    prediction: Prediction | None = None


def submit(
    ctx: ToolContext,
    sql: str = "",
    response: str = "",
    assumptions: str = "",
    resubmit: str = "submit this version",
) -> tuple[str, Submission | None]:
    """Check and run a final query; the submission when it ran cleanly.

    A failed submission is an observation, not the end of the episode: the
    model sees why and can fix it. When the caller's budget can no longer pay
    for a query, the query is accepted after the schema checks without running.
    """
    sql = (sql or "").strip().rstrip(";")
    if not sql:
        if not prompts.information_answers_offered(ctx.information_answers):
            return "Error: sql is empty.", None
        response = (response or "").strip()
        if not response:
            return "Error: sql and response are both empty.", None
        if not ctx.information_retrieved:
            return (
                "Not accepted. An answer without SQL has to come from the catalog: "
                "retrieve what you will say first (purpose=information), or write "
                "a query.",
                None,
            )
        return "Accepted.", Submission("", response, assumptions, None)
    connector = connector_for(ctx, sql)
    if connector is None:
        return no_database_error(ctx, sql), None
    check = check_query(ctx, connector, sql)
    if check.problems:
        return "Not accepted. Fix these first:\n- " + "\n- ".join(check.problems), None
    if not _can_run_queries(ctx):
        # Running it would be refused, and a refusal is not a wrong query.
        unrun = QueryResponse(result=[], sliced=False)
        submission = Submission(sql, (response or "").strip(), assumptions, unrun)
        ctx.submitted_database = getattr(connector, "database_name", None)
        return "Accepted without running it: no budget left for a query.", submission
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
            f"Not accepted. The query only runs as rewritten here; {resubmit} "
            f"instead:\n{db_response.sql_correction}"
        ), None
    submission = Submission(sql, (response or "").strip(), assumptions, db_response)
    ctx.submitted_database = getattr(connector, "database_name", None)
    return "Accepted.", submission


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _call_key(name: str, args: dict) -> str:
    return f"{name}:{json.dumps(args, sort_keys=True, default=str)}"


def _retrieve(
    ctx: ToolContext, query: str, kinds: list[str] | None, purpose: str
) -> str:
    """The SQL view, or the information view when information answers are on."""
    if purpose == "information" and prompts.information_answers_offered(
        ctx.information_answers
    ):
        answer = information(ctx, query, kinds)
        # Only once the lookup has returned: a lookup that raised must not let a
        # later answer without SQL pass as one that came from the catalog. One
        # that found nothing did return, and still counts.
        ctx.information_retrieved = True
        return answer
    return retrieve(ctx, query, kinds)


def _can_run_queries(ctx: ToolContext) -> bool:
    return ctx.can_run_queries is None or ctx.can_run_queries()


def _submit_draft(
    ctx: ToolContext, draft: str | None, calls: int, trajectory: list[dict[str, Any]]
) -> EpisodeResult | None:
    """Submit the latest drafted query when the budget ended the episode."""
    if not draft:
        return None
    observation, submission = submit(ctx, draft)
    trajectory.append({"role": "draft_submission", "content": observation})
    if submission is None:
        return None
    return EpisodeResult(submission, "submitted_draft", calls, trajectory)


def _with_cost_status(ctx: ToolContext, text: str) -> str:
    status = ctx.cost_status() if ctx.cost_status else None
    return f"{text}\n\n{status}" if status else text


def _dispatch(
    ctx: ToolContext, name: str, args: dict, offered: list[str]
) -> tuple[str, Submission | Prediction | None]:
    handlers: dict[str, Callable[..., str]] = {
        ToolName.RETRIEVE: lambda query="", kinds=None, purpose="sql": _retrieve(
            ctx, query, kinds, purpose
        ),
        ToolName.RUN_SQL: lambda sql="": run_sql(ctx, sql),
        ToolName.GREP: lambda page=0, pattern="", offset=0: grep(
            ctx, page, pattern, offset
        ),
    }
    try:
        if name not in offered:
            return f"Unknown tool {name!r}. Use {', '.join(offered)}.", None
        if name == ToolName.SUBMIT:
            return submit(ctx, **args)
        if name == ToolName.RUN_SQL and prompts.is_answer(name, args):
            final_args = {k: v for k, v in args.items() if k != "final"}
            return submit(
                ctx, **final_args, resubmit="run this version with final=true"
            )
        if name == ToolName.RUN_SQL and not config.submit_tool_enabled():
            args = {k: v for k, v in args.items() if k == "sql"}
        if name == ToolName.PREDICT:
            return predict(ctx)
        return handlers[name](**args), None
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
    mode: EpisodeMode = EpisodeMode.SQL,
    custom_prompts: str | None = None,
    sql_examples: list[dict[str, str]] | None = None,
    value_anchors: list[dict[str, str]] | None = None,
    shorten_answer: bool = False,
) -> EpisodeResult:
    """Run the model with *mode*'s tools until it answers or the budget ends.

    ``sql`` answers with ``submit``, ``predict`` with ``predict``, and ``auto``
    offers both and lets the model choose. *custom_prompts* is the
    organization's sections from :func:`prompts.custom_prompts_section`.
    *sql_examples*, *value_anchors* and *shorten_answer* shape a SQL answer, so
    a ``predict`` episode ignores them.
    """
    writes_sql = mode != EpisodeMode.PREDICT
    system_prompt = prompts.system_prompt_for(
        mode, dialect, information_answers=ctx.information_answers
    )
    if custom_prompts:
        system_prompt = f"{system_prompt}\n\n{custom_prompts}"
    # Dialect rules, e.g. for MySQL: do not CAST or ROUND computed values,
    # which changes the digits an exact grader compares.
    dialect_rules = format_dialect_rules(dialect).strip() if writes_sql else ""
    if dialect_rules:
        system_prompt = f"{system_prompt}\n\n{dialect_rules}"
    if shorten_answer and writes_sql:
        system_prompt = f"{system_prompt}\n\n{prompts.SHORT_ANSWER_RULES}"
    user_message = prompts.user_message(
        question,
        evidence,
        value_anchors=value_anchors if writes_sql else None,
        sql_examples=sql_examples if writes_sql else None,
    )
    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=_with_cost_status(ctx, user_message)),
    ]
    trajectory: list[dict[str, Any]] = [
        {"role": m.type, "content": m.content} for m in messages
    ]
    tools = prompts.tools_for(mode, information_answers=ctx.information_answers)
    offered = [t["function"]["name"] for t in tools]
    bound = llm.bind_tools(tools)
    budget = config.max_tool_calls()
    calls = 0
    nudges = 0
    made: set[str] = set()
    draft: str | None = None
    out_of_queries = False

    while True:
        last_turn = calls >= budget
        if (
            mode != EpisodeMode.PREDICT
            and not out_of_queries
            and not _can_run_queries(ctx)
        ):
            out_of_queries = True
            no_queries_left = prompts.no_queries_left_for()
            messages.append(HumanMessage(content=no_queries_left))
            trajectory.append({"role": "human", "content": no_queries_left})
            tools = prompts.tools_for(
                mode, last_turn=True, information_answers=ctx.information_answers
            )
            offered = [t["function"]["name"] for t in tools]
            bound = llm.bind_tools(tools)
        if last_turn:
            last_call = prompts.last_call_for(
                mode, information_answers=ctx.information_answers
            )
            messages.append(HumanMessage(content=last_call))
            trajectory.append({"role": "human", "content": last_call})
            tools = prompts.tools_for(
                mode, last_turn=True, information_answers=ctx.information_answers
            )
            offered = [t["function"]["name"] for t in tools]
            bound = llm.bind_tools(tools)
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
                if out_of_queries:
                    fallback = _submit_draft(ctx, draft, calls, trajectory)
                    if fallback is not None:
                        return fallback
                return EpisodeResult(None, "no_submission", calls, trajectory)
            nudge = prompts.nudge_for(mode, information_answers=ctx.information_answers)
            messages.append(HumanMessage(content=nudge))
            trajectory.append({"role": "human", "content": nudge})
            continue
        nudges = 0
        answered_over_budget = False

        for call in tool_calls:
            name = call.get("name") or ""
            args = call.get("args") or {}
            if (
                name in (ToolName.RUN_SQL, ToolName.SUBMIT)
                and (args.get("sql") or "").strip()
            ):
                draft = args["sql"]
            key = _call_key(name, args)
            answer: Submission | Prediction | None
            answers = prompts.is_answer(name, args)
            if calls >= budget and (not answers or answered_over_budget):
                # Past the budget a reply gets one answer attempt; each attempt
                # can run SQL, so further ones in the same reply are refused.
                observation, answer = (prompts.out_of_calls_for(mode), None)
            elif key in made and not answers:
                # Counted though not run: otherwise a model that repeats one
                # call never reaches the budget and the episode never ends.
                calls += 1
                observation, answer = (
                    "You already made this exact call; its result is above.",
                    None,
                )
            else:
                if calls >= budget:
                    answered_over_budget = True
                made.add(key)
                calls += 1
                observation, answer = _dispatch(ctx, name, args, offered)
            if answer is None:
                observation = _with_cost_status(ctx, observation)
            messages.append(
                ToolMessage(content=observation, tool_call_id=call.get("id") or name)
            )
            trajectory.append({"role": "tool", "name": name, "content": observation})
            if isinstance(answer, Submission):
                return EpisodeResult(answer, "submitted", calls, trajectory)
            if isinstance(answer, Prediction):
                return EpisodeResult(
                    None, "predicted", calls, trajectory, prediction=answer
                )

        if last_turn:
            if out_of_queries:
                fallback = _submit_draft(ctx, draft, calls, trajectory)
                if fallback is not None:
                    return fallback
            return EpisodeResult(None, "budget_exhausted", calls, trajectory)
