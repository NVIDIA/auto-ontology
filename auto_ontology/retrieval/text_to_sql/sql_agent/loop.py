# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The construction turn loop.

Bounded on three axes at once — turns, database queries, and wall clock —
because they fail differently: a model can loop cheaply within one turn by
batching tool calls, burn the question's whole time budget on one slow probe,
or simply never decide to stop. Any of the three expiring ends the loop with
the best query seen so far rather than with nothing, since a mediocre answer
scores above an empty one.

Tools are declared as OpenAI function schemas and dispatched by name here
instead of through LangChain's tool-execution machinery. The loop needs to
treat ``submit_sql`` as terminal rather than as a call to make, and wants
every tool failure rendered back to the model as text, which is the opposite
of what an executor's exception handling does.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from auto_ontology.retrieval.text_to_sql.models import SQLGenerationModel
from auto_ontology.retrieval.text_to_sql.sql_agent import config
from auto_ontology.retrieval.text_to_sql.sql_agent.prompts import (
    FINAL_EXTRACTION_PROMPT,
    MISSING_TOOL_CALL_REMINDER,
    PROTOCOL_PROMPT,
    deadline_warning,
)
from auto_ontology.retrieval.text_to_sql.sql_agent.tools import SQLAgentToolbox
from auto_ontology.utils.llm_invoke import (
    safe_invoke_messages,
    safe_invoke_with_structured_output,
)

logger = logging.getLogger(__name__)

SUBMIT_TOOL = "submit_sql"
UNCONSTRUCTABLE_TOOL = "report_unconstructable"

# Warn the model once it is this close to the turn cap, so the last turn is
# spent submitting rather than being cut off mid-exploration.
_WARN_AT_TURNS_LEFT = 2

# Ceiling on one tool result as it goes back into the conversation. Generous
# enough that no current tool hits it, and low enough that eight turns of
# worst-case results still leave room for the schema block above them.
_MAX_TOOL_RESULT_CHARS = 6000


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


TOOL_SCHEMAS: list[dict] = [
    _fn(
        "run_sql",
        "Run one read-only SELECT against the real database and see the rows. "
        "Use it to confirm that a value, spelling or relationship is what you "
        "think it is before you depend on it.",
        {
            "sql": {
                "type": "string",
                "description": "A single SELECT or WITH query. No other statement is allowed.",
            },
            "purpose": {
                "type": "string",
                "description": "What this query is meant to establish, in a few words.",
            },
        },
        ["sql"],
    ),
    _fn(
        "check_sql",
        "Run the static validators your final query must pass — catalog parse, "
        "placeholder detection, and aggregation sanity — without executing it.",
        {"sql": {"type": "string", "description": "The draft query to check."}},
        ["sql"],
    ),
    _fn(
        "diagnose_sql",
        "Probe the live database for problems a static check cannot see: wrong "
        "filter literals, wrong numeric scale, truncating integer division, "
        "non-existent JSON key paths, and joins that follow no real "
        "relationship. Costs several queries.",
        {"sql": {"type": "string", "description": "The draft query to diagnose."}},
        ["sql"],
    ),
    _fn(
        "inspect_table",
        "Show one in-scope table's full schema and a few real rows. Use when "
        "the trimmed schema above may have dropped a column you need.",
        {"table_name": {"type": "string", "description": "Exact table name."}},
        ["table_name"],
    ),
    _fn(
        "search_tables",
        "Search for tables that are not in scope yet and add any match to the "
        "available schema.",
        {
            "query": {
                "type": "string",
                "description": "What the missing table would contain, in natural language.",
            }
        },
        ["query"],
    ),
    _fn(
        SUBMIT_TOOL,
        "Submit your final answer. This ends your turn.",
        {
            "thought": {
                "type": "string",
                "description": "How you arrived at this query and what you verified.",
            },
            "sql_code": {"type": "string", "description": "The final SQL query."},
            "response": {
                "type": "string",
                "description": "A one-sentence description of what the query returns.",
            },
        },
        ["thought", "sql_code", "response"],
    ),
    _fn(
        UNCONSTRUCTABLE_TOOL,
        "Declare that the available schema cannot answer the question. Only "
        "use this after looking — not because the first attempt was awkward.",
        {
            "reason": {
                "type": "string",
                "description": "What the question needs that the schema does not have.",
            }
        },
        ["reason"],
    ),
]


@dataclass
class SQLAgentOutcome:
    """What the loop produced, and why it stopped.

    ``result`` is ``None`` only when the agent declared the question
    unconstructable or every attempt to get SQL out of it failed; the caller
    maps that onto the graph's ``unconstructable`` decision.
    """

    result: Optional[SQLGenerationModel]
    stop_reason: str
    turns: int
    unconstructable_reason: str = ""
    trajectory: list[dict[str, Any]] = field(default_factory=list)


# Claude behind the LiteLLM/Bedrock gateway sometimes serialises a
# multi-argument tool call into the *first* argument's value, leaving the rest
# as literal markup inside it:
#
#   {"thought": "...grain is what links.</thought>
#                <parameter name=\"sql_code\">SELECT t.trans_id FROM ..."}
#
# Measured on a 70-question slice this hit 3% of questions, and because
# sql_code was the casualty every one of them was scored as "the agent could
# not construct a query" despite the agent having verified a working query the
# turn before. Same family of gateway leak that ``StrictLLMOutputModel``
# already strips ``$``-prefixed keys for.
_LEAKED_PARAM_RE = re.compile(
    r"<parameter\s+name=\"(?P<name>[A-Za-z_]+)\"\s*>"
    r"(?P<value>.*?)(?=</parameter>|<parameter\s+name=|\Z)",
    re.DOTALL,
)
_TRAILING_CLOSE_TAG_RE = re.compile(r"\s*</[A-Za-z_]+>\s*\Z")


def _recover_leaked_arguments(args: dict) -> tuple[dict, bool]:
    """Split a collapsed tool call back into its arguments.

    Returns the arguments and whether anything had to be recovered, so the
    caller can log how often the gateway is doing this. An argument that
    already arrived intact is never overwritten.
    """
    if not any(
        isinstance(v, str) and "<parameter" in v for v in args.values()
    ):
        return args, False

    recovered = dict(args)
    for key, value in args.items():
        if not isinstance(value, str) or "<parameter" not in value:
            continue
        head, _, tail = value.partition("<parameter")
        recovered[key] = _TRAILING_CLOSE_TAG_RE.sub("", head).strip()
        for match in _LEAKED_PARAM_RE.finditer(f"<parameter{tail}"):
            name = match.group("name")
            if str(recovered.get(name) or "").strip():
                continue
            recovered[name] = _TRAILING_CLOSE_TAG_RE.sub(
                "", match.group("value")
            ).strip()
    return recovered, True


def _truncate(text: str, limit: int = 1200) -> str:
    return text if len(text) <= limit else f"{text[:limit]}… [{len(text) - limit} more chars]"


def run_sql_agent(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    toolbox: SQLAgentToolbox,
) -> SQLAgentOutcome:
    """Drive the construction conversation to a final query.

    *messages* is the same prompt the single-call path builds; the protocol
    rules are appended here so the two differ only in the loop.
    """
    dispatch: dict[str, Callable[..., str]] = {
        "run_sql": toolbox.run_sql,
        "check_sql": toolbox.check_sql,
        "diagnose_sql": toolbox.diagnose_sql,
        "inspect_table": toolbox.inspect_table,
        "search_tables": toolbox.search_tables,
    }

    # Mid-loop feedback goes in as a *user* message, never a system one.
    # Bedrock hoists system messages out of the conversation, so a system
    # message appended after a reply that made no tool call leaves the
    # conversation ending on the assistant's turn, and the next call dies
    # with "the conversation must end with a user message" — which breaks
    # the loop and discards whatever the feedback was trying to fix. The
    # protocol below is the one genuine system instruction, and it is safe
    # there because the prompt it follows ends with the question.
    conversation: list[BaseMessage] = list(messages) + [
        SystemMessage(content=PROTOCOL_PROMPT)
    ]
    trajectory: list[dict[str, Any]] = []
    last_draft = ""
    deadline = time.monotonic() + config.deadline_s()
    max_turns = config.max_turns()
    bound = llm.bind_tools(TOOL_SCHEMAS)

    turn = 0
    bounces = 0
    while turn < max_turns:
        turn += 1
        turns_left = max_turns - turn
        if turns_left <= _WARN_AT_TURNS_LEFT:
            conversation.append(HumanMessage(content=deadline_warning(turns_left + 1)))

        try:
            reply = safe_invoke_messages(bound, conversation)
        except Exception as exc:  # noqa: BLE001 — a dead turn must not kill the question
            logger.error("SQL agent turn %d failed: %s", turn, exc, exc_info=True)
            trajectory.append({"turn": turn, "error": str(exc)})
            break

        conversation.append(reply)
        tool_calls = []
        for call in list(getattr(reply, "tool_calls", None) or []):
            args, leaked = _recover_leaked_arguments(call.get("args") or {})
            if leaked:
                logger.warning(
                    "Recovered collapsed arguments for %s on turn %d: %s",
                    call.get("name"),
                    turn,
                    sorted(args),
                )
            tool_calls.append({**call, "args": args})
        trajectory.append(
            {
                "turn": turn,
                "text": _truncate(str(getattr(reply, "content", "") or "")),
                "tool_calls": [
                    {"name": c.get("name"), "args": c.get("args")} for c in tool_calls
                ],
            }
        )

        if not tool_calls:
            conversation.append(HumanMessage(content=MISSING_TOOL_CALL_REMINDER))
            continue

        terminal: Optional[dict] = None
        for call in tool_calls:
            name = call.get("name") or ""
            args = call.get("args") or {}
            # Recorded before the call is dispatched, not after it succeeds:
            # a draft the model put up for checking is the query it is working
            # towards whether or not there was time left to check it, and it
            # is what _best_effort falls back on.
            if name in ("check_sql", "diagnose_sql"):
                draft = str(args.get("sql") or "").strip()
                if draft:
                    last_draft = draft
            if name in (SUBMIT_TOOL, UNCONSTRUCTABLE_TOOL):
                # Answered here rather than mid-list: a turn that both checks
                # something and submits should still get its check answered,
                # in case the submission turns out to be unusable.
                terminal = terminal or call
                conversation.append(
                    _tool_message(call, "Recorded.", name)
                )
                continue
            handler = dispatch.get(name)
            if handler is None:
                conversation.append(
                    _tool_message(call, f"No tool named {name!r}.", name)
                )
                continue
            if time.monotonic() >= deadline:
                conversation.append(
                    _tool_message(
                        call,
                        "Out of time — submit the best query you have now.",
                        name,
                    )
                )
                continue
            result = _call_tool(handler, args, name)
            conversation.append(_tool_message(call, result, name))
            trajectory[-1].setdefault("results", []).append(
                {"name": name, "result": _truncate(result)}
            )

        if terminal is not None:
            if (
                terminal.get("name") == SUBMIT_TOOL
                and bounces < config.max_submission_bounces()
                and time.monotonic() < deadline
            ):
                sql = str((terminal.get("args") or {}).get("sql_code") or "").strip()
                complaint = _gate_submission(
                    sql or last_draft, toolbox, first_pass=bounces == 0
                )
                if complaint:
                    bounces += 1
                    logger.info(
                        "Submission bounced (%d/%d) on turn %d",
                        bounces,
                        config.max_submission_bounces(),
                        turn,
                    )
                    trajectory[-1].setdefault("results", []).append(
                        {"name": "submission_gate", "result": _truncate(complaint)}
                    )
                    conversation.append(HumanMessage(content=complaint))
                    continue
            return _finish(terminal, turn, trajectory, last_draft)

        if time.monotonic() >= deadline:
            logger.info("SQL agent hit its deadline after %d turn(s)", turn)
            return _best_effort(
                llm, messages, toolbox, last_draft, "deadline", turn, trajectory
            )

    logger.info("SQL agent reached its turn cap (%d)", max_turns)
    return _best_effort(
        llm, messages, toolbox, last_draft, "turn_cap", turn, trajectory
    )


def _gate_submission(
    sql: str, toolbox: SQLAgentToolbox, *, first_pass: bool
) -> str:
    """What is wrong with a submitted query, or ``""`` to let it through.

    The model was free to run these checks itself and, measured over 70
    questions, called the live-probe one twice. Offering a check the model may
    decline is not a check, so the ones that catch scoring errors are applied
    to every submission here instead. The model still gets to do the fixing —
    it is handed the findings and keeps its remaining turns.
    """
    if not sql.strip():
        return ""

    problems: list[str] = []
    problems.extend(toolbox.static_findings(sql))

    # Executed before the elaboration checks rather than after, because one
    # of them reads the submission's own rows and there is no reason to run
    # the query twice.
    preview = toolbox.preview_result(sql)

    # Only on the first pass: the probe suite is the expensive half, and a
    # revision that fixed what it flagged should not pay for it again.
    if first_pass:
        problems.extend(toolbox.probe_findings(sql))
        problems.extend(toolbox.elaboration_findings(sql, preview))

    if not preview["ok"]:
        problems.append(
            f"Your query fails to execute: {preview['error']}"
        )
    elif preview["row_count"] == 0:
        problems.append(
            "Your query runs but returns no rows. The question implies there "
            "is an answer, so something in it excludes every row — most often "
            "a filter literal that does not match the stored value, or a join "
            "that no row satisfies. Find which one by removing filters until "
            "rows appear."
        )

    if not problems:
        return ""

    joined = "\n\n".join(problems)
    return (
        "Your submission was checked before being accepted, and it has "
        f"problems:\n\n{joined}\n\n"
        "Fix them and call `submit_sql` again. If after looking you are "
        "satisfied the query is right and the check is mistaken, submit the "
        "same query again and it will be accepted."
    )


def _tool_message(call: dict, content: str, name: str) -> ToolMessage:
    """Wrap a tool result for the conversation, capped in size.

    The renderers in ``tools`` already bound each result, but they bound them
    one at a time; eight turns of them accumulate in a single context window,
    and a question that overflows it is lost outright rather than answered
    badly.
    """
    return ToolMessage(
        content=_truncate(content, _MAX_TOOL_RESULT_CHARS),
        tool_call_id=call.get("id") or name,
        name=name,
    )


def _call_tool(handler: Callable[..., str], args: dict, name: str) -> str:
    """Invoke a tool, rendering any failure back to the model as text."""
    try:
        return handler(**args)
    except TypeError as exc:
        return f"Wrong arguments for {name}: {exc}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("SQL agent tool %s raised", name, exc_info=True)
        return f"{name} failed: {exc}"


def _finish(
    call: dict, turn: int, trajectory: list[dict], last_draft: str = ""
) -> SQLAgentOutcome:
    name = call.get("name")
    args = call.get("args") or {}
    if name == UNCONSTRUCTABLE_TOOL:
        return SQLAgentOutcome(
            result=None,
            stop_reason="unconstructable",
            turns=turn,
            unconstructable_reason=str(args.get("reason") or "").strip()
            or "The available schema cannot answer this question.",
            trajectory=trajectory,
        )

    sql_code = str(args.get("sql_code") or "").strip()
    if not sql_code and last_draft.strip():
        # A submission that arrives without its query still means the model
        # decided it was done, so the draft it last put through a check is a
        # far better reading of its intent than "unconstructable".
        logger.warning("Empty submission — falling back to the last checked draft")
        sql_code = last_draft.strip()
    if not sql_code:
        return SQLAgentOutcome(
            result=None,
            stop_reason="unconstructable",
            turns=turn,
            unconstructable_reason="The agent submitted an empty query.",
            trajectory=trajectory,
        )
    return SQLAgentOutcome(
        result=SQLGenerationModel(
            thought=str(args.get("thought") or "").strip() or "(no reasoning given)",
            sql_code=sql_code,
            response=str(args.get("response") or "").strip() or "Query result.",
        ),
        stop_reason="submitted",
        turns=turn,
        trajectory=trajectory,
    )


def _best_effort(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    toolbox: SQLAgentToolbox,
    last_draft: str,
    stop_reason: str,
    turn: int,
    trajectory: list[dict],
) -> SQLAgentOutcome:
    """Salvage an answer from a loop that ran out of budget.

    A draft the model already put through ``check_sql`` or ``diagnose_sql`` is
    taken as-is: it is the query it was working towards, and on the deadline
    path there is no time left to ask for another. Only when there is no draft
    at all do we spend one more call, over a flattened transcript rather than
    the tool-call history, which the gateway will not accept without the tool
    definitions still bound.
    """
    if last_draft.strip():
        logger.info("SQL agent returning its last checked draft (%s)", stop_reason)
        return SQLAgentOutcome(
            result=SQLGenerationModel(
                thought=f"Returned the last validated draft after {stop_reason}.",
                sql_code=last_draft.strip(),
                response="Query result.",
            ),
            stop_reason=stop_reason,
            turns=turn,
            trajectory=trajectory,
        )

    summary = _flatten(toolbox)
    final_messages = list(messages)
    if summary:
        final_messages.append(
            HumanMessage(content=f"What you checked so far:\n\n{summary}")
        )
    final_messages.append(HumanMessage(content=FINAL_EXTRACTION_PROMPT))
    try:
        result = safe_invoke_with_structured_output(
            llm, final_messages, SQLGenerationModel
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("SQL agent final extraction failed: %s", exc, exc_info=True)
        return SQLAgentOutcome(
            result=None,
            stop_reason=stop_reason,
            turns=turn,
            unconstructable_reason="The agent produced no query within its budget.",
            trajectory=trajectory,
        )
    return SQLAgentOutcome(
        result=result, stop_reason=stop_reason, turns=turn, trajectory=trajectory
    )


def _flatten(toolbox: SQLAgentToolbox) -> str:
    """Render the tool history as plain text for the fallback call."""
    blocks = []
    for entry in toolbox.calls:
        args = json.dumps(entry["args"], default=str)
        blocks.append(f"{entry['tool']}({_truncate(args, 400)}):\n{_truncate(entry['result'], 800)}")
    return "\n\n".join(blocks)


__all__ = ["run_sql_agent", "SQLAgentOutcome", "TOOL_SCHEMAS", "SUBMIT_TOOL"]
