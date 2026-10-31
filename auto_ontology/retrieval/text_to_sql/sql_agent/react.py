# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A plain ReAct agent holding what the deep agent holds, minus deepagents.

The deep agent is a ReAct loop wrapped in a framework this task does not use:
it never plans with ``write_todos``, never delegates, and has its filesystem
tools stripped. What it does use is its own tool set, its own protocol and its
own clean-up of the answer. This module runs exactly those on the bounded
tool-calling loop in :mod:`.loop`, so the two arms differ in the framework and
nothing else:

* the same six tools, with the same descriptions (:func:`.deep.build_tools`,
  ``repair_sql`` included, which the plain loop never had);
* the same protocol (:func:`.deep.protocol_prompt`) and role line;
* the same system stack -- role line first, then the merged prompt -- so the
  endpoint's prefix cache works the same;
* the same forced tool calls (``tool_choice="required"``) and no
  ``report_unconstructable`` exit, which the deep agent does not have;
* the same identifier-quoting clean-up before the gate, and the same gate;
* the same final tool, :class:`SQLGenerationModel` as LangChain renders it,
  answered with the same "Returning structured response" reply.

What remains different is the loop itself: a turn cap of 30
(:func:`.config.react_max_turns`) and a 600 s budget
(:func:`.config.react_deadline_s`) replace the LangGraph recursion limit, and
the loop sends no turn-cap warnings, as the framework sends none.

Every message the model sees is built here rather than by a framework, so a
trajectory is exactly what was sent, which is what training on it needs.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.structured_output import _SchemaSpec
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from auto_ontology.retrieval.text_to_sql.models import SQLGenerationModel
from auto_ontology.retrieval.text_to_sql.sql_agent import config
from auto_ontology.retrieval.text_to_sql.sql_agent.deep import (
    ROLE_PROMPT,
    _merge_leading_system,
    _normalise,
    build_tools,
    has_analyses,
    protocol_prompt,
)
from auto_ontology.retrieval.text_to_sql.sql_agent.loop import (
    SQLAgentOutcome,
    run_sql_agent,
)
from auto_ontology.retrieval.text_to_sql.sql_agent.tools import SQLAgentToolbox

_ANSWER_TOOL = SQLGenerationModel.__name__

# The deep agent ends by calling a hidden tool LangChain builds from
# SQLGenerationModel: named after it, described by its docstring, with its
# fields. The model reads all three, so the loop's final tool is that same
# schema rather than submit_sql. report_unconstructable is left out because the
# deep agent has no such exit.
_TERMINAL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            **convert_to_openai_tool(SQLGenerationModel)["function"],
            "description": _SchemaSpec(SQLGenerationModel).description,
        },
    }
]


def _answer_reply(args: dict) -> str:
    """The tool result LangChain sends the deep agent for its answer."""
    try:
        answer = SQLGenerationModel(**args)
    except Exception:  # noqa: BLE001
        return "Recorded."
    return f"Returning structured response: {answer}"


def react_tools(toolbox: SQLAgentToolbox, state: Any) -> tuple[list[dict], dict]:
    """The deep agent's tools as OpenAI schemas, and the handler for each."""
    tools = build_tools(toolbox, state)
    schemas = [convert_to_openai_tool(tool) for tool in tools] + _TERMINAL_SCHEMAS
    handlers = {tool.name: tool.func for tool in tools}
    return schemas, handlers


def run_react_sql_agent(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    toolbox: SQLAgentToolbox,
    state: Any,
) -> SQLAgentOutcome:
    """Construct SQL with the deep agent's tools and protocol on a plain loop."""
    schemas, handlers = react_tools(toolbox, state)
    # Role line as its own system message and tool calls forced, as the deep
    # agent sends them: deepagents places its system prompt ahead of the merged
    # stack and its structured answer binds with tool_choice="required".
    return run_sql_agent(
        llm,
        [SystemMessage(content=ROLE_PROMPT), *_merge_leading_system(messages)],
        toolbox,
        tool_schemas=schemas,
        handlers=handlers,
        protocol=protocol_prompt(include_analyses=has_analyses(state)),
        normalise=lambda sql: _normalise(sql, toolbox),
        tool_choice="required",
        submit_tool=_ANSWER_TOOL,
        submit_reply=_answer_reply,
        turn_warnings=False,
        max_turns=config.react_max_turns(),
        deadline_s=config.react_deadline_s(),
    )


__all__ = ["react_tools", "run_react_sql_agent"]
