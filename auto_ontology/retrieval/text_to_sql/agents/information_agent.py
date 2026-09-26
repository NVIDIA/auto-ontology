# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool-using agent that answers catalog and semantic-metadata questions."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any, Dict

from langchain_core.messages import AIMessage, SystemMessage
from pydantic import Field, model_validator

from auto_ontology.retrieval.text_to_sql.agents.information_tools import (
    InformationToolName,
    run_information_tool,
)
from auto_ontology.retrieval.text_to_sql.base import BaseAgent
from auto_ontology.retrieval.text_to_sql.state import (
    AgentState,
    get_question_for_processing,
)
from auto_ontology.utils.llm_invoke import (
    StrictLLMOutputModel,
    invoke_with_structured_output,
)

MAX_TOOL_CALLS = 6
MAX_AGENT_ITERATIONS = 8
MAX_OBSERVATION_CHARS = 12_000
MAX_CANDIDATES_PER_KIND = 12


class InformationActionType(StrEnum):
    """Possible next steps selected by the metadata agent."""

    TOOL = "tool"
    ANSWER = "answer"


class InformationToolArguments(StrictLLMOutputModel):
    """The bounded set of identifiers accepted by metadata tools."""

    dataset_id: str | None = None
    dataset_name: str | None = None
    table_id: str | None = None
    table_name: str | None = None
    column_id: str | None = None
    column_name: str | None = None
    database_name: str | None = None


class InformationAgentAction(StrictLLMOutputModel):
    """One structured action in the information-agent loop."""

    action: InformationActionType
    tool: InformationToolName | None = None
    arguments: InformationToolArguments | None = None
    answer: str | None = None
    source_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_action_fields(self) -> InformationAgentAction:
        if self.action == InformationActionType.TOOL:
            if self.tool is None or self.arguments is None:
                raise ValueError("tool actions require tool and arguments")
            if self.answer is not None:
                raise ValueError("tool actions must not include an answer")
        else:
            if not self.answer or not self.answer.strip():
                raise ValueError("answer actions require a non-empty answer")
            if self.tool is not None or self.arguments is not None:
                raise ValueError("answer actions must not include tool fields")
        return self


def _candidate_view(candidate: Any) -> dict[str, Any]:
    """Keep useful grounding fields while bounding vector-hit payloads."""
    if not isinstance(candidate, dict):
        return {}
    metadata = candidate.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    keys = (
        "id",
        "name",
        "text",
        "description",
        "database_name",
        "schema_name",
        "table_id",
        "table_name",
        "source_column",
        "term_name",
        "source",
    )
    result: dict[str, Any] = {}
    for key in keys:
        value = candidate.get(key)
        if value in (None, ""):
            value = metadata.get(key)
        if value not in (None, ""):
            result[key] = value
    return result


def _initial_context(state: AgentState) -> dict[str, Any]:
    path_state = state.get("path_state", {})

    def candidates(key: str) -> list[dict[str, Any]]:
        raw = path_state.get(key)
        if not isinstance(raw, list):
            return []
        return [
            view
            for item in raw[:MAX_CANDIDATES_PER_KIND]
            if (view := _candidate_view(item))
        ]

    return {
        "question": get_question_for_processing(state),
        "entities": path_state.get("entities") or [],
        "subject": path_state.get("subject"),
        "dataset": path_state.get("retrieval_database") or path_state.get("target_db"),
        "extracted_evidence": path_state.get("extracted_evidence") or "",
        "column_attribute_candidates": candidates("retrieved_column_attributes"),
        "sql_attribute_candidates": candidates("retrieved_sql_attributes"),
        "custom_analysis_candidates": candidates("retrieved_custom_analyses"),
        "subject_candidate": _candidate_view(path_state.get("retrieved_subject_term")),
    }


def _render_prompt(
    context: dict[str, Any],
    observations: list[dict[str, Any]],
    *,
    tool_calls_remaining: int,
) -> str:
    observation_text = json.dumps(observations, default=str, ensure_ascii=False)
    if len(observation_text) > MAX_OBSERVATION_CHARS:
        observation_text = observation_text[-MAX_OBSERVATION_CHARS:]
        observation_text = "[earlier observations omitted] " + observation_text

    return f"""You answer questions about data catalog metadata. You do not query source
data and you do not write SQL. Use only the supplied candidates and read-only tool
observations. Never invent tables, columns, formulas, descriptions, or relationships.

Available tools:
- get_dataset: dataset identity plus bounded schemas and tables.
- get_table: table details, columns, semantic attributes, and connected tables.
- get_column: column details, calculation/semantic attributes, owning table, and
  connected tables.
- get_table_semantic_fks: semantic foreign-key connections for a table.

Prefer IDs already present in candidates or observations. Use the dataset name to
scope name lookups. If the user asks how a measure is calculated or defined, report
its description/formula or SQL semantic attributes, owning table/column, and connected
tables. If the user asks which columns a table has, list them and its connected tables.
State clearly when metadata is missing or ambiguous.

Return action="tool" when more metadata is needed. Return action="answer" once the
question can be answered. For the final answer, provide concise Markdown and list the
catalog IDs actually supporting it in source_ids. Do not expose this workflow.

Tool calls remaining: {tool_calls_remaining}

Initial context:
{json.dumps(context, default=str, ensure_ascii=False)}

Tool observations:
{observation_text}
"""


def _collect_ids(value: Any) -> set[str]:
    ids: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if (key == "id" or key.endswith("_id")) and item:
                ids.add(str(item))
            ids.update(_collect_ids(item))
    elif isinstance(value, list):
        for item in value:
            ids.update(_collect_ids(item))
    return ids


def _collect_column_ids(value: Any) -> set[str]:
    column_ids: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"column", "source_column", "foreign_key_column"} and isinstance(
                item, dict
            ):
                if item.get("id"):
                    column_ids.add(str(item["id"]))
            if key in {"columns", "connected_columns", "referencing_columns"}:
                column_ids.update(_collect_ids(item))
            column_ids.update(_collect_column_ids(item))
    elif isinstance(value, list):
        for item in value:
            column_ids.update(_collect_column_ids(item))
    return column_ids


def _fallback_answer(context: dict[str, Any], reason: str) -> str:
    candidates = context.get("column_attribute_candidates") or []
    names = [
        str(candidate.get("name") or candidate.get("text") or "").strip()
        for candidate in candidates[:5]
    ]
    names = [name for name in names if name]
    candidate_note = (
        "\n\nRelevant catalog candidates found: " + ", ".join(names) + "."
        if names
        else ""
    )
    return (
        "I couldn't complete the metadata lookup safely. "
        f"{reason} No source data was queried.{candidate_note}"
    )


def _final_response(answer: str, column_ids: list[str]) -> dict[str, Any]:
    return {
        "response": answer,
        "sql_code": "",
        "sql_columns": column_ids,
        "custom_analyses_used": [],
        "sql_response_from_db": None,
    }


class InformationAgent(BaseAgent):
    """Answer an information intent through a bounded structured tool loop."""

    def __init__(self) -> None:
        super().__init__("information_agent")

    def validate_input(self, state: AgentState) -> bool:
        return bool(get_question_for_processing(state))

    def execute(self, state: AgentState) -> Dict[str, Any]:
        path_state = state.get("path_state", {})
        context = _initial_context(state)
        observations: list[dict[str, Any]] = []
        call_records: list[dict[str, Any]] = []
        seen_calls: set[str] = set()
        known_source_ids = _collect_ids(context)
        known_column_ids: set[str] = set()
        answer: str | None = None
        final_source_ids: list[str] = []
        tool_call_count = 0

        for _ in range(MAX_AGENT_ITERATIONS):
            prompt = _render_prompt(
                context,
                observations,
                tool_calls_remaining=MAX_TOOL_CALLS - tool_call_count,
            )
            try:
                action = invoke_with_structured_output(
                    state["llm"],
                    [SystemMessage(content=prompt)],
                    InformationAgentAction,
                )
            except Exception:
                self.logger.exception("Information-agent LLM call failed")
                action = None

            if action is None:
                answer = _fallback_answer(
                    context, "The metadata reasoning step failed."
                )
                break

            if action.action == InformationActionType.ANSWER:
                answer = (action.answer or "").strip()
                final_source_ids = list(dict.fromkeys(action.source_ids))
                break

            assert action.tool is not None
            assert action.arguments is not None
            arguments = action.arguments.model_dump(exclude_none=True)
            signature = json.dumps(
                {"tool": action.tool.value, "arguments": arguments},
                sort_keys=True,
            )
            if signature in seen_calls:
                observations.append(
                    {
                        "tool": action.tool.value,
                        "arguments": arguments,
                        "result": {
                            "ok": False,
                            "error": {
                                "code": "duplicate_call",
                                "message": "This exact tool call was already made.",
                            },
                        },
                    }
                )
                continue
            if tool_call_count >= MAX_TOOL_CALLS:
                answer = _fallback_answer(
                    context, "The metadata lookup reached its tool-call limit."
                )
                break

            seen_calls.add(signature)
            tool_call_count += 1
            try:
                result = run_information_tool(action.tool, arguments)
            except Exception as exc:
                self.logger.exception("Information metadata tool failed")
                result = {
                    "ok": False,
                    "error": {
                        "code": "tool_failure",
                        "message": f"{type(exc).__name__}: {exc}",
                    },
                }
            observations.append(
                {
                    "tool": action.tool.value,
                    "arguments": arguments,
                    "result": result,
                }
            )
            call_records.append(
                {
                    "tool": action.tool.value,
                    "arguments": arguments,
                    "ok": bool(result.get("ok")),
                }
            )
            known_source_ids.update(_collect_ids(result))
            known_column_ids.update(_collect_column_ids(result))

        if answer is None:
            answer = _fallback_answer(
                context, "The metadata lookup reached its reasoning limit."
            )

        sources = [
            source_id for source_id in final_source_ids if source_id in known_source_ids
        ]
        linked_column_ids = [
            source_id for source_id in sources if source_id in known_column_ids
        ]
        final_response = _final_response(answer, linked_column_ids)
        new_path_state = {
            **path_state,
            "information_tool_calls": call_records,
            "information_sources": sources,
            "formatted_response": answer,
            "final_response": final_response,
        }
        return {
            "messages": list(state.get("messages") or []) + [AIMessage(content=answer)],
            "path_state": new_path_state,
        }


__all__ = [
    "InformationActionType",
    "InformationAgent",
    "InformationAgentAction",
    "InformationToolArguments",
    "MAX_AGENT_ITERATIONS",
    "MAX_TOOL_CALLS",
]
