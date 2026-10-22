# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool-using agent that prepares grounded semantic context for SQL generation."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any, Dict

from langchain_core.messages import SystemMessage
from pydantic import Field, model_validator

from auto_ontology.catalog.constants import Labels
from auto_ontology.retrieval.entity_coverage.agents.question_extraction import (
    _filter_glossary_by_used_names,
)
from auto_ontology.retrieval.text_to_sql.agents.candidates_preparation import (
    CandidatePreparationAgent,
)
from auto_ontology.retrieval.text_to_sql.agents.semantic_preparation_tools import (
    SemanticPreparationToolName,
    run_semantic_preparation_tool,
)
from auto_ontology.retrieval.text_to_sql.base import BaseAgent, record_thought
from auto_ontology.retrieval.text_to_sql.state import (
    AgentState,
    get_original_question,
    get_question_for_processing,
)
from auto_ontology.semantic.constants import (
    LABEL_COLUMN_ATTRIBUTE,
    LABEL_SQL_ATTRIBUTE,
)
from auto_ontology.utils.llm_invoke import (
    StrictLLMOutputModel,
    invoke_with_structured_output,
)

MAX_TOOL_CALLS = 14
MAX_AGENT_ITERATIONS = 16
MAX_OBSERVATION_CHARS = 30_000
_GRAPH_NODE_NAME = "prepare_semantic_context"


class SemanticPreparationActionType(StrEnum):
    """Possible structured actions in the preparation loop."""

    TOOL = "tool"
    COMPLETE = "complete"


class SemanticPreparationToolArguments(StrictLLMOutputModel):
    """Bounded union of arguments accepted by preparation tools."""

    query: str | None = None
    labels: list[str] | None = None
    database_name: str | None = None
    limit: int | None = Field(default=None, ge=1, le=10)
    table_id: str | None = None
    table_name: str | None = None
    column_attribute_id: str | None = None
    column_attribute_name: str | None = None
    term_id: str | None = None
    term_name: str | None = None
    value: str | int | float | bool | None = None
    attribute_ids: list[str] | None = Field(default=None, max_length=12)
    anchor_attribute_id: str | None = None
    destination_attribute_id: str | None = None
    table_ids: list[str] | None = Field(default=None, max_length=20)
    anchor_table_id: str | None = None


class SemanticPreparationResult(StrictLLMOutputModel):
    """The model's selected semantic context, expressed only as observed IDs."""

    normalized_question: str
    entities: list[str] = Field(default_factory=list)
    subject: str | None = None
    target_db: str | None = None
    used_glossary_names: list[str] = Field(default_factory=list)
    column_attribute_ids: list[str] = Field(default_factory=list)
    sql_attribute_ids: list[str] = Field(default_factory=list)
    custom_analysis_ids: list[str] = Field(default_factory=list)
    subject_term_id: str | None = None
    table_ids: list[str] = Field(default_factory=list)
    primary_attribute_id: str | None = None
    reasoning: str = ""


class SemanticPreparationAction(StrictLLMOutputModel):
    """One tool call or terminal semantic-context selection."""

    action: SemanticPreparationActionType
    tool: SemanticPreparationToolName | None = None
    arguments: SemanticPreparationToolArguments | None = None
    result: SemanticPreparationResult | None = None

    @model_validator(mode="after")
    def _validate_action(self) -> SemanticPreparationAction:
        if self.action == SemanticPreparationActionType.TOOL:
            if self.tool is None or self.arguments is None:
                raise ValueError("tool actions require tool and arguments")
            if self.result is not None:
                raise ValueError("tool actions must not contain result")
        else:
            if self.result is None:
                raise ValueError("complete actions require result")
            if self.tool is not None or self.arguments is not None:
                raise ValueError("complete actions must not contain tool fields")
        return self


def _render_prompt(
    state: AgentState,
    observations: list[dict[str, Any]],
    *,
    calls_remaining: int,
) -> str:
    path_state = state.get("path_state", {})
    connectors = [
        {
            "database_name": getattr(connector, "database_name", None),
            "dialect": getattr(connector, "dialect", None),
        }
        for connector in state.get("connectors") or []
    ]
    context = {
        "original_question": get_original_question(state),
        "question_for_processing": get_question_for_processing(state),
        "evidence": state.get("evidence") or "",
        "target_database_hint": path_state.get("target_db"),
        "connectors": connectors,
        "glossary": state.get("glossary") or [],
        "question_type": path_state.get("question_type"),
    }
    observation_text = json.dumps(observations, default=str, ensure_ascii=False)
    if len(observation_text) > MAX_OBSERVATION_CHARS:
        observation_text = (
            "[earlier observations omitted] "
            + observation_text[-MAX_OBSERVATION_CHARS:]
        )
    return f"""You prepare semantic context for a downstream text-to-SQL model.
You never write SQL. Work like a careful database analyst: identify every requested
output, grouping, filter, relationship, and literal value needed to answer the
question, then use read-only tools to ground them in one database.

Available tools:
- search_semantic_layer: vector-search Terms, ColumnAttributes, SqlAttributes, and
  CustomAnalyses. Search separately for distinct concepts and requested outputs.
- search_tables: vector-search table descriptions when semantic attributes do not
  cover a needed table.
- get_column_attribute: inspect an exact semantic attribute and linked columns.
- get_table: inspect a table, columns, semantic attributes, and connections.
- probe_value: given one literal value and ORDERED suspected ColumnAttribute IDs,
  checks case-insensitive equality across all suspects first, then text containment.
- find_join_path: inspect the verified shortest path between two attributes.
- discover_connections: inspect junction, bridge, and shared-parent structures for
  selected tables.

Rules:
1. Extract literal values from both the question and evidence, including text,
   numbers, and dates. Do not treat requested column/concept names as values.
2. For each real literal, search semantic metadata for likely columns, then call
   probe_value. A matched attribute is important: include it in column_attribute_ids
   and consider it for the anchor and join paths. Never invent a value match.
3. Preserve the question's meaning and literal text in normalized_question. Remove
   instructions or noise, but do not replace one value with a different evidence value.
4. Select the smallest complete set of observed semantic IDs and tables. Include bridge
   or junction tables only when tool observations show they are needed.
5. Prefer formulas/CustomAnalyses only when they match the requested calculation.
6. Before complete, inspect join paths from the chosen primary attribute to the other
   required attributes. The primary attribute should be a generic, stable field that
   connects the required context; a verified value attribute may be primary when it
   best anchors the question.
7. Every returned ID must have appeared in a successful tool observation. If metadata
   is unavailable, return the best grounded subset; never fabricate identifiers.

Return action=\"tool\" while more grounding is needed. Return action=\"complete\" with
the selected IDs and concise reasoning when the context is sufficient.

Tool calls remaining: {calls_remaining}

Request context:
{json.dumps(context, default=str, ensure_ascii=False)}

Tool observations:
{observation_text}
"""


def _walk_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk_dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_dicts(item)


def _records_by_id(observations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for observation in observations:
        result = observation.get("result") or {}
        if not result.get("ok"):
            continue
        for record in _walk_dicts(result.get("data")):
            identifier = record.get("id")
            if identifier:
                records.setdefault(str(identifier), record)
    return records


def _hits_for_ids(
    ids: list[str],
    records: dict[str, dict[str, Any]],
    *,
    label: str,
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for identifier in dict.fromkeys(ids):
        record = records.get(identifier)
        if record is None:
            continue
        hit = dict(record)
        hit["id"] = identifier
        hit.setdefault("label", label)
        hit.setdefault(
            "text",
            hit.get("description") or hit.get("name") or identifier,
        )
        hits.append(hit)
    return hits


def _verified_value_anchors(
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for observation in observations:
        if observation.get("tool") != SemanticPreparationToolName.PROBE_VALUE.value:
            continue
        result = observation.get("result") or {}
        data = result.get("data") if result.get("ok") else None
        if not isinstance(data, dict) or not data.get("column_attribute_id"):
            continue
        phrase = str(data.get("input_value") or "")
        table = str(data.get("table_name") or "")
        column = str(data.get("column_name") or "")
        key = (phrase.casefold(), table.casefold(), column.casefold())
        if key in seen:
            continue
        seen.add(key)
        anchors.append(
            {
                "phrase": phrase,
                "kind": "value",
                "tbl": table,
                "col": column,
                "stored_value": str(data.get("stored_value") or ""),
                "n_rows": "1+",
                "column_attribute_id": str(data["column_attribute_id"]),
                "match_type": str(data.get("match_type") or ""),
            }
        )
    return anchors


def _fallback_result(
    state: AgentState,
    observations: list[dict[str, Any]],
) -> SemanticPreparationResult:
    records = _records_by_id(observations)
    attribute_ids = [
        identifier
        for identifier, record in records.items()
        if record.get("label") == LABEL_COLUMN_ATTRIBUTE or "column_attribute" in record
    ]
    table_ids = [
        identifier
        for identifier, record in records.items()
        if record.get("label") == Labels.TABLE or "columns" in record
    ]
    question = get_question_for_processing(state)
    return SemanticPreparationResult(
        normalized_question=question,
        entities=[question] if question else [],
        target_db=state.get("path_state", {}).get("target_db"),
        column_attribute_ids=attribute_ids[:5],
        table_ids=table_ids[:10],
        primary_attribute_id=attribute_ids[0] if attribute_ids else None,
        reasoning="Used the best validated context available after preparation stopped.",
    )


class SemanticPreparationAgent(BaseAgent):
    """Prepare all semantic candidates through one bounded tool-using agent."""

    def __init__(self) -> None:
        super().__init__("semantic_preparation")

    def validate_input(self, state: AgentState) -> bool:
        return bool(get_question_for_processing(state))

    def execute(self, state: AgentState) -> Dict[str, Any]:
        path_state = state.get("path_state", {})
        observations: list[dict[str, Any]] = []
        call_records: list[dict[str, Any]] = []
        seen_calls: set[str] = set()
        tool_calls = 0
        completion: SemanticPreparationResult | None = None
        llm = state.get("reasoning_llm") or state["llm"]

        for _ in range(MAX_AGENT_ITERATIONS):
            prompt = _render_prompt(
                state,
                observations,
                calls_remaining=MAX_TOOL_CALLS - tool_calls,
            )
            try:
                action = invoke_with_structured_output(
                    llm,
                    [SystemMessage(content=prompt)],
                    SemanticPreparationAction,
                )
            except Exception:
                self.logger.exception("Semantic-preparation reasoning failed")
                action = None
            if action is None:
                break
            if action.action == SemanticPreparationActionType.COMPLETE:
                completion = action.result
                break

            assert action.tool is not None
            assert action.arguments is not None
            arguments = action.arguments.model_dump(exclude_none=True)
            signature = json.dumps(
                {"tool": action.tool.value, "arguments": arguments},
                sort_keys=True,
                default=str,
            )
            if signature in seen_calls:
                observations.append(
                    {
                        "tool": action.tool.value,
                        "arguments": arguments,
                        "result": _error_result(
                            "duplicate_call",
                            "This exact tool call was already made.",
                        ),
                    }
                )
                continue
            if tool_calls >= MAX_TOOL_CALLS:
                break
            seen_calls.add(signature)
            tool_calls += 1
            try:
                result = run_semantic_preparation_tool(
                    action.tool,
                    arguments,
                    semantic_retriever=state.get("semantic_retriever"),
                    data_retriever=state.get("data_retriever"),
                    connectors=state.get("connectors") or [],
                )
            except Exception as exc:
                self.logger.exception("Semantic-preparation tool failed")
                result = _error_result(
                    "tool_failure",
                    f"{type(exc).__name__}: {exc}",
                )
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

        if not _records_by_id(observations):
            for tool, arguments in (
                (
                    SemanticPreparationToolName.SEARCH_SEMANTIC_LAYER,
                    {
                        "query": get_question_for_processing(state),
                        "database_name": path_state.get("target_db"),
                        "limit": 5,
                    },
                ),
                (
                    SemanticPreparationToolName.SEARCH_TABLES,
                    {
                        "query": get_question_for_processing(state),
                        "database_name": path_state.get("target_db"),
                        "limit": 5,
                    },
                ),
            ):
                result = run_semantic_preparation_tool(
                    tool,
                    arguments,
                    semantic_retriever=state.get("semantic_retriever"),
                    data_retriever=state.get("data_retriever"),
                    connectors=state.get("connectors") or [],
                )
                observations.append(
                    {
                        "tool": tool.value,
                        "arguments": arguments,
                        "result": result,
                    }
                )
                call_records.append(
                    {
                        "tool": tool.value,
                        "arguments": arguments,
                        "ok": bool(result.get("ok")),
                    }
                )
        if completion is None:
            completion = _fallback_result(state, observations)

        records = _records_by_id(observations)
        known_ids = set(records)

        def verified(ids: list[str]) -> list[str]:
            return [
                identifier
                for identifier in dict.fromkeys(ids)
                if identifier in known_ids
            ]

        column_attribute_ids = verified(completion.column_attribute_ids)
        anchors = _verified_value_anchors(observations)
        for anchor in anchors:
            attribute_id = anchor["column_attribute_id"]
            if attribute_id in known_ids and attribute_id not in column_attribute_ids:
                column_attribute_ids.append(attribute_id)
        sql_attribute_ids = verified(completion.sql_attribute_ids)
        custom_analysis_ids = verified(completion.custom_analysis_ids)
        table_ids = verified(completion.table_ids)
        primary_attribute_id = completion.primary_attribute_id
        if primary_attribute_id not in column_attribute_ids:
            primary_attribute_id = (
                column_attribute_ids[0] if column_attribute_ids else None
            )

        known_database_names = {
            str(getattr(connector, "database_name", "")).casefold(): str(
                getattr(connector, "database_name", "")
            )
            for connector in state.get("connectors") or []
            if getattr(connector, "database_name", None)
        }
        hinted_database = str(path_state.get("target_db") or "")
        target_db = known_database_names.get(str(completion.target_db or "").casefold())
        if target_db is None:
            target_db = hinted_database or (
                next(iter(known_database_names.values()))
                if len(known_database_names) == 1
                else None
            )

        retrieved_column_attributes = _hits_for_ids(
            column_attribute_ids,
            records,
            label=LABEL_COLUMN_ATTRIBUTE,
        )
        retrieved_sql_attributes = _hits_for_ids(
            sql_attribute_ids,
            records,
            label=LABEL_SQL_ATTRIBUTE,
        )
        retrieved_custom_analyses = _hits_for_ids(
            custom_analysis_ids,
            records,
            label=Labels.CUSTOM_ANALYSIS,
        )
        selected_path_state = {
            **path_state,
            "normalized_question": (
                completion.normalized_question.strip()
                or get_question_for_processing(state)
            ),
            "entities": completion.entities
            or [completion.normalized_question or get_question_for_processing(state)],
            "subject": completion.subject,
            "target_db": target_db,
            "retrieved_column_attributes": retrieved_column_attributes,
            "retrieved_sql_attributes": retrieved_sql_attributes,
            "retrieved_custom_analyses": retrieved_custom_analyses,
            "candidates": (
                retrieved_column_attributes
                + retrieved_sql_attributes
                + retrieved_custom_analyses
            ),
            "similar_questions": path_state.get("similar_questions") or [],
            "semantic_preparation_primary_attribute_id": primary_attribute_id,
            "semantic_preparation_table_ids": table_ids,
            "semantic_preparation_tool_calls": call_records,
            "semantic_preparation_reasoning": completion.reasoning,
        }
        if completion.subject_term_id in known_ids:
            selected_path_state["retrieved_subject_term"] = records[
                completion.subject_term_id
            ]

        selected_state: AgentState = {
            **state,
            "path_state": selected_path_state,
            "value_anchors": anchors,
            "glossary": _filter_glossary_by_used_names(
                state.get("glossary") or [],
                completion.used_glossary_names,
            ),
        }
        prepared = CandidatePreparationAgent().execute(selected_state)
        prepared_path_state = prepared["path_state"]
        prepared_path_state["semantic_preparation_tool_calls"] = call_records
        prepared_path_state["semantic_preparation_reasoning"] = completion.reasoning
        if completion.reasoning:
            record_thought(
                prepared_path_state,
                _GRAPH_NODE_NAME,
                completion.reasoning,
            )

        return {
            "path_state": prepared_path_state,
            "value_anchors": prepared.get("value_anchors", anchors),
            "glossary": selected_state["glossary"],
        }


def _error_result(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message}}


__all__ = [
    "MAX_AGENT_ITERATIONS",
    "MAX_TOOL_CALLS",
    "SemanticPreparationAction",
    "SemanticPreparationActionType",
    "SemanticPreparationAgent",
    "SemanticPreparationResult",
    "SemanticPreparationToolArguments",
]
