# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract concise question-relevant constraints from prepared semantic context."""

from __future__ import annotations

import re
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import Field

from auto_ontology.retrieval.text_to_sql.base import BaseAgent, record_thought
from auto_ontology.retrieval.text_to_sql.state import (
    AgentState,
    get_question_for_processing,
)
from auto_ontology.utils.llm_invoke import (
    StrictLLMOutputModel,
    safe_invoke_with_structured_output,
)

_GRAPH_NODE_NAME = "extract_semantic_context"
_MAX_CONSTRAINTS = 8

_SYSTEM_PROMPT = """\
Select the smallest set of constraints that materially helps construct SQL for
the supplied question. Every source is a retrieved ColumnAttribute description.
Its free text may contain definitions, value examples, usage instructions,
warnings, or commonsense/domain evidence; extract those meanings from the text
rather than expecting separate structured fields.

Use only the numbered source records. A selected constraint must be directly
supported by every source_id it cites. Keep exact physical identifiers, formulas,
operators, value representations, grain, and relationship roles from the source.

Select only information that changes or resolves a SQL decision:
- a user concept's exact field or stored-value representation;
- a required or prohibited way to use a field;
- a definition, formula, result grain, or aggregation rule;
- a relationship endpoint, join role, or other schema behavior.

Do not restate obvious names or types. Do not copy general description prose.
Values mentioned as examples in a description are possible stored
representations, not required filters; select one only when the question or
authoritative evidence asks for that value.
Authoritative evidence has higher priority than these sources. Never contradict,
weaken, broaden, or infer a requirement beyond it. If nothing is both relevant
and useful, return an empty constraints list."""


class ExtractedSemanticConstraint(StrictLLMOutputModel):
    """One minimal instruction selected from known semantic sources."""

    kind: Literal[
        "definition",
        "field_mapping",
        "value_representation",
        "required_usage",
        "prohibited_usage",
        "formula_or_grain",
        "relationship_role",
    ]
    instruction: str = Field(
        min_length=1,
        description="Concise SQL-relevant instruction supported by the cited sources.",
    )
    source_ids: list[str] = Field(
        min_length=1,
        max_length=4,
        description="IDs of the numbered source records that directly support this.",
    )


class SemanticContextExtractionResult(StrictLLMOutputModel):
    """Structured result from semantic-context selection."""

    reasoning: str = Field(default="")
    constraints: list[ExtractedSemanticConstraint] = Field(
        default_factory=list,
        max_length=_MAX_CONSTRAINTS,
    )


def _text(value: Any, *, limit: int | None = None) -> str:
    """Normalize prompt text while keeping the source's substantive wording."""
    if value is None:
        return ""
    rendered = re.sub(r"\s+", " ", str(value)).strip()
    return rendered[:limit] if limit is not None else rendered


def _source_id(prefix: str, raw_id: Any, fallback: str) -> str:
    value = _text(raw_id, limit=200) or fallback
    return f"{prefix}:{value}"


def build_semantic_sources(state: AgentState) -> dict[str, dict[str, str]]:
    """Build sources only from retrieved ColumnAttribute embedding text."""
    path_state = state.get("path_state", {})
    sources: dict[str, dict[str, str]] = {}
    for index, attribute in enumerate(
        path_state.get("retrieved_column_attributes") or []
    ):
        if not isinstance(attribute, dict):
            continue
        description = _text(attribute.get("text"))
        if not description:
            continue
        source_id = _source_id("column_attribute", attribute.get("id"), str(index))
        if source_id in sources:
            continue
        name = _text(attribute.get("name")) or "unnamed"
        term = _text(attribute.get("term_name"))
        table = _text(attribute.get("table_name"))
        column = _text(attribute.get("source_column"))
        location = ".".join(part for part in (table, column) if part)
        label_parts = [name]
        if term:
            label_parts.append(f"term {term}")
        if location:
            label_parts.append(location)
        sources[source_id] = {
            "label": " | ".join(label_parts),
            "content": description,
        }
    return sources


def _render_sources(sources: dict[str, dict[str, str]]) -> str:
    return "\n".join(
        f"{source_id} | {source['label']} | {source['content']}"
        for source_id, source in sources.items()
    )


def validate_constraints(
    result: SemanticContextExtractionResult,
    sources: dict[str, dict[str, str]],
) -> list[ExtractedSemanticConstraint]:
    """Keep only source-backed, non-duplicated constraints."""
    kept: list[ExtractedSemanticConstraint] = []
    seen: set[str] = set()
    for constraint in result.constraints:
        if any(source_id not in sources for source_id in constraint.source_ids):
            continue
        normalized = re.sub(r"\W+", " ", constraint.instruction).casefold().strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        kept.append(constraint)
    return kept


def format_semantic_constraints(
    constraints: list[ExtractedSemanticConstraint],
    sources: dict[str, dict[str, str]],
) -> str:
    """Render validated constraints with exact source labels."""
    if not constraints:
        return ""
    lines = [
        "## Important Semantic Context",
        "Use these source-backed constraints when they apply to the question. "
        "Authoritative Evidence overrides this context. Do not infer requirements "
        "that are not stated here.",
    ]
    for constraint in constraints:
        labels = "; ".join(
            sources[source_id]["label"] for source_id in constraint.source_ids
        )
        lines.append(
            f"- [{constraint.kind}] {constraint.instruction} (source: {labels})"
        )
    return "\n".join(lines)


class SemanticContextExtractionAgent(BaseAgent):
    """Select minimal semantic/catalog constraints for SQL generation."""

    def __init__(self) -> None:
        super().__init__("semantic_context_extraction")

    def execute(self, state: AgentState) -> dict[str, Any]:
        path_state = state.get("path_state", {})
        sources = build_semantic_sources(state)
        empty_update = {
            "semantic_constraints": [],
            "semantic_constraints_context": "",
        }
        if not sources:
            return {"path_state": {**path_state, **empty_update}}

        llm = state.get("reasoning_llm") or state["llm"]
        prompt = (
            f"Question:\n{get_question_for_processing(state)}\n\n"
            f"Authoritative Evidence:\n{state.get('evidence') or '(none)'}\n\n"
            f"Numbered source records:\n{_render_sources(sources)}"
        )
        try:
            result = safe_invoke_with_structured_output(
                llm,
                [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ],
                SemanticContextExtractionResult,
            )
        except Exception:
            self.logger.exception(
                "Semantic context extraction failed; continuing empty"
            )
            return {"path_state": {**path_state, **empty_update}}

        if result is None:
            self.logger.warning("Semantic context extraction returned no result")
            return {"path_state": {**path_state, **empty_update}}

        constraints = validate_constraints(result, sources)
        context = format_semantic_constraints(constraints, sources)
        reasoning = (result.reasoning or "").strip()
        if reasoning:
            record_thought(path_state, _GRAPH_NODE_NAME, reasoning)
        return {
            "path_state": {
                **path_state,
                "semantic_constraints": [
                    constraint.model_dump() for constraint in constraints
                ],
                "semantic_constraints_context": context,
            }
        }


__all__ = [
    "ExtractedSemanticConstraint",
    "SemanticContextExtractionAgent",
    "SemanticContextExtractionResult",
    "build_semantic_sources",
    "format_semantic_constraints",
    "validate_constraints",
]
