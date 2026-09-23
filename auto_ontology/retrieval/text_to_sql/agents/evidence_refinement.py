# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conservatively ground evidence literals against the prepared schema."""

from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Any, Dict, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import Field

from auto_ontology.retrieval.text_to_sql.base import BaseAgent, record_thought
from auto_ontology.retrieval.text_to_sql.formatters_util import format_tables_for_prompt
from auto_ontology.retrieval.text_to_sql.state import (
    AgentState,
    get_question_for_processing,
)
from auto_ontology.utils.llm_invoke import (
    StrictLLMOutputModel,
    safe_invoke_structured_nr,
)
from auto_ontology.utils.sample_values import stringify_sample_values

_GRAPH_NODE_NAME = "refine_evidence"
_COMPARISON_OPERATORS = frozenset({"=", "!=", "<>", ">", ">=", "<", "<="})
_FORMULA_OPERATOR_RE = re.compile(r"[*+/]")
_OPERATOR_CUES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\b(?:at least|no less than|starting from|on or after)\b", re.I),
        ">=",
    ),
    (
        re.compile(
            r"\b(?:greater than|larger than|(?<!no )more than|(?<!or )after)\b",
            re.I,
        ),
        ">",
    ),
    (re.compile(r"\b(?:at most|no more than|up to|on or before)\b", re.I), "<="),
    (
        re.compile(r"\b(?:(?<!no )less than|fewer than|(?<!or )before)\b", re.I),
        "<",
    ),
    (re.compile(r"\b(?:exactly|equal to)\b", re.I), "="),
)

_SYSTEM_PROMPT = """\
You refine evidence before SQL generation by correcting only mistakes that are
certain from the user's question or the supplied column sample values.

Return complete corrected lines, not substring patches. For every repair:
- copy original_line exactly from the numbered evidence, without its ``N| `` prefix;
- return corrected_line as the complete fixed version of that same line;
- preserve every other word, formula, mapping, and instruction byte-for-byte;
- set corrected_value to the exact final scalar value or comparison operator used
  in corrected_line (quotes around a string are optional in corrected_value).

Allowed repairs:
1. string_representation: fix a string value or its casing only when one sample
   value for the named table and column proves the exact stored representation;
2. constant_value: replace a mistaken scalar constant only when the sanitized
   question explicitly supplies the intended constant, or a column sample proves it;
3. predicate_operator: change only =, !=, <>, >, >=, <, or <= when the sanitized
   question makes the evidence operator unquestionably wrong. For example, "at
   least", "starting from", and "on or after" are inclusive; "more than" is strict.

Never change formulas, arithmetic, table names, column names, mappings, prose,
logical connectors, ordering, grouping, or projection instructions. Never add or
remove evidence. If a repair is uncertain, return no patch for it. An empty repair
list is the correct answer whenever the evidence may already be valid."""


class EvidenceLineRepair(StrictLLMOutputModel):
    """One complete corrected evidence line."""

    line_number: int = Field(ge=1, description="One-based evidence line number.")
    original_line: str = Field(
        description="The complete original evidence line, copied exactly."
    )
    corrected_line: str = Field(
        description="The complete corrected line with no unrelated rewrites."
    )
    kind: Literal["string_representation", "constant_value", "predicate_operator"]
    corrected_value: str = Field(
        description=(
            "Exact final scalar value or comparison operator used in corrected_line."
        )
    )
    table_name: str = Field(
        default="",
        description="Table containing the value; required for sample-backed repairs.",
    )
    column_name: str = Field(
        default="",
        description="Column containing the value; required for sample-backed repairs.",
    )
    reason: str = Field(description="Why the correction is certain.")


class EvidenceRefinementResult(StrictLLMOutputModel):
    """Structured evidence refinement response."""

    reasoning: str = Field(default="")
    repairs: list[EvidenceLineRepair] = Field(default_factory=list)


def _unquote_literal(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _question_contains_literal(question: str, literal: str) -> bool:
    value = _unquote_literal(literal)
    if not value:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(value)}(?!\w)", question))


def _unambiguous_question_operator(question: str) -> str | None:
    operators = {
        operator for pattern, operator in _OPERATOR_CUES if pattern.search(question)
    }
    return operators.pop() if len(operators) == 1 else None


def _column_samples(
    tables: list[dict[str, Any]],
    table_name: str,
    column_name: str,
) -> list[str] | None:
    """Return samples only when the table/column reference resolves uniquely."""
    wanted_table = table_name.casefold().strip()
    wanted_column = column_name.casefold().strip()
    matches: list[list[str]] = []

    for table in tables:
        actual_table = str(table.get("name") or "")
        qualified_table = ".".join(
            str(part)
            for part in (
                table.get("database_name"),
                table.get("schema_name"),
                actual_table,
            )
            if part
        )
        if wanted_table not in {
            actual_table.casefold(),
            qualified_table.casefold(),
        }:
            continue
        for column in table.get("columns") or []:
            if not isinstance(column, dict):
                continue
            if str(column.get("name") or "").casefold() != wanted_column:
                continue
            matches.append(stringify_sample_values(column.get("sample_values")))

    return matches[0] if len(matches) == 1 else None


def _sample_supports(
    repair: EvidenceLineRepair,
    tables: list[dict[str, Any]],
) -> bool:
    if not repair.table_name or not repair.column_name:
        return False
    samples = _column_samples(tables, repair.table_name, repair.column_name)
    if not samples:
        return False
    return _unquote_literal(repair.corrected_value) in samples


def _corrected_line_contains_value(repair: EvidenceLineRepair) -> bool:
    value = _unquote_literal(repair.corrected_value)
    return bool(value) and value in repair.corrected_line


def _is_local_line_correction(repair: EvidenceLineRepair) -> bool:
    """Reject broad prose rewrites while allowing one compact value correction."""
    changed = [
        opcode
        for opcode in SequenceMatcher(
            None, repair.original_line, repair.corrected_line, autojunk=False
        ).get_opcodes()
        if opcode[0] != "equal"
    ]
    if not changed or len(changed) > 3:
        return False
    changed_chars = sum(
        (old_end - old_start) + (new_end - new_start)
        for _, old_start, old_end, new_start, new_end in changed
    )
    value_length = len(_unquote_literal(repair.corrected_value))
    return changed_chars <= max(32, value_length * 3 + 8)


def _valid_repair(
    repair: EvidenceLineRepair,
    line: str,
    question: str,
    tables: list[dict[str, Any]],
) -> bool:
    if (
        repair.original_line != line
        or repair.corrected_line == line
        or "\n" in repair.original_line
        or "\n" in repair.corrected_line
        or not repair.corrected_value.strip()
        or _FORMULA_OPERATOR_RE.search(line)
        or not _is_local_line_correction(repair)
    ):
        return False

    if repair.kind == "predicate_operator":
        return (
            repair.corrected_value.strip() in _COMPARISON_OPERATORS
            and repair.corrected_value.strip()
            == _unambiguous_question_operator(question)
            and repair.corrected_value.strip() in repair.corrected_line
        )

    if not _corrected_line_contains_value(repair):
        return False
    if repair.kind == "string_representation":
        return _sample_supports(repair, tables)
    return _question_contains_literal(
        question, repair.corrected_value
    ) or _sample_supports(repair, tables)


def apply_evidence_repairs(
    evidence: str,
    repairs: list[EvidenceLineRepair],
    question: str,
    tables: list[dict[str, Any]],
) -> tuple[str, list[dict[str, str | int]]]:
    """Atomically apply validated full-line repairs.

    The LLM supplies complete corrected lines, while this function owns assembly
    of the final evidence so every unmentioned line remains byte-for-byte intact.
    """
    lines = evidence.splitlines(keepends=True)
    accepted: list[dict[str, str | int]] = []
    seen_line_numbers: set[int] = set()

    for repair in repairs:
        index = repair.line_number - 1
        if index < 0 or index >= len(lines) or repair.line_number in seen_line_numbers:
            return evidence, []
        seen_line_numbers.add(repair.line_number)

        original_with_ending = lines[index]
        line_ending = original_with_ending[len(original_with_ending.rstrip("\r\n")) :]
        original_line = original_with_ending.removesuffix(line_ending)
        if not _valid_repair(repair, original_line, question, tables):
            return evidence, []
        lines[index] = repair.corrected_line + line_ending
        accepted.append(repair.model_dump())

    return "".join(lines), accepted


class EvidenceRefinementAgent(BaseAgent):
    """Correct certain evidence literals while leaving everything else verbatim."""

    def __init__(self) -> None:
        super().__init__("evidence_refinement")

    def validate_input(self, state: AgentState) -> bool:
        return bool((state.get("evidence") or "").strip())

    def execute(self, state: AgentState) -> Dict[str, Any]:
        evidence = state.get("evidence") or ""
        if not evidence.strip():
            return {}

        path_state = dict(state.get("path_state") or {})
        question = get_question_for_processing(state)
        tables = list(path_state.get("relevant_tables") or [])
        numbered_evidence = "\n".join(
            f"{index}| {line}" for index, line in enumerate(evidence.splitlines(), 1)
        )
        tables_block = format_tables_for_prompt(
            tables,
            target_db=path_state.get("target_db"),
        )
        prompt = (
            f"Sanitized question:\n{question}\n\n"
            f"Evidence (line numbers are metadata only):\n{numbered_evidence}\n\n"
            f"Relevant tables, columns, and sample values:\n{tables_block}"
        )

        result = safe_invoke_structured_nr(
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)],
            EvidenceRefinementResult,
        )
        if result is None:
            self.logger.warning(
                "Evidence refinement failed — keeping original evidence"
            )
            return {"evidence": evidence, "path_state": path_state}

        refined, accepted = apply_evidence_repairs(
            evidence, result.repairs, question, tables
        )
        if result.reasoning.strip():
            record_thought(path_state, _GRAPH_NODE_NAME, result.reasoning.strip())
        if accepted:
            path_state["evidence_original"] = evidence
            path_state["evidence_repairs_applied"] = accepted
            self.logger.info("Applied %d evidence repair(s)", len(accepted))
        else:
            self.logger.info("No certain evidence repairs found")

        return {"evidence": refined, "path_state": path_state}


__all__ = [
    "EvidenceRefinementAgent",
    "EvidenceRefinementResult",
    "EvidenceLineRepair",
    "apply_evidence_repairs",
]
