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
from auto_ontology.retrieval.text_to_sql.connector_routing import (
    resolve_connector_from_tables,
)
from auto_ontology.retrieval.text_to_sql.formatters_util import (
    format_tables_for_prompt,
)
from auto_ontology.retrieval.text_to_sql.state import (
    AgentState,
    get_question_for_processing,
)
from auto_ontology.utils.llm_invoke import (
    StrictLLMOutputModel,
    safe_invoke_structured_nr,
)
from auto_ontology.utils.sample_values import stringify_sample_values
from auto_ontology.utils.sql_identifiers import qualified_name, quoted_identifier

_GRAPH_NODE_NAME = "refine_evidence"
_COMPARISON_OPERATORS = frozenset({"=", "!=", "<>", ">", ">=", "<", "<="})
_COMPARISON_RE = re.compile(r"(?<![<>=!])(?:>=|<=|!=|<>|=|>|<)(?![<>=])")
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
certain from the user's question, verified database value anchors, or supplied
column sample values. Prefer an exact value anchor for the named table and column;
only fall back to that column's sample values when no matching anchor proves the
stored representation.

Return complete corrected lines, not substring patches. For every repair:
- copy original_line exactly from the numbered evidence, without its ``N| `` prefix;
- return corrected_line as the complete fixed version of that same line;
- preserve every other word, formula, mapping, and instruction byte-for-byte;
- set corrected_value to the exact final scalar value or comparison operator used
  in corrected_line (quotes around a string are optional in corrected_value).

Also return corrected_question as the complete supplied sanitized question with
the same proven scalar/operator corrections applied wherever those values occur in
its grounded rules. If no repair applies to the question, copy it unchanged.

Allowed repairs:
1. string_representation: fix a string value or its casing only when one value
   anchor, or as a fallback one sample value, for the named table and column proves
   the exact stored representation;
2. constant_value: replace a mistaken scalar constant only when the sanitized
   question explicitly supplies the intended constant, a value anchor proves it,
   or as a fallback a column sample proves it;
3. predicate_operator: change only =, !=, <>, >, >=, <, or <= when the sanitized
   question makes the evidence operator unquestionably wrong. For example, "at
   least", "starting from", and "on or after" are inclusive; "more than" is strict.

Never change formulas, arithmetic, schema names, table names, column names, mappings, prose,
logical connectors, ordering, grouping, or projection instructions. Never add or
remove evidence. If a repair is uncertain, return no patch for it. An empty repair
list is the correct answer whenever the evidence may already be valid."""

_GROUNDING_SYSTEM_PROMPT = """\
Ground field references before value correction. Use only the supplied relevant
tables section and exact qualified-column list.

Return the complete grounded_question and grounded_evidence, ready for SQL
generation. Replace field references with the exact dialect-quoted physical
references supplied in the prompt. Copy those references verbatim.

STRICT SCOPE:
- Replace ONLY fields that are described or mapped in the Evidence section.
- Treat the natural-language phrase before evidence cues such as "refers to",
  "means", "is defined as", or "corresponds to" as the concept label for the
  physical field on the right-hand side.
- In grounded_question, search case-insensitively for that concept label or its
  semantic equivalent. If present, you MUST replace the minimal matching question
  phrase with the complete grounded rule from evidence: the physical column AND
  its operator, value, formula, or condition. Copy those rule inputs unchanged at
  this phase; value correction happens later. Never leave that question concept in
  natural language after grounding its evidence.
  The question phrase does NOT need to match the evidence wording exactly:
  account for capitalization, singular/plural forms, inflections, and ordinary
  paraphrases. Preserve surrounding qualifiers and all unrelated wording. If the
  evidence-defined concept is not semantically present in the question, leave the
  question unchanged. Do not ground any other question field or phrase.
- In grounded_evidence, replace only the evidence's own field references.
- A field appearing in the tables section but not described by evidence MUST remain
  untouched in both outputs.

Preserve every scalar value, operator, formula, arithmetic expression, filter,
grouping, ordering, projection, and unrelated word exactly. Explicit
quoted/backticked column names in evidence are authoritative and must map to that
exact catalog column, never to a semantically similar column. If an
evidence-described field is uncertain, leave it unchanged."""


class EvidenceGroundingResult(StrictLLMOutputModel):
    """Structured schema-grounding response produced before value repair."""

    reasoning: str = Field(default="")
    grounded_question: str = Field(
        description=(
            "Complete question with the minimal semantic phrase corresponding to "
            "each evidence-described field replaced by its complete grounded rule "
            "(physical field plus operator/value/formula), even when wording "
            "differs; every other part must be unchanged."
        )
    )
    grounded_evidence: str = Field(
        description=(
            "Complete evidence with only its field references grounded; values, "
            "operators, formulas, and all unrelated text must be unchanged."
        )
    )


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
    corrected_question: str = Field(
        default="",
        description=(
            "Complete grounded question with the same accepted value/operator "
            "corrections applied. Never change schema, table, or column references."
        ),
    )


def _catalog_columns(
    tables: list[dict[str, Any]],
    target_db: str | None,
    dialect: str | None,
    connector: Any | None = None,
) -> list[dict[str, str]]:
    """Flatten relevant tables into validated physical-column targets."""
    result: list[dict[str, str]] = []
    for table in tables:
        raw_table = str(table.get("name") or "").strip()
        schema = str(table.get("schema_name") or "").strip()
        database = str(table.get("database_name") or target_db or "").strip()
        table_description = str(table.get("description") or "").strip()
        if not raw_table:
            continue
        qualified_table = (
            connector.qualify(schema or None, raw_table)
            if connector is not None
            else qualified_name(database, schema, raw_table, dialect=dialect)
        )
        for column in table.get("columns") or []:
            if not isinstance(column, dict):
                continue
            raw_column = str(column.get("name") or "").strip()
            if not raw_column:
                continue
            column_details = [
                str(column.get(key) or "").strip()
                for key in ("description", "usage_evidence", "constraints", "format")
                if str(column.get(key) or "").strip()
            ]
            result.append(
                {
                    "table": raw_table,
                    "schema": schema,
                    "qualified_table": qualified_table,
                    "column": raw_column,
                    "physical": (
                        f"{qualified_table}.{quoted_identifier(raw_column, dialect)}"
                    ),
                    "table_description": table_description,
                    "column_description": " | ".join(column_details),
                }
            )
    return result


def _explicit_identifier_targets(
    evidence: str,
    catalog_columns: list[dict[str, str]],
) -> dict[str, str]:
    """Resolve quoted identifiers for validation without rewriting evidence."""
    locked: dict[str, str] = {}
    for match in re.finditer(r"`([^`]+)`|\"([^\"]+)\"", evidence):
        identifier = (match.group(1) or match.group(2) or "").strip()
        if not identifier:
            continue
        candidates = [
            entry
            for entry in catalog_columns
            if entry["column"].casefold() == identifier.casefold()
        ]
        if len(candidates) > 1:
            candidates = [
                entry
                for entry in candidates
                if re.search(
                    rf"(?<!\w){re.escape(entry['table'])}(?!\w)",
                    evidence,
                    re.IGNORECASE,
                )
            ]
        if len(candidates) == 1:
            locked[match.group(0)] = candidates[0]["physical"]
    return locked


def validate_schema_grounding(
    question: str,
    evidence: str,
    result: EvidenceGroundingResult,
    tables: list[dict[str, Any]],
    *,
    target_db: str | None = None,
    dialect: str | None = None,
    connector: Any | None = None,
) -> tuple[str, str, bool]:
    """Accept a complete LLM grounding atomically or preserve both inputs."""
    catalog_columns = _catalog_columns(tables, target_db, dialect, connector)
    grounded_question = result.grounded_question.strip()
    grounded_evidence = result.grounded_evidence.strip()
    if not grounded_question or not grounded_evidence:
        return question, evidence, False

    explicit_targets = _explicit_identifier_targets(evidence, catalog_columns)
    if any(physical not in grounded_evidence for physical in explicit_targets.values()):
        return question, evidence, False

    changed = grounded_question != question or grounded_evidence != evidence
    allowed_physical = {entry["physical"] for entry in catalog_columns}
    if changed and not any(
        physical in f"{grounded_question}\n{grounded_evidence}"
        for physical in allowed_physical
    ):
        return question, evidence, False

    return grounded_question, grounded_evidence, changed


def _validate_corrected_question(
    grounded_question: str,
    corrected_question: str,
    accepted_repairs: list[dict[str, str | int]],
    tables: list[dict[str, Any]],
    *,
    target_db: str | None,
    dialect: str | None,
    connector: Any | None,
) -> str:
    """Accept value-only question updates while locking every physical field."""
    candidate = corrected_question.strip()
    if not candidate:
        return grounded_question
    if not accepted_repairs:
        return grounded_question if candidate != grounded_question else candidate

    physical_columns = {
        entry["physical"]
        for entry in _catalog_columns(tables, target_db, dialect, connector)
    }
    original_fields = {
        physical for physical in physical_columns if physical in grounded_question
    }
    corrected_fields = {
        physical for physical in physical_columns if physical in candidate
    }
    if corrected_fields != original_fields:
        return grounded_question
    return candidate


def _unquote_literal(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _format_value_anchors(value_anchors: list[dict[str, Any]]) -> str:
    """Render only exact anchor facts needed to validate evidence values."""
    lines: list[str] = []
    for anchor in value_anchors:
        table = str(anchor.get("tbl") or "").strip()
        column = str(anchor.get("col") or "").strip()
        value = str(anchor.get("stored_value") or "")
        if anchor.get("kind", "value") == "absent" or not (table and column and value):
            continue
        lines.append(f'- {table}."{column}" has stored value {value!r}')
    return "\n".join(lines)


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


def _anchor_supports(
    repair: EvidenceLineRepair,
    value_anchors: list[dict[str, Any]],
) -> bool:
    """Return whether an exact value anchor proves the repaired scalar."""
    if not repair.table_name or not repair.column_name:
        return False

    wanted_table = repair.table_name.casefold().strip()
    wanted_column = repair.column_name.casefold().strip()
    wanted_value = _unquote_literal(repair.corrected_value)
    for anchor in value_anchors:
        anchor_table = str(anchor.get("tbl") or "").casefold().strip()
        anchor_column = str(anchor.get("col") or "").casefold().strip()
        anchor_value = str(anchor.get("stored_value") or "")
        if (
            anchor.get("kind", "value") != "absent"
            and anchor_table
            and (
                anchor_table == wanted_table
                or wanted_table.rsplit(".", 1)[-1] == anchor_table
            )
            and anchor_column == wanted_column
            and anchor_value == wanted_value
        ):
            return True
    return False


def _stored_value_supports(
    repair: EvidenceLineRepair,
    value_anchors: list[dict[str, Any]],
    tables: list[dict[str, Any]],
) -> bool:
    """Check exact value anchors first, then fall back to column samples."""
    return _anchor_supports(repair, value_anchors) or _sample_supports(repair, tables)


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


def _repair_preserves_field_side(repair: EvidenceLineRepair) -> bool:
    """Allow the repair to change only the operator or its right-hand scalar."""
    original_match = _COMPARISON_RE.search(repair.original_line)
    corrected_match = _COMPARISON_RE.search(repair.corrected_line)
    if original_match is None or corrected_match is None:
        return False

    original_left = repair.original_line[: original_match.start()]
    corrected_left = repair.corrected_line[: corrected_match.start()]
    if original_left != corrected_left:
        return False

    original_operator = original_match.group(0)
    corrected_operator = corrected_match.group(0)
    if repair.kind == "predicate_operator":
        return (
            corrected_operator == repair.corrected_value.strip()
            and repair.original_line[original_match.end() :]
            == repair.corrected_line[corrected_match.end() :]
        )
    if original_operator != corrected_operator:
        return False

    corrected_rhs = repair.corrected_line[corrected_match.end() :]
    value = _unquote_literal(repair.corrected_value)
    value_match = re.search(re.escape(value), corrected_rhs)
    if not value or value_match is None:
        return False

    allowed_start = value_match.start()
    allowed_end = value_match.end()
    if allowed_start > 0 and corrected_rhs[allowed_start - 1] in {"'", '"'}:
        allowed_start -= 1
    if allowed_end < len(corrected_rhs) and corrected_rhs[allowed_end] in {"'", '"'}:
        allowed_end += 1

    original_rhs = repair.original_line[original_match.end() :]
    changed = [
        opcode
        for opcode in SequenceMatcher(
            None, original_rhs, corrected_rhs, autojunk=False
        ).get_opcodes()
        if opcode[0] != "equal"
    ]
    return bool(changed) and all(
        allowed_start <= new_start <= new_end <= allowed_end
        for _, _, _, new_start, new_end in changed
    )


def _has_formula_on_right_hand_side(line: str) -> bool:
    """Detect arithmetic formulas without mistaking punctuation in field names."""
    comparison = _COMPARISON_RE.search(line)
    return bool(comparison and _FORMULA_OPERATOR_RE.search(line[comparison.end() :]))


def _valid_repair(
    repair: EvidenceLineRepair,
    line: str,
    question: str,
    tables: list[dict[str, Any]],
    value_anchors: list[dict[str, Any]],
) -> bool:
    if (
        repair.original_line != line
        or repair.corrected_line == line
        or "\n" in repair.original_line
        or "\n" in repair.corrected_line
        or not repair.corrected_value.strip()
        or _has_formula_on_right_hand_side(line)
        or not _is_local_line_correction(repair)
        or not _repair_preserves_field_side(repair)
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
        return _stored_value_supports(repair, value_anchors, tables)
    return _question_contains_literal(
        question, repair.corrected_value
    ) or _stored_value_supports(repair, value_anchors, tables)


def apply_evidence_repairs(
    evidence: str,
    repairs: list[EvidenceLineRepair],
    question: str,
    tables: list[dict[str, Any]],
    value_anchors: list[dict[str, Any]] | None = None,
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
        if not _valid_repair(
            repair, original_line, question, tables, list(value_anchors or [])
        ):
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
        target_db = path_state.get("target_db")
        connector = resolve_connector_from_tables(tables, state.get("connectors") or [])
        dialect = getattr(connector, "dialect", None)
        tables_block = format_tables_for_prompt(
            tables,
            target_db=target_db,
            dialect=dialect,
        )
        catalog_columns = _catalog_columns(tables, target_db, dialect, connector)
        qualified_columns_block = (
            "\n".join(
                (
                    f"- {entry['physical']}"
                    + (
                        f" | table: {entry['table_description']}"
                        if entry["table_description"]
                        else ""
                    )
                    + (
                        f" | column: {entry['column_description']}"
                        if entry["column_description"]
                        else ""
                    )
                )
                for entry in catalog_columns
            )
            or "(none)"
        )

        grounding_prompt = (
            f"Question:\n{question}\n\n"
            f"Evidence:\n{evidence}\n\n"
            "Relevant dialect-quoted columns (the only allowed mapping targets; "
            "copy references verbatim):\n"
            f"{qualified_columns_block}"
        )
        grounding_result = safe_invoke_structured_nr(
            [
                SystemMessage(content=_GROUNDING_SYSTEM_PROMPT),
                HumanMessage(content=grounding_prompt),
            ],
            EvidenceGroundingResult,
        )
        if grounding_result is None:
            grounded_question, grounded_evidence, grounding_applied = (
                question,
                evidence,
                False,
            )
        else:
            grounded_question, grounded_evidence, grounding_applied = (
                validate_schema_grounding(
                    question,
                    evidence,
                    grounding_result,
                    tables,
                    target_db=target_db,
                    dialect=dialect,
                    connector=connector,
                )
            )
        if grounding_result is None:
            self.logger.warning(
                "Evidence schema grounding failed — keeping question and evidence "
                "unchanged"
            )
        elif grounding_result.reasoning.strip():
            grounding_reasoning = grounding_result.reasoning.strip()
            record_thought(path_state, _GRAPH_NODE_NAME, grounding_reasoning)
        if grounding_applied:
            path_state["normalized_question"] = grounded_question
            self.logger.info("Applied LLM-grounded question and evidence")
        else:
            self.logger.info("No certain evidence schema groundings found")

        # Value correction is deliberately phase 2. It receives already-grounded
        # field references, and apply_evidence_repairs enforces that those field
        # references cannot change.
        value_anchors = list(state.get("value_anchors") or [])
        numbered_evidence = "\n".join(
            f"{index}| {line}"
            for index, line in enumerate(grounded_evidence.splitlines(), 1)
        )
        anchors_block = _format_value_anchors(value_anchors)
        values_context = (
            f"Verified database value anchors (use these first):\n{anchors_block}\n\n"
            if anchors_block
            else ""
        )
        prompt = (
            f"Sanitized question:\n{grounded_question}\n\n"
            f"Evidence (line numbers are metadata only):\n{numbered_evidence}\n\n"
            f"{values_context}"
            f"Relevant tables, columns, and fallback sample values:\n{tables_block}"
        )

        result = safe_invoke_structured_nr(
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)],
            EvidenceRefinementResult,
        )
        if result is None:
            self.logger.warning(
                "Evidence value refinement failed — keeping schema-grounded evidence"
            )
            return {"evidence": grounded_evidence, "path_state": path_state}

        refined, accepted = apply_evidence_repairs(
            grounded_evidence,
            result.repairs,
            grounded_question,
            tables,
            value_anchors,
        )
        final_question = _validate_corrected_question(
            grounded_question,
            result.corrected_question,
            accepted,
            tables,
            target_db=target_db,
            dialect=dialect,
            connector=connector,
        )
        if final_question != question:
            path_state["normalized_question"] = final_question
        if result.reasoning.strip():
            record_thought(path_state, _GRAPH_NODE_NAME, result.reasoning.strip())
        if accepted:
            path_state["evidence_repairs_applied"] = accepted
            self.logger.info("Applied %d evidence repair(s)", len(accepted))
        else:
            self.logger.info("No certain evidence repairs found")

        return {"evidence": refined, "path_state": path_state}


__all__ = [
    "EvidenceRefinementAgent",
    "EvidenceGroundingResult",
    "EvidenceRefinementResult",
    "EvidenceLineRepair",
    "validate_schema_grounding",
    "apply_evidence_repairs",
]
