# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentic, read-only validation of literals used by generated SQL."""

from __future__ import annotations

import json
import logging
from enum import StrEnum
from typing import Any, Literal

import sqlglot
from langchain_core.messages import SystemMessage
from pydantic import Field, model_validator
from sqlglot import exp

from auto_ontology.retrieval.text_to_sql.agents.empty_like_result_check import (
    _set_sql_code,
)
from auto_ontology.retrieval.text_to_sql.base import BaseAgent, record_thought
from auto_ontology.retrieval.text_to_sql.connector_routing import (
    resolve_connector_from_tables,
)
from auto_ontology.retrieval.text_to_sql.db_probe.executor import (
    ProbeExecutor,
    is_read_only_select,
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
    invoke_with_structured_output,
)

logger = logging.getLogger(__name__)

_GRAPH_NODE_NAME = "validate_sql_values"
MAX_TOOL_CALLS = 20
MAX_AGENT_ITERATIONS = 22
MAX_OBSERVATION_CHARS = 16_000

_SYSTEM_PROMPT = """\
You validate every database value used by a generated SQL query.

You receive an exhaustive checklist of column-bound literals. Work through every
checklist item. Use query_database to verify that the exact value exists in the
stated column. If it exists, keep it. If it does not, use further small queries
(LIKE is allowed for discovery) to learn the actual stored representation. Do not
guess. If the database is unavailable or several values remain plausible, keep the
original value.

The tool is read-only and limited to relevant tables. Keep probes narrow: select
only the needed column and use this search order for string values:
1. exact equality with the original value;
2. if that fails, trim leading and trailing whitespace from the value, try exact
   equality with the trimmed value, then search the same column with LIKE using the
   trimmed text;
3. only if those literal searches fail, consider case variants or semantic
   synonyms/symbol meanings supported by the question or evidence.
Never jump to a synonym before testing the trimmed literal and LIKE. Discovery LIKE
predicates are probes, not permission to weaken the generated SQL: use the exact
discovered stored value in corrected_sql.

Finish only after every checklist id has been reviewed. Return the complete SQL.
Change literals only; never alter projections, expressions, predicates, operators,
joins, tables, columns, grouping, ordering, or limits. Report every changed literal
as a substitution and cite both the successful exact query showing the original is
absent and a successful proof query whose returned rows contain the replacement.
Return exactly one review per checklist id. Mark it exists, corrected, or unverified;
include the final literal and successful proof query for exists/corrected. A cached
proven finding may be reused with no new tool call.

If a changed literal is also present in Evidence, return the complete evidence with
only that literal changed. Otherwise copy Evidence exactly. Never alter evidence
field references, formulas, operators, mappings, or prose."""


class ValueValidationActionType(StrEnum):
    QUERY = "query_database"
    FINISH = "finish"


class ValueQueryArguments(StrictLLMOutputModel):
    sql: str
    purpose: str


class ValueSubstitution(StrictLLMOutputModel):
    check_id: str
    original_literal: str
    replacement_literal: str
    absence_query: str
    proof_query: str


class ValueReview(StrictLLMOutputModel):
    check_id: str
    original_literal: str
    final_literal: str
    status: Literal["exists", "corrected", "unverified"]
    proof_query: str = ""


class ValueValidationAction(StrictLLMOutputModel):
    action: ValueValidationActionType
    arguments: ValueQueryArguments | None = None
    corrected_sql: str | None = None
    corrected_evidence: str | None = None
    reviewed_check_ids: list[str] = Field(default_factory=list)
    reviews: list[ValueReview] = Field(default_factory=list)
    substitutions: list[ValueSubstitution] = Field(default_factory=list)
    reasoning: str = ""

    @model_validator(mode="after")
    def _validate_action(self) -> "ValueValidationAction":
        if self.action == ValueValidationActionType.QUERY:
            if self.arguments is None:
                raise ValueError("query_database requires arguments")
            if self.corrected_sql is not None:
                raise ValueError("query_database cannot return corrected SQL")
        else:
            if not (self.corrected_sql or "").strip():
                raise ValueError("finish requires corrected_sql")
            if self.arguments is not None:
                raise ValueError("finish cannot include query arguments")
        return self


def _single_column(expression: exp.Expression) -> exp.Column | None:
    columns = list(expression.find_all(exp.Column))
    if isinstance(expression, exp.Column):
        columns.insert(0, expression)
    unique = {column.sql(): column for column in columns}
    if len(unique) != 1:
        return None
    return next(iter(unique.values()))


def _literal_item(
    *,
    index: int,
    predicate: exp.Expression,
    column: exp.Column,
    literal: exp.Literal,
    dialect: str | None,
) -> dict[str, Any]:
    return {
        "id": f"value_{index}",
        "column": column.sql(dialect=dialect),
        "predicate": predicate.sql(dialect=dialect),
        "literal": literal.sql(dialect=dialect),
        "value": literal.this,
        "is_string": literal.is_string,
    }


def extract_value_checks(sql: str, dialect: str | None = None) -> list[dict[str, Any]]:
    """Return column-bound literals from predicates, including CASE conditions."""
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return []

    found: list[tuple[exp.Expression, exp.Column, exp.Literal]] = []
    binary_types = (
        exp.EQ,
        exp.NEQ,
        exp.GT,
        exp.GTE,
        exp.LT,
        exp.LTE,
        exp.Like,
        exp.ILike,
    )
    for predicate in tree.walk():
        if isinstance(predicate, binary_types):
            left = predicate.this
            right = predicate.expression
            if isinstance(right, exp.Literal):
                column = _single_column(left)
                if column is not None:
                    found.append((predicate, column, right))
            elif isinstance(left, exp.Literal):
                column = _single_column(right)
                if column is not None:
                    found.append((predicate, column, left))
        elif isinstance(predicate, exp.In):
            column = _single_column(predicate.this)
            if column is not None and predicate.args.get("query") is None:
                for value in predicate.expressions:
                    if isinstance(value, exp.Literal):
                        found.append((predicate, column, value))
        elif isinstance(predicate, exp.Between):
            column = _single_column(predicate.this)
            if column is not None:
                for key in ("low", "high"):
                    value = predicate.args.get(key)
                    if isinstance(value, exp.Literal):
                        found.append((predicate, column, value))

    checks: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for predicate, column, literal in found:
        key = (id(predicate), column.sql(), literal.sql())
        if key in seen:
            continue
        seen.add(key)
        checks.append(
            _literal_item(
                index=len(checks) + 1,
                predicate=predicate,
                column=column,
                literal=literal,
                dialect=dialect,
            )
        )
    return checks


def _normalize_query(sql: str) -> str:
    return " ".join(sql.strip().rstrip(";").split())


def _allowed_probe_tables(sql: str, tables: list[dict], dialect: str | None) -> bool:
    """Restrict model-authored probes to the already-retrieved table scope."""
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return False
    allowed = {
        str(table.get("name") or "").strip().casefold()
        for table in tables
        if table.get("name")
    }
    ctes = {
        str(cte.alias_or_name or "").strip().casefold()
        for cte in tree.find_all(exp.CTE)
    }
    referenced = {
        str(table.name or "").strip().casefold()
        for table in tree.find_all(exp.Table)
        if str(table.name or "").strip().casefold() not in ctes
    }
    return bool(referenced) and referenced <= allowed


def _literal_skeleton(sql: str, dialect: str | None) -> str | None:
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return None

    def replace(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Literal):
            return exp.Literal.string("__VALUE__")
        return node

    return tree.transform(replace, copy=True).sql(dialect=dialect)


def _all_literals(sql: str, dialect: str | None) -> list[str]:
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return []
    return [literal.sql(dialect=dialect) for literal in tree.find_all(exp.Literal)]


def _result_contains_literal(result: dict[str, Any], literal: str) -> bool:
    target = literal.strip()
    if len(target) >= 2 and target[0] == target[-1] and target[0] in {"'", '"'}:
        target = target[1:-1]
    target = target.casefold()
    rows = result.get("rows") or []
    return any(
        str(value).strip().casefold() == target
        for row in rows
        if isinstance(row, dict)
        for value in row.values()
    )


def _validate_evidence(
    original: str,
    candidate: str | None,
    substitutions: list[ValueSubstitution],
) -> str:
    candidate = candidate if candidate is not None else original
    if candidate == original:
        return original
    reverted = candidate
    changed_relevant_value = False
    for substitution in substitutions:
        old = substitution.original_literal
        new = substitution.replacement_literal
        if new in reverted and (old in original or old.strip("'\"") in original):
            reverted = reverted.replace(new, old)
            reverted = reverted.replace(new.strip("'\""), old.strip("'\""))
            changed_relevant_value = True
    if changed_relevant_value and reverted == original:
        return candidate
    return original


def _validate_final_action(
    action: ValueValidationAction,
    *,
    original_sql: str,
    original_evidence: str,
    checks: list[dict[str, Any]],
    successful_queries: dict[str, dict[str, Any]],
    dialect: str | None,
) -> tuple[str, str, list[dict[str, str]], bool]:
    expected_ids = {check["id"] for check in checks}
    if set(action.reviewed_check_ids) != expected_ids:
        return original_sql, original_evidence, [], False

    reviews_by_id = {review.check_id: review for review in action.reviews}
    if set(reviews_by_id) != expected_ids or len(action.reviews) != len(expected_ids):
        return original_sql, original_evidence, [], False
    for check in checks:
        review = reviews_by_id[check["id"]]
        if review.original_literal != check["literal"]:
            return original_sql, original_evidence, [], False
        if review.status == "exists" and review.final_literal != check["literal"]:
            return original_sql, original_evidence, [], False
        if review.status == "corrected" and review.final_literal == check["literal"]:
            return original_sql, original_evidence, [], False

    candidate_sql = (action.corrected_sql or "").strip()
    if not is_read_only_select(candidate_sql):
        return original_sql, original_evidence, [], False
    if _literal_skeleton(candidate_sql, dialect) != _literal_skeleton(
        original_sql, dialect
    ):
        return original_sql, original_evidence, [], False

    checks_by_id = {check["id"]: check for check in checks}
    accepted: list[dict[str, str]] = []
    reported_pairs: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    for substitution in action.substitutions:
        check = checks_by_id.get(substitution.check_id)
        absence = successful_queries.get(_normalize_query(substitution.absence_query))
        proof = successful_queries.get(_normalize_query(substitution.proof_query))
        if (
            check is None
            or substitution.check_id in seen_ids
            or substitution.original_literal != check["literal"]
            or substitution.original_literal == substitution.replacement_literal
            or absence is None
            or not absence.get("ok")
            or int(absence.get("row_count") or 0) != 0
            or proof is None
            or not proof.get("ok")
            or not _result_contains_literal(proof, substitution.replacement_literal)
        ):
            return original_sql, original_evidence, [], False
        seen_ids.add(substitution.check_id)
        pair = (substitution.original_literal, substitution.replacement_literal)
        review = reviews_by_id[substitution.check_id]
        if (
            review.status != "corrected"
            or review.final_literal != substitution.replacement_literal
            or _normalize_query(review.proof_query)
            != _normalize_query(substitution.proof_query)
        ):
            return original_sql, original_evidence, [], False
        reported_pairs.append(pair)
        accepted.append(substitution.model_dump())

    before_literals = _all_literals(original_sql, dialect)
    after_literals = _all_literals(candidate_sql, dialect)
    if len(before_literals) != len(after_literals):
        return original_sql, original_evidence, [], False
    changed_pairs = [
        pair for pair in zip(before_literals, after_literals) if pair[0] != pair[1]
    ]
    if sorted(changed_pairs) != sorted(reported_pairs):
        return original_sql, original_evidence, [], False
    if not changed_pairs and candidate_sql != original_sql:
        return original_sql, original_evidence, [], False

    candidate_checks = extract_value_checks(candidate_sql, dialect)
    if len(candidate_checks) != len(checks):
        return original_sql, original_evidence, [], False
    changed_check_ids: set[str] = set()
    for original_check, candidate_check in zip(checks, candidate_checks):
        if original_check["column"] != candidate_check["column"]:
            return original_sql, original_evidence, [], False
        if original_check["literal"] != candidate_check["literal"]:
            changed_check_ids.add(original_check["id"])
    if changed_check_ids != seen_ids:
        return original_sql, original_evidence, [], False

    evidence = _validate_evidence(
        original_evidence,
        action.corrected_evidence,
        action.substitutions,
    )
    return candidate_sql, evidence, accepted, True


def _render_prompt(
    *,
    question: str,
    sql: str,
    evidence: str,
    checks: list[dict[str, Any]],
    tables_block: str,
    connector_context: str,
    observations: list[dict[str, Any]],
    findings: dict[str, Any],
    calls_remaining: int,
) -> str:
    observations_text = json.dumps(observations, default=str)
    if len(observations_text) > MAX_OBSERVATION_CHARS:
        observations_text = observations_text[-MAX_OBSERVATION_CHARS:]
    return (
        f"Question:\n{question}\n\n"
        f"Generated SQL:\n{sql}\n\n"
        f"Evidence:\n{evidence or '(none)'}\n\n"
        f"Connector:\n{connector_context}\n\n"
        f"Value checklist:\n{json.dumps(checks, default=str, indent=2)}\n\n"
        f"Relevant schema:\n{tables_block}\n\n"
        f"Cached proven findings:\n{json.dumps(findings, default=str)}\n\n"
        f"Tool observations:\n{observations_text}\n\n"
        f"Database queries remaining: {calls_remaining}"
    )


class SQLValueValidationAgent(BaseAgent):
    """Use a bounded query-tool loop to validate generated SQL literals."""

    def __init__(self) -> None:
        super().__init__("sql_value_validation")

    def validate_input(self, state: AgentState) -> bool:
        return True

    def execute(self, state: AgentState) -> dict[str, Any]:
        if not state.get("validate_sql_values", False):
            return {}

        path_state = dict(state.get("path_state") or {})
        generation_result = path_state.get("sql_generation_result")
        sql_code = (
            getattr(generation_result, "sql_code", "")
            or path_state.get("sql_code")
            or ""
        ).strip()
        if not sql_code:
            return {}

        tables = list(path_state.get("relevant_tables") or [])
        connector = resolve_connector_from_tables(tables, state.get("connectors") or [])
        dialect = getattr(connector, "dialect", None)
        checks = extract_value_checks(sql_code, dialect)
        if not checks:
            return {}

        evidence = state.get("evidence") or ""
        cache = dict(state.get("sql_value_validation_cache") or {})
        query_cache = dict(cache.get("queries") or {})
        findings = dict(cache.get("findings") or {})
        connector_key = (
            f"{getattr(connector, 'database_name', '')}:"
            f"{getattr(connector, 'dialect', '')}"
        )
        relevant_findings = {
            key: value
            for key, value in findings.items()
            if isinstance(value, dict)
            and value.get("connector") == connector_key
            and any(
                value.get("column") == check["column"]
                and value.get("original_literal") == check["literal"]
                for check in checks
            )
        }
        observations: list[dict[str, Any]] = []
        successful_queries: dict[str, dict[str, Any]] = {
            key.removeprefix(f"{connector_key}::"): value
            for key, value in query_cache.items()
            if key.startswith(f"{connector_key}::")
            and isinstance(value, dict)
            and value.get("ok")
        }
        tables_block = format_tables_for_prompt(
            tables,
            target_db=path_state.get("target_db"),
            dialect=dialect,
        )
        executor = ProbeExecutor(connector, max_calls=MAX_TOOL_CALLS)
        action: ValueValidationAction | None = None

        for _ in range(MAX_AGENT_ITERATIONS):
            prompt = _render_prompt(
                question=get_question_for_processing(state),
                sql=sql_code,
                evidence=evidence,
                checks=checks,
                tables_block=tables_block,
                connector_context=connector_key,
                observations=observations,
                findings=relevant_findings,
                calls_remaining=executor.budget_left,
            )
            try:
                action = invoke_with_structured_output(
                    state["llm"],
                    [
                        SystemMessage(content=_SYSTEM_PROMPT),
                        SystemMessage(content=prompt),
                    ],
                    ValueValidationAction,
                )
            except Exception:
                logger.exception("SQL value-validation reasoning failed")
                action = None
            if action is None:
                break
            if action.action == ValueValidationActionType.FINISH:
                break

            assert action.arguments is not None
            probe_sql = action.arguments.sql
            signature = _normalize_query(probe_sql)
            cache_key = f"{connector_key}::{signature}"
            if cache_key in query_cache:
                result = query_cache[cache_key]
                source = "cache"
            elif not _allowed_probe_tables(probe_sql, tables, dialect):
                result = {
                    "ok": False,
                    "rows": None,
                    "error": "query references a table outside the relevant schema",
                }
                source = "rejected"
            else:
                result = executor.run(probe_sql, purpose=action.arguments.purpose)
                query_cache[cache_key] = result
                if result.get("ok"):
                    successful_queries[signature] = result
                source = "database"
            observations.append(
                {
                    "sql": probe_sql,
                    "purpose": action.arguments.purpose,
                    "source": source,
                    "result": result,
                }
            )

        cache["queries"] = query_cache
        updates: dict[str, Any] = {"sql_value_validation_cache": cache}
        if action is None or action.action != ValueValidationActionType.FINISH:
            return updates

        corrected_sql, corrected_evidence, accepted, final_valid = (
            _validate_final_action(
                action,
                original_sql=sql_code,
                original_evidence=evidence,
                checks=checks,
                successful_queries=successful_queries,
                dialect=dialect,
            )
        )
        if accepted:
            _set_sql_code(path_state, corrected_sql)
            path_state["sql_value_substitutions"] = accepted
            updates["sql_value_validation_cache"] = cache
            updates["path_state"] = path_state
            if corrected_evidence != evidence:
                updates["evidence"] = corrected_evidence
            record_thought(
                path_state,
                _GRAPH_NODE_NAME,
                action.reasoning.strip()
                or f"Validated values and applied {len(accepted)} correction(s).",
            )
        elif final_valid and action.reasoning.strip():
            record_thought(path_state, _GRAPH_NODE_NAME, action.reasoning.strip())
            updates["path_state"] = path_state

        checks_by_id = {check["id"]: check for check in checks}
        for review in action.reviews if final_valid else []:
            check = checks_by_id.get(review.check_id)
            proof = successful_queries.get(_normalize_query(review.proof_query))
            if (
                check is None
                or review.status == "unverified"
                or proof is None
                or not proof.get("ok")
                or not _result_contains_literal(proof, review.final_literal)
            ):
                continue
            finding_key = json.dumps(
                {
                    "connector": connector_key,
                    "column": check["column"],
                    "value": check["value"],
                },
                sort_keys=True,
            )
            findings[finding_key] = {
                "connector": connector_key,
                "column": check["column"],
                "original_literal": check["literal"],
                "final_literal": review.final_literal,
                "status": review.status,
                "proof_query": review.proof_query,
            }
        cache["findings"] = findings
        updates["sql_value_validation_cache"] = cache
        return updates


__all__ = [
    "MAX_AGENT_ITERATIONS",
    "MAX_TOOL_CALLS",
    "SQLValueValidationAgent",
    "ValueSubstitution",
    "ValueReview",
    "ValueValidationAction",
    "ValueValidationActionType",
    "extract_value_checks",
]
