# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentic, read-only validation of literals used by generated SQL."""

from __future__ import annotations

import json
import logging
import re
from enum import StrEnum
from typing import Any

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
# One line per probe is enough for the model to choose the next query. Only the
# latest lines are repeated; earlier proofs stay in the query cache.
MAX_PROMPT_OBSERVATIONS = 8
_MAX_OBSERVATION_VALUES = 5

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
discovered stored value as the replacement literal.

Finish only after every checklist id has been reviewed. Change literals only; never
alter projections, expressions, predicates, operators, joins, tables, columns,
grouping, ordering, or limits. Report only the literals that must change. Each
substitution is a check_id and the exact stored SQL literal (quotes included for
strings). Do not return the SQL or the evidence; both are updated from those
substitutions. A substitution is accepted only when a successful probe returned
no rows for the original literal and another successful probe returned the
replacement. A cached proven finding may be reused with no new tool call.

When a changed literal is also present in Evidence, it is updated in place. Never
alter evidence field references, formulas, operators, mappings, or prose."""


class ValueValidationActionType(StrEnum):
    QUERY = "query_database"
    FINISH = "finish"


class ValueQueryArguments(StrictLLMOutputModel):
    sql: str
    purpose: str


class ValueSubstitution(StrictLLMOutputModel):
    check_id: str
    replacement_literal: str


class ValueValidationAction(StrictLLMOutputModel):
    action: ValueValidationActionType
    arguments: ValueQueryArguments | None = None
    substitutions: list[ValueSubstitution] = Field(default_factory=list)
    reasoning: str = ""

    @model_validator(mode="after")
    def _validate_action(self) -> "ValueValidationAction":
        if self.action == ValueValidationActionType.QUERY:
            if self.arguments is None:
                raise ValueError("query_database requires arguments")
            if self.substitutions:
                raise ValueError("query_database cannot include substitutions")
        else:
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


def _column_name(column_sql: str) -> str:
    return column_sql.split(".")[-1].strip('"`[]').casefold()


def _sql_mentions_column(sql: str, column_sql: str) -> bool:
    name = re.escape(_column_name(column_sql))
    return re.search(rf"(?<![\w]){name}(?![\w])", sql, re.IGNORECASE) is not None


def _sql_mentions_literal(sql: str, literal: str) -> bool:
    folded = sql.casefold()
    if literal.casefold() in folded:
        return True
    bare = literal.strip("'\"").casefold()
    return bool(bare) and bare in folded


def _as_sql_literal(original: str, replacement: str) -> str:
    """Keep the checklist's quoting when the model returns a bare value."""
    replacement = replacement.strip()
    if (
        len(original) >= 2
        and original[0] == original[-1]
        and original[0] in {"'", '"'}
        and not (
            len(replacement) >= 2
            and replacement[0] == replacement[-1]
            and replacement[0] in {"'", '"'}
        )
    ):
        quote = original[0]
        return f"{quote}{replacement.replace(quote, quote * 2)}{quote}"
    return replacement


def _replace_once(sql: str, old: str, new: str) -> str | None:
    """Replace one checklist literal without touching a longer token."""
    if len(old) >= 2 and old[0] == old[-1] and old[0] in {"'", '"'}:
        index = sql.find(old)
        if index < 0:
            return None
        return sql[:index] + new + sql[index + len(old) :]
    match = re.search(rf"(?<![\w.]){re.escape(old)}(?![\w.])", sql)
    if match is None:
        return None
    return sql[: match.start()] + new + sql[match.end() :]


def _match_absence(
    queries: dict[str, dict[str, Any]],
    column_sql: str,
    literal: str,
) -> dict[str, Any] | None:
    for result in queries.values():
        sql = str(result.get("sql") or "")
        if (
            result.get("ok")
            and int(result.get("row_count") or 0) == 0
            and _sql_mentions_column(sql, column_sql)
            and _sql_mentions_literal(sql, literal)
        ):
            return result
    return None


def _match_proof(
    queries: dict[str, dict[str, Any]],
    column_sql: str,
    literal: str,
) -> dict[str, Any] | None:
    for result in queries.values():
        sql = str(result.get("sql") or "")
        if (
            result.get("ok")
            and _sql_mentions_column(sql, column_sql)
            and _result_contains_literal(result, literal)
        ):
            return result
    return None


def _replace_evidence(evidence: str, accepted: list[dict[str, str]]) -> str:
    updated = evidence
    for item in accepted:
        old = item["original_literal"]
        new = item["replacement_literal"]
        if old and old in updated:
            updated = updated.replace(old, new)
    return updated


def _apply_proven_substitutions(
    action: ValueValidationAction,
    *,
    original_sql: str,
    original_evidence: str,
    checks: list[dict[str, Any]],
    successful_queries: dict[str, dict[str, Any]],
    dialect: str | None,
) -> tuple[str, str, list[dict[str, str]], bool]:
    """Apply literal edits the probes proved. Reject the whole finish otherwise."""
    if not action.substitutions:
        return original_sql, original_evidence, [], True

    checks_by_id = {check["id"]: check for check in checks}
    replacements: dict[str, str] = {}
    accepted: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for substitution in action.substitutions:
        check = checks_by_id.get(substitution.check_id)
        if check is None or substitution.check_id in seen_ids:
            return original_sql, original_evidence, [], False
        replacement = _as_sql_literal(
            check["literal"], substitution.replacement_literal
        )
        if replacement == check["literal"]:
            return original_sql, original_evidence, [], False
        absence = _match_absence(successful_queries, check["column"], check["literal"])
        proof = _match_proof(successful_queries, check["column"], replacement)
        if absence is None or proof is None:
            return original_sql, original_evidence, [], False
        seen_ids.add(substitution.check_id)
        replacements[substitution.check_id] = replacement
        accepted.append(
            {
                "check_id": substitution.check_id,
                "original_literal": check["literal"],
                "replacement_literal": replacement,
            }
        )

    candidate_sql = original_sql
    for check in checks:
        replacement = replacements.get(check["id"])
        if replacement is None:
            continue
        replaced = _replace_once(candidate_sql, check["literal"], replacement)
        if replaced is None:
            return original_sql, original_evidence, [], False
        candidate_sql = replaced

    if not is_read_only_select(candidate_sql):
        return original_sql, original_evidence, [], False
    if _literal_skeleton(candidate_sql, dialect) != _literal_skeleton(
        original_sql, dialect
    ):
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
            if replacements.get(original_check["id"]) != candidate_check["literal"]:
                return original_sql, original_evidence, [], False
    if changed_check_ids != seen_ids:
        return original_sql, original_evidence, [], False

    return (
        candidate_sql,
        _replace_evidence(original_evidence, accepted),
        accepted,
        True,
    )


def _column_types(tables: list[dict]) -> dict[str, str]:
    types: dict[str, str] = {}
    for table in tables:
        columns = table.get("columns")
        if not isinstance(columns, list):
            continue
        for column in columns:
            if not isinstance(column, dict) or not column.get("name"):
                continue
            types.setdefault(
                str(column["name"]).casefold(),
                str(column.get("data_type") or "unknown"),
            )
    return types


def _checked_columns_block(checks: list[dict[str, Any]], tables: list[dict]) -> str:
    """Names and types of the columns the checklist actually binds."""
    types = _column_types(tables)
    lines: list[str] = []
    seen: set[str] = set()
    for check in checks:
        column = str(check["column"])
        key = column.casefold()
        if key in seen:
            continue
        seen.add(key)
        data_type = types.get(_column_name(column), "unknown")
        lines.append(f"- {column} ({data_type})")
    return "\n".join(lines) or "(none)"


def _render_checks(checks: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{check['id']}: {check['literal']} | {check['predicate']}" for check in checks
    )


def _render_findings(findings: dict[str, Any]) -> str:
    lines: list[str] = []
    for value in findings.values():
        if not isinstance(value, dict):
            continue
        lines.append(
            f"{value.get('column')} {value.get('original_literal')} -> "
            f"{value.get('final_literal')} ({value.get('status')})"
        )
    return "\n".join(lines) or "(none)"


def _observation_line(sql: str, result: dict[str, Any]) -> str:
    """One probe line: outcome, row count, and a few values from the first column."""
    compact_sql = " ".join(sql.split())
    if not result.get("ok"):
        error = " ".join(str(result.get("error") or "failed").split())
        return f"FAIL {compact_sql} :: {error[:200]}"
    values: list[str] = []
    for row in result.get("rows") or []:
        if not isinstance(row, dict) or not row:
            continue
        values.append(str(next(iter(row.values()))))
        if len(values) >= _MAX_OBSERVATION_VALUES:
            break
    row_count = int(result.get("row_count") or 0)
    extra = row_count - len(values)
    sample = ", ".join(values)
    if extra > 0:
        sample = f"{sample} (+{extra} more)" if sample else f"(+{extra} more)"
    line = f"OK rows={row_count} {compact_sql}"
    if sample:
        line = f"{line} :: {sample}"
    return line


def _render_observations(observations: list[str]) -> str:
    if not observations:
        return "(none)"
    shown = observations[-MAX_PROMPT_OBSERVATIONS:]
    omitted = len(observations) - len(shown)
    lines = [f"{omitted} earlier probes omitted"] if omitted else []
    lines.extend(shown)
    return "\n".join(lines)


def _store_finding(
    findings: dict[str, Any],
    connector_key: str,
    check: dict[str, Any],
    *,
    final_literal: str,
    status: str,
) -> None:
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
        "final_literal": final_literal,
        "status": status,
    }


def _render_prompt(
    *,
    question: str,
    sql: str,
    evidence: str,
    checks: list[dict[str, Any]],
    columns_block: str,
    connector_context: str,
    observations: list[str],
    findings: dict[str, Any],
    calls_remaining: int,
) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Generated SQL:\n{sql}\n\n"
        f"Evidence:\n{evidence or '(none)'}\n\n"
        f"Connector:\n{connector_context}\n\n"
        f"Checked columns:\n{columns_block}\n\n"
        f"Value checklist:\n{_render_checks(checks)}\n\n"
        f"Cached proven findings:\n{_render_findings(findings)}\n\n"
        f"Tool observations:\n{_render_observations(observations)}\n\n"
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
        observations: list[str] = []
        successful_queries: dict[str, dict[str, Any]] = {
            key.removeprefix(f"{connector_key}::"): value
            for key, value in query_cache.items()
            if key.startswith(f"{connector_key}::")
            and isinstance(value, dict)
            and value.get("ok")
        }
        columns_block = _checked_columns_block(checks, tables)
        executor = ProbeExecutor(connector, max_calls=MAX_TOOL_CALLS)
        action: ValueValidationAction | None = None

        for _ in range(MAX_AGENT_ITERATIONS):
            prompt = _render_prompt(
                question=get_question_for_processing(state),
                sql=sql_code,
                evidence=evidence,
                checks=checks,
                columns_block=columns_block,
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
            elif not _allowed_probe_tables(probe_sql, tables, dialect):
                result = {
                    "ok": False,
                    "rows": None,
                    "error": "query references a table outside the relevant schema",
                }
            else:
                result = executor.run(probe_sql, purpose=action.arguments.purpose)
                query_cache[cache_key] = result
            if result.get("ok"):
                successful_queries[signature] = result
            observations.append(_observation_line(probe_sql, result))

        cache["queries"] = query_cache
        updates: dict[str, Any] = {"sql_value_validation_cache": cache}
        if action is None or action.action != ValueValidationActionType.FINISH:
            return updates

        corrected_sql, corrected_evidence, accepted, final_valid = (
            _apply_proven_substitutions(
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

        if final_valid:
            accepted_ids = {item["check_id"] for item in accepted}
            checks_by_id = {check["id"]: check for check in checks}
            for item in accepted:
                check = checks_by_id[item["check_id"]]
                _store_finding(
                    findings,
                    connector_key,
                    check,
                    final_literal=item["replacement_literal"],
                    status="corrected",
                )
            for check in checks:
                if check["id"] in accepted_ids:
                    continue
                if (
                    _match_proof(successful_queries, check["column"], check["literal"])
                    is None
                ):
                    continue
                _store_finding(
                    findings,
                    connector_key,
                    check,
                    final_literal=check["literal"],
                    status="exists",
                )
        cache["findings"] = findings
        updates["sql_value_validation_cache"] = cache
        return updates


__all__ = [
    "MAX_AGENT_ITERATIONS",
    "MAX_TOOL_CALLS",
    "SQLValueValidationAgent",
    "ValueSubstitution",
    "ValueValidationAction",
    "ValueValidationActionType",
    "extract_value_checks",
]
