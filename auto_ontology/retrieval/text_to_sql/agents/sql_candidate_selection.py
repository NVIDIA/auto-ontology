# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Choose between the semantic and schema-only SQL candidates.

Stages, each consulted only when the previous one leaves the choice open:

1. Static checks: a candidate that does not parse, or that the deterministic
   validators reject (degenerate, vacuous GROUP BY, missing aggregation), loses.
2. Execution: both candidates run read-only through ``ProbeExecutor``. One that
   errors, returns no rows, or returns only NULLs loses. Identical result sets
   are an agreement, and the semantic candidate is kept.
3. Pairwise LLM judge over the SQL and the result previews, asked in both
   orders. A verdict counts only when both orders name the same candidate.
4. Description rerank against the question and evidence, as the tie-breaker.

The semantic candidate is the default whenever no stage decides.
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import sqlglot
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from auto_ontology.connectors.base import SQLDatabase
from auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation import (
    detect_degenerate_sql,
    detect_missing_aggregation,
    detect_vacuous_group_by,
    quote_known_mixed_case_identifiers,
)
from auto_ontology.retrieval.text_to_sql.db_probe.executor import ProbeExecutor
from auto_ontology.retrieval.text_to_sql.models import (
    SQLGenerationModel,
    SQLJudgeModel,
)
from auto_ontology.retrieval.text_to_sql.prompts import (
    SQL_JUDGE_SYSTEM_PROMPT,
    SQL_JUDGE_USER_PROMPT,
)
from auto_ontology.utils.llm_invoke import safe_invoke_with_structured_output
from auto_ontology.utils.rerank import get_rerank_kwargs, rerank_hits

logger = logging.getLogger(__name__)

DEFAULT_SOURCE = "semantic"
_PREVIEW_ROWS = 10
_PREVIEW_CELL_CHARS = 80


@dataclass
class SQLCandidate:
    source: str
    response: SQLGenerationModel
    messages: list[BaseMessage]
    static_issue: str = ""
    execution: dict[str, Any] | None = None
    judge_votes: list[str] = field(default_factory=list)
    rerank_score: float | None = None

    @property
    def sql(self) -> str:
        return self.response.sql_code

    def summary(self) -> dict[str, Any]:
        execution = None
        if self.execution is not None:
            execution = {
                "ok": self.execution.get("ok"),
                "row_count": self.execution.get("row_count"),
                "error": self.execution.get("error"),
            }
        return {
            "source": self.source,
            "sql_code": self.sql,
            "description": self.response.description,
            "static_issue": self.static_issue or None,
            "execution": execution,
            "judge_votes": self.judge_votes,
            "rerank_score": self.rerank_score,
        }


@dataclass
class SelectionContext:
    llm: Any
    question: str
    evidence: str
    tables_text: str
    connector: SQLDatabase | None
    dialect: str | None
    target_db: str | None
    relevant_tables: list[dict]


@dataclass
class Selection:
    selected: SQLCandidate
    stage: str
    reason: str


def static_issue(sql: str, dialect: str | None, target_db: str | None) -> str:
    """The first deterministic reason *sql* cannot answer the question, or ``""``."""
    try:
        sqlglot.parse_one(sql, read=dialect or None)
    except sqlglot.errors.SqlglotError as exc:
        return f"does not parse: {str(exc).splitlines()[0]}"
    checks = (
        lambda: detect_degenerate_sql(sql, dialect),
        lambda: detect_vacuous_group_by(sql, dialect, target_db),
        lambda: detect_missing_aggregation(sql, dialect, target_db),
    )
    for check in checks:
        try:
            reason = check()
        except Exception:
            logger.warning("Static SQL check failed; skipping it.", exc_info=True)
            continue
        if reason:
            return reason
    return ""


def _normalize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    if math.isnan(number):
        return None
    return round(number, 6)


def result_key(rows: list[dict[str, Any]]) -> tuple:
    """Order-insensitive fingerprint of a result set, ignoring column names."""
    normalized = [tuple(_normalize(v) for v in row.values()) for row in rows]
    return tuple(sorted(normalized, key=repr))


def execution_issue(execution: dict[str, Any]) -> str:
    """Why an executed candidate cannot be the answer, or ``""``."""
    if not execution.get("ok"):
        return f"execution error: {execution.get('error')}"
    rows = execution.get("rows") or []
    if not rows:
        return "returned no rows"
    if all(_normalize(v) is None for row in rows for v in row.values()):
        return "returned only NULL values"
    return ""


def format_result_preview(execution: dict[str, Any] | None) -> str:
    if execution is None:
        return "(not executed)"
    if not execution.get("ok"):
        return f"Error: {execution.get('error')}"
    rows = execution.get("rows") or []
    if not rows:
        return "(no rows)"
    columns = list(rows[0].keys())

    def cell(value: Any) -> str:
        text = "NULL" if _normalize(value) is None else str(value)
        text = " ".join(text.split())
        if len(text) > _PREVIEW_CELL_CHARS:
            text = text[: _PREVIEW_CELL_CHARS - 3] + "..."
        return text

    lines = [" | ".join(str(c) for c in columns)]
    lines += [
        " | ".join(cell(row.get(c)) for c in columns) for row in rows[:_PREVIEW_ROWS]
    ]
    total = execution.get("row_count") or len(rows)
    more = "+" if execution.get("truncated") else ""
    shown = min(len(rows), _PREVIEW_ROWS)
    return f"{total}{more} row(s), showing {shown}:\n" + "\n".join(lines)


def _execute(candidates: list[SQLCandidate], ctx: SelectionContext) -> None:
    """Run each candidate inline: connectors can be bound to the calling thread."""
    with ProbeExecutor(ctx.connector, max_calls=len(candidates)) as executor:
        for candidate in candidates:
            sql = quote_known_mixed_case_identifiers(
                candidate.sql, ctx.relevant_tables, ctx.dialect
            )
            candidate.execution = executor.run(sql, purpose="sql candidate selection")


def _judge_once(
    ctx: SelectionContext, first: SQLCandidate, second: SQLCandidate
) -> str | None:
    """One judge call with *first* shown as A; returns the winner's source."""
    prompt = SQL_JUDGE_USER_PROMPT.format(
        question=ctx.question,
        evidence=ctx.evidence or "(none)",
        tables=ctx.tables_text,
        sql_a=first.sql,
        result_a=format_result_preview(first.execution),
        sql_b=second.sql,
        result_b=format_result_preview(second.execution),
    )
    messages = [
        SystemMessage(content=SQL_JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    try:
        verdict = safe_invoke_with_structured_output(ctx.llm, messages, SQLJudgeModel)
    except Exception:
        logger.warning("SQL judge call failed.", exc_info=True)
        return None
    if verdict is None:
        return None
    logger.info(
        "SQL judge (A=%s): %s — %s", first.source, verdict.winner, verdict.thought
    )
    return first.source if verdict.winner == "A" else second.source


def _judge(ctx: SelectionContext, a: SQLCandidate, b: SQLCandidate) -> None:
    with ThreadPoolExecutor(max_workers=2) as pool:
        forward = pool.submit(_judge_once, ctx, a, b)
        swapped = pool.submit(_judge_once, ctx, b, a)
        votes = [forward.result(), swapped.result()]
    for vote in votes:
        if vote == a.source:
            a.judge_votes.append(vote)
        elif vote == b.source:
            b.judge_votes.append(vote)


def _rerank(ctx: SelectionContext, candidates: list[SQLCandidate]) -> str | None:
    query = f"Question:\n{ctx.question}"
    if ctx.evidence:
        query += f"\n\nEvidence:\n{ctx.evidence}"
    hits = [{"id": c.source, "text": c.response.description} for c in candidates]
    try:
        ranked = rerank_hits(query, hits, **get_rerank_kwargs())
    except Exception:
        logger.exception("SQL description rerank failed.")
        return None
    scores = {hit.get("id"): hit.get("_rerank_score") for hit in ranked}
    for candidate in candidates:
        candidate.rerank_score = scores.get(candidate.source)
    return ranked[0].get("id") if ranked else None


def select_candidate(
    candidates: list[SQLCandidate], ctx: SelectionContext
) -> Selection:
    """Pick one candidate; see the module docstring for the stages."""
    by_source = {c.source: c for c in candidates}
    default = by_source.get(DEFAULT_SOURCE, candidates[0])
    if len(candidates) == 1:
        return Selection(default, "only_candidate", "the other candidate failed")

    for candidate in candidates:
        candidate.static_issue = static_issue(candidate.sql, ctx.dialect, ctx.target_db)
    clean = [c for c in candidates if not c.static_issue]
    if len(clean) == 1:
        loser = next(c for c in candidates if c.static_issue)
        return Selection(clean[0], "static", f"{loser.source}: {loser.static_issue}")
    pool = clean or candidates

    if ctx.connector is not None:
        _execute(pool, ctx)
        issues = {c.source: execution_issue(c.execution or {}) for c in pool}
        good = [c for c in pool if not issues[c.source]]
        if len(good) == 1:
            loser = next(c for c in pool if issues[c.source])
            return Selection(
                good[0], "execution", f"{loser.source}: {issues[loser.source]}"
            )
        if len(good) == 2 and result_key(good[0].execution["rows"]) == result_key(
            good[1].execution["rows"]
        ):
            winner = by_source.get(DEFAULT_SOURCE, good[0])
            return Selection(
                winner, "agreement", "both candidates return the same rows"
            )

    a, b = pool[0], pool[1]
    _judge(ctx, a, b)
    for candidate in (a, b):
        if len(candidate.judge_votes) == 2:
            return Selection(
                candidate,
                "judge",
                "the judge chose it with the candidates in both orders",
            )

    winner_source = _rerank(ctx, pool)
    if winner_source in by_source:
        return Selection(
            by_source[winner_source],
            "rerank",
            "the judge was split, so the description closest to the question won",
        )
    return Selection(default, "default", "no stage could decide")
