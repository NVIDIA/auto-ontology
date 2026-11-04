# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diagnose a wrong-result debug turn before SQL reconstruction.

BIRD-Interact's grader reports only that the result rows did not match. The
reconstruction prompt then sees a generic checklist and often emits the same
SQL. This module names one concrete mismatch from the question, the rejected
SQL, and the rows that SQL returned, and writes that into ``path_state["error"]``
so ``reconstruct_sql`` repairs the named part.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .state import InteractiveSessionState

logger = logging.getLogger(__name__)

_WRONG_RESULT_MARKER = "The SQL produced incorrect results."

# Dropped once a diagnosis says the mismatch is semantic. Left in place when
# the rows already answer the question, because the debug turn then retries
# numeric type and tie order without changing the values.
_PRESERVE_VALUES_NOTE = (
    " If your existing query's filters, grain, projections, "
    "and ordering already satisfy every requirement above, preserve the exact "
    "same logical result rows and order (a wrong-result verdict can be caused "
    "solely by SQL numeric type differences, which are retried automatically "
    "when the result values stay unchanged)."
)

_WRONG_RESULT_HINT = (
    "The SQL produced incorrect results. "
    "Do NOT modify formula coefficients, formula structure or column and field names."
    "Address whichever of the following applies, or fix a different issue you identify:\n\n"
    "1. COMPOUND FILTER COMPLETENESS: Re-check the evidence and question for "
    "every condition they imply, not just the primary one. Common drops: an "
    "entity/category/type filter stated in the original question (e.g. a specific "
    "visa class, organ, facility, or group), an IS NOT NULL guard on a column "
    "or JSONB field feeding a computed metric, aggregate, or ratio (so rows with "
    "NULL metrics do not appear or float to the top of DESC sorts), a "
    "positivity/non-zero guard on a denominator, a second threshold in a "
    "compound AND/OR, an exact value or unit filter (e.g. status = 'Completed', "
    "unit = 'year'). If the evidence lists multiple conditions, verify all of "
    "them are still present in your SQL.\n\n"
    "2. OUTPUT GRAIN AND ROW SELECTION: Check whether the question wants one "
    'row per group or a single aggregate/top-1 row, and whether "most '
    'recent"/"latest snapshot"/dedup logic is implied — an extra join or an '
    "extra GROUP BY column can silently fan out or split a group that should "
    'stay merged. If the question asks for top-N or "the one with the highest '
    'X," add LIMIT N, but NEVER add LIMIT 1 to a multi-row or per-entity query. '
    "Judge from the question's phrasing whether unmatched "
    "primary-entity rows should still appear.\n\n"
    "3. ORDER BY, PROJECTION, AND SCALE: Keep ORDER BY <primary_metric> DESC "
    "(including for cost, fee, time, and count metrics) unless ascending order "
    "is explicitly requested; do NOT drop a valid ORDER BY or add unrequested "
    "secondary tie-breaker columns. Project only the requested columns in the "
    "EXACT left-to-right order stated in the question and clarifications (when "
    "a breakdown asks for counts and percentages across categories, group all "
    "counts together followed by all percentages unless interleaving is "
    "explicitly requested; for 'basic info' or 'details', select only the "
    "entity's ID and name/title — never dump all table columns or add an "
    "unrequested status label column; when profiling or auditing a single "
    "specific entity ID, include that entity ID column first if requested). "
    "If the knowledge base defines a percentage/proportion formula with `* 100`, "
    "preserve `* 100`." + _PRESERVE_VALUES_NOTE
)

_MAX_SAMPLE_ROWS = 5
_MAX_NULL_SCAN_ROWS = 200
_MAX_CELL_CHARS = 60
_MAX_SAMPLE_CHARS = 1800
_MAX_QUESTION_CHARS = 2000
_MAX_EVIDENCE_CHARS = 4000
_MAX_SQL_CHARS = 4000
_MAX_DIAGNOSIS_CHARS = 700

_TYPE_OR_ORDER_DEFAULT = (
    "The SQL already matches the question; only a numeric type or tie order "
    "is likely to differ."
)


def wrong_result_hint() -> str:
    """Checklist seeded when a submit ran and the grader rejected the rows."""
    return _WRONG_RESULT_HINT


def format_result_sample(sql_response: Any) -> str:
    """Short, prompt-sized description of the rows a rejected query returned."""
    rows = _coerce_rows(sql_response)
    if rows is None:
        return ""
    if not rows:
        return "0 rows."

    columns: list[str] = []
    seen: set[str] = set()
    scanned = rows[:_MAX_NULL_SCAN_ROWS]
    for row in scanned:
        for key in row:
            name = str(key)
            if name not in seen:
                seen.add(name)
                columns.append(name)

    null_notes: list[str] = []
    for column in columns:
        missing = sum(1 for row in scanned if row.get(column) is None)
        if missing:
            null_notes.append(f"{column} null {missing}/{len(scanned)}")

    header = f"{len(rows)} rows; columns: {', '.join(columns)}"
    if len(rows) > len(scanned):
        header += f" (nulls counted on the first {len(scanned)})"
    if null_notes:
        header += "; " + ", ".join(null_notes)

    shown = min(_MAX_SAMPLE_ROWS, len(rows))
    lines = [header, f"First {shown} rows:"]
    for row in rows[:_MAX_SAMPLE_ROWS]:
        cells = [f"{column}={_format_cell(row.get(column))}" for column in columns]
        lines.append("- " + ", ".join(cells))
    text = "\n".join(lines)
    if len(text) <= _MAX_SAMPLE_CHARS:
        return text
    return text[:_MAX_SAMPLE_CHARS].rstrip() + "…"


def parse_result_feedback(text: str) -> tuple[str, str]:
    """Split a diagnosis reply into ``(MISMATCH|TYPE_OR_ORDER, diagnosis)``.

    An unlabelled reply is treated as a mismatch. An empty reply is ``("", "")``.
    """
    verdict = ""
    diagnosis_parts: list[str] = []
    for raw_line in (text or "").splitlines():
        # Models often wrap the labels in markdown emphasis.
        line = raw_line.strip().replace("*", "").strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("VERDICT:"):
            verdict = line.split(":", 1)[1].strip().upper()
            continue
        if upper.startswith("DIAGNOSIS:"):
            diagnosis_parts.append(line.split(":", 1)[1].strip())
            continue
        if diagnosis_parts:
            diagnosis_parts.append(line)

    diagnosis = _clip(" ".join(diagnosis_parts), _MAX_DIAGNOSIS_CHARS)
    if verdict.startswith("TYPE_OR_ORDER"):
        return "TYPE_OR_ORDER", diagnosis
    if verdict.startswith("MISMATCH"):
        return "MISMATCH", diagnosis

    body = _clip(" ".join((text or "").split()), _MAX_DIAGNOSIS_CHARS)
    if not body:
        return "", ""
    return "MISMATCH", body


def diagnose_result_mismatch(
    *,
    question: str,
    evidence: str,
    sql: str,
    result_sample: str,
    numeric_kinds: str,
) -> str:
    """Ask the reasoning model which part of *sql* the result rows contradict."""
    from auto_ontology.utils.llm_invoke import get_llm_client, safe_invoke_text

    prompt = _diagnosis_prompt(
        question=question,
        evidence=evidence,
        sql=sql,
        result_sample=result_sample,
        numeric_kinds=numeric_kinds,
    )
    return safe_invoke_text(get_llm_client(), prompt).strip()


def attach_result_mismatch_diagnosis(
    session: InteractiveSessionState, evidence: str
) -> None:
    """Replace the generic wrong-result checklist's lead with a real diagnosis.

    No-op unless this call is the debug resume seeded for a result mismatch.
    A failed model call leaves the checklist in place.
    """
    error = session.path_state.get("error") or ""
    if _WRONG_RESULT_MARKER not in error:
        return
    if session.path_state.get("_resume_from") != "reconstruct_sql":
        return

    sql = _rejected_sql(session)
    if not sql:
        logger.info(
            "[%s] Result mismatch diagnosis skipped: no rejected SQL",
            session.task_id,
        )
        return

    sample = format_result_sample(session.path_state.get("sql_response_from_db"))
    try:
        raw = diagnose_result_mismatch(
            question=session.working_question,
            evidence=evidence or "",
            sql=sql,
            result_sample=sample,
            numeric_kinds=_kinds_text(session, sql),
        )
    except Exception:
        logger.warning(
            "[%s] Result mismatch diagnosis failed; keeping the checklist",
            session.task_id,
            exc_info=True,
        )
        return

    verdict, diagnosis = parse_result_feedback(raw)
    if verdict == "TYPE_OR_ORDER":
        diagnosis = diagnosis or _TYPE_OR_ORDER_DEFAULT
        session.path_state["error"] = (
            "RESULT MISMATCH DIAGNOSIS: "
            + diagnosis
            + "\n\nThe executed result is consistent with the question. Keep "
            "the same filters, joins, grain, and projected values. A numeric "
            "type or tie-order difference is retried separately when those "
            "values stay unchanged.\n\n" + error
        )
    elif verdict == "MISMATCH" and diagnosis:
        sample_block = sample or (
            "Result rows from the rejected SQL were not retained."
        )
        session.path_state["error"] = (
            "RESULT MISMATCH DIAGNOSIS: "
            + diagnosis
            + "\n\nResult the rejected SQL returned:\n"
            + sample_block
            + "\n\nAn external grader executed this SQL and rejected the "
            "result rows. The reference answer is not available. Change the "
            "part named in the diagnosis. Do not resubmit this SQL unchanged."
            "\n\n" + error.replace(_PRESERVE_VALUES_NOTE, "").rstrip()
        )
    else:
        logger.info(
            "[%s] Result mismatch diagnosis was empty",
            session.task_id,
        )
        return

    # The diagnosis already decided this is a repair of the current SQL, so
    # reconstruction must not spend its one analysis call hunting for tables.
    session.path_state["error_known_fixable"] = True
    logger.info(
        "[%s] Result mismatch diagnosis (%s): %s",
        session.task_id,
        verdict,
        diagnosis[:200],
    )


def _diagnosis_prompt(
    *,
    question: str,
    evidence: str,
    sql: str,
    result_sample: str,
    numeric_kinds: str,
) -> str:
    sample = result_sample.strip() or "(result rows were not retained)"
    evidence_text = _clip(evidence, _MAX_EVIDENCE_CHARS) or "(none)"
    return (
        "You are diagnosing why a SQL query's result was rejected.\n\n"
        "An external grader executed the SQL and compared the result rows to "
        "a hidden reference. It reported only that the result is wrong. You "
        "cannot see the reference rows. Do not invent them.\n\n"
        "Question:\n"
        + _clip(question, _MAX_QUESTION_CHARS)
        + "\n\nEvidence and clarifications:\n"
        + evidence_text
        + "\n\nRejected SQL:\n"
        + _clip(sql, _MAX_SQL_CHARS)
        + "\n\nResult this SQL returned on our database:\n"
        + sample
        + "\n\nNumeric types of the output columns, when recorded: "
        + (numeric_kinds or "unknown")
        + "\n\nThe grader rounds numbers and then compares them exactly, so "
        "a float and a numeric with the same digits are different answers. "
        "It also compares row order when the task requires an order.\n\n"
        "Decide one verdict:\n"
        "- TYPE_OR_ORDER: the SQL already answers the question. Filters, "
        "grain, joins, and projected values look right. The mismatch is only "
        "a numeric type or the order of tied rows.\n"
        "- MISMATCH: name the single most likely semantic error.\n\n"
        "Reply in exactly this shape:\n"
        "VERDICT: MISMATCH\n"
        "or\n"
        "VERDICT: TYPE_OR_ORDER\n\n"
        "DIAGNOSIS: one or two sentences naming the clause to change and why "
        "the returned rows show that. Do not write SQL."
    )


def _rejected_sql(session: InteractiveSessionState) -> str:
    sql = str(session.path_state.get("sql_code") or "").strip()
    if sql:
        return sql
    generated = session.path_state.get("sql_generation_result")
    return str(getattr(generated, "sql_code", "") or "").strip()


def _kinds_text(session: InteractiveSessionState, sql: str) -> str:
    recorded_for = str(session.path_state.get("result_numeric_kinds_sql") or "")
    kinds = session.path_state.get("result_numeric_kinds")
    if not kinds or recorded_for.strip() != sql.strip():
        return "unknown"
    return ", ".join(str(kind) for kind in kinds)


def _coerce_rows(sql_response: Any) -> list[dict] | None:
    """Rows from the execution agent's stored JSON, or None when unreadable."""
    payload = sql_response
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], str):
        payload = payload[0]
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return None
    if payload is None:
        return None
    if isinstance(payload, list) and not payload:
        return []
    if isinstance(payload, list) and isinstance(payload[0], dict):
        return [row for row in payload if isinstance(row, dict)]
    return None


def _format_cell(value: Any) -> str:
    if value is None:
        return "NULL"
    text = str(value).replace("\n", " ")
    if len(text) <= _MAX_CELL_CHARS:
        return text
    return text[:_MAX_CELL_CHARS].rstrip() + "…"


def _clip(text: str, limit: int) -> str:
    collapsed = (text or "").strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rstrip() + "…"
