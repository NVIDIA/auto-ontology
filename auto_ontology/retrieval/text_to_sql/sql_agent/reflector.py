"""A second opinion on a finished query, and a minimal repair when it is wrong.

Why this exists when the protocol prompt already says most of it
----------------------------------------------------------------

Three things it says are said nowhere else, and the distinction is the whole
point of the module.

The protocol tells the *author* to check its own SELECT list, its own
conditions and its own formula. That instruction has been measured, and the
strongest version of it made things worse: an edit telling the agent to count
the question's outputs against its columns caused it to pull filter values and
sort keys *into* the SELECT on eight of the twelve questions it changed. The
author cannot audit itself, for the reason :data:`deep.ADJUDICATOR_PROMPT`
already gives about its own drafts -- it produced the query from a set of
assumptions, and re-reading it with those same assumptions is not a check.

The gate in :mod:`~.loop` does audit the finished query, but only
mechanically: static lint, probe findings, execution errors, empty results and
a handful of constructed variants. It never asks whether the SELECT list
matches what the question named. Its own docstring states the principle this
module extends -- "Offering a check the model may decline is not a check" --
after the model called the live-probe check twice in seventy questions. The
semantic checks were left as prose in the protocol, which is exactly the form
that principle says does not work.

So the check here is applied by a different reader, to every accepted answer,
and the repair is made by that reader rather than by asking the author to
think again.

What it looks for, and why it leans towards deletion
-----------------------------------------------------

Measured over the 336 questions of a full run where no candidate of ten was
ever correct, the agent's queries differ from gold by having *too much* far
more often than too little: a ``DISTINCT`` gold does not have 56 times against
30 missing, ``MAX`` 16 against 2, ``COUNT`` 14 against 3, an extra ``GROUP BY``
17 against 13. The same asymmetry shows in the conditions -- a question naming
one administrator was answered with a query checking three administrator
columns, which is a better question and a wrong answer.

The validator is therefore told that excess is the common failure, and the
categories it may report are weighted accordingly. It is also told to report
nothing far more often than something, because a reviewer that always finds
fault would churn the 80% of answers that are already right, and churn on a
correct answer is a loss.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from auto_ontology.retrieval.text_to_sql.sql_agent import config
from auto_ontology.utils.llm_invoke import invoke_with_structured_output

logger = logging.getLogger(__name__)

#: The only faults the validator may raise. A closed list because an open one
#: invites style notes, performance opinions and null-handling advice, none of
#: which change whether the answer is right and all of which cost a rewrite.
#: The first four are the measured failure modes, in descending frequency.
CATEGORIES = (
    "extra_distinct",
    "extra_condition",
    "extra_column",
    "wrong_source",
    "missing_column",
    "formula_mismatch",
)


VALIDATOR_PROMPT = """\
You are checking one finished SQL query against the question it answers. You
did not write it. The person who did cannot check it neutrally, because they
would re-read it with the assumptions that produced it, so you are reading it
instead.

Report a fault only when you can point at the words in the question that the
query contradicts. If you have to argue that the question probably meant
something other than it says, there is no fault. Most queries you see are
correct: reporting nothing is the common and correct outcome, and a rewrite
forced on a query that was already right is a loss, not a neutral act.

Certainty is a condition of reporting, not a quality of the report. If you
find yourself writing "borderline", "however", "may", "possibly", or arguing
both sides of the same element, you are not certain and there is no fault to
report. Each fault is one plain sentence naming one element and the single
edit it needs. A fault you have to justify at length is a fault you should
have left out.

By far the most common fault is doing more than was asked. In measurement, a
query carried a DISTINCT the question never called for almost twice as often
as it omitted one it needed, an unasked-for MAX eight times as often, an
unasked-for COUNT five times as often. A query that filters out rows that look
wrong, returns a column because it is helpful, or deduplicates because
duplicates look untidy is answering a better question, and a better question is
the wrong question. Weight your reading accordingly: look for excess first.

These are the only faults you may report.

  extra_distinct    A DISTINCT, GROUP BY or aggregate the question does not
                    call for. The question asks for rows, and the query
                    collapses them; or it asks "how many X" and the query
                    counts something other than X.
  extra_condition   A filter the question and the evidence never state. Being
                    defensible is not enough -- excluding district totals from
                    a question about schools is a fault if the question did
                    not ask for the exclusion.
  extra_column      The SELECT returns something the question did not name.
                    Count what the question asks to be returned and compare.
  wrong_source      A column read from the wrong table, where the same name or
                    the same idea exists in more than one table in the query.
  missing_column    The question names something to return that the SELECT
                    does not.
  formula_mismatch  The evidence defines a formula, a term or a literal, and
                    the query does not follow it exactly.

For each fault give the category and one concrete sentence naming the element
at fault and what should happen to it. Do not rewrite the query here.
"""


REFLECTOR_PROMPT = """\
You make the smallest possible edit to a SQL query that resolves the faults
you are given, and no other edit.

You are not reviewing the query and you are not improving it. Faults that are
not on the list stay in. Formatting, aliasing, table order, casing and style
stay exactly as they are. If a fault says a DISTINCT is unasked for, remove
that DISTINCT and change nothing else.

If the listed faults cannot be resolved by editing -- if resolving one would
require guessing at a column that may not exist, or would contradict another --
return the query exactly as given. An unchanged query is a valid answer and is
better than a speculative rewrite.

Return one read-only SQL statement, executable as written.
"""


class _Fault(BaseModel):
    category: str = Field(description="One of the listed category names.")
    detail: str = Field(description="One sentence: the element and what to do.")


class _Review(BaseModel):
    faults: list[_Fault] = Field(
        default_factory=list,
        description="Empty when the query answers the question as asked.",
    )


class _Repair(BaseModel):
    sql: str = Field(description="The edited query, or the original unchanged.")


def _context(question: str, evidence: str, sql: str, preview: Optional[dict]) -> str:
    parts = [f"Question:\n{question.strip()}"]
    if evidence and evidence.strip():
        parts.append(f"Evidence:\n{evidence.strip()}")
    parts.append(f"Query:\n{sql.strip()}")
    if preview and preview.get("ok"):
        columns = ", ".join(str(c) for c in (preview.get("columns") or []))
        rows = preview.get("rows") or []
        shown = "\n".join(str(row) for row in rows[:3])
        parts.append(
            f"It returns {preview.get('row_count', len(rows))} row(s) over "
            f"columns: {columns}\nFirst rows:\n{shown}"
        )
    return "\n\n".join(parts)


def review(
    llm: BaseChatModel,
    *,
    question: str,
    evidence: str,
    sql: str,
    preview: Optional[dict] = None,
) -> list[str]:
    """Faults in *sql* as a reader of the question would see them, or ``[]``.

    Unknown categories are dropped rather than passed on. The validator is
    asked for a closed list and mostly honours it, but an invented category is
    a fault nobody wrote a repair rule for, and letting it through would put
    style notes into the reflector's instructions.
    """
    if not sql.strip() or not question.strip():
        return []
    try:
        result = invoke_with_structured_output(
            llm,
            [
                SystemMessage(content=VALIDATOR_PROMPT),
                HumanMessage(content=_context(question, evidence, sql, preview)),
            ],
            _Review,
        )
    except Exception as exc:  # noqa: BLE001 — a review that fails is no review
        logger.warning("Semantic review failed: %s", exc)
        return []

    faults = []
    for fault in (getattr(result, "faults", None) or [])[: config.reflector_max_faults()]:
        category = (fault.category or "").strip().lower()
        detail = (fault.detail or "").strip()
        if category in CATEGORIES and detail:
            faults.append(f"{category}: {detail}")
        elif detail:
            logger.debug("Dropping review fault in unknown category %r", category)
    return faults


def repair(
    llm: BaseChatModel, *, sql: str, faults: list[str]
) -> str:
    """*sql* with the listed faults edited out, or *sql* unchanged."""
    if not faults:
        return sql
    listed = "\n".join(f"- {fault}" for fault in faults)
    try:
        result = invoke_with_structured_output(
            llm,
            [
                SystemMessage(content=REFLECTOR_PROMPT),
                HumanMessage(content=f"Query:\n{sql.strip()}\n\nFaults:\n{listed}"),
            ],
            _Repair,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Semantic repair failed: %s", exc)
        return sql
    edited = (getattr(result, "sql", "") or "").strip()
    return edited or sql


def reflect(
    llm: BaseChatModel,
    *,
    question: str,
    evidence: str,
    sql: str,
    toolbox: Any,
) -> tuple[str, dict[str, Any]]:
    """Review *sql* and return the repaired query, with a record of what happened.

    The repair is accepted only if it still runs and still returns rows where
    the original did. Both rails exist because the failure they guard against
    is silent: a repair that drops a condition too many returns an empty result
    and scores identically to a syntax error, and either would turn a query
    that was merely suspect into one that is certainly wrong.
    """
    record: dict[str, Any] = {"faults": [], "applied": False}
    preview = None
    try:
        preview = toolbox.preview_result(sql)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not preview before review: %s", exc)

    faults = review(llm, question=question, evidence=evidence, sql=sql, preview=preview)
    record["faults"] = faults
    if not faults:
        return sql, record

    edited = repair(llm, sql=sql, faults=faults)
    if " ".join(edited.split()).lower() == " ".join(sql.split()).lower():
        record["reason"] = "reflector returned the query unchanged"
        return sql, record

    try:
        after = toolbox.preview_result(edited)
    except Exception as exc:  # noqa: BLE001
        record["reason"] = f"repaired query could not be previewed: {exc}"
        return sql, record

    if not after.get("ok"):
        record["reason"] = f"repaired query does not run: {after.get('error')}"
        return sql, record
    if (
        preview
        and preview.get("ok")
        and preview.get("row_count")
        and not after.get("row_count")
    ):
        record["reason"] = "repaired query returns no rows where the original did"
        return sql, record

    record["applied"] = True
    record["before"] = sql
    return edited, record


__all__ = [
    "CATEGORIES",
    "REFLECTOR_PROMPT",
    "VALIDATOR_PROMPT",
    "reflect",
    "repair",
    "review",
]
