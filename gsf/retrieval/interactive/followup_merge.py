from __future__ import annotations

import logging

from gsf.utils.llm_invoke import safe_invoke_text_nr

logger = logging.getLogger(__name__)

_FOLLOW_UP_MERGE_PROMPT = """\
You are rewriting a follow-up database question into a clear, self-contained question \
for a SQL generator.

Follow-up question:
{p2_question}

Previous question (Phase 1):
{p1_question}
Previous SQL:
{p1_sql}


Task: Rewrite the follow-up question into a single, complete, standalone question \
for the SQL generator. Resolve any underspecified terms, references, or concepts \
in the follow-up using the previous question and SQL — pull in the exact column names, \
table names, formulas, thresholds, and conditions that the follow-up depends on. \
Include as much or as little of the previous SQL's structure as the follow-up requires.

Rules:
- Resolve references to prior concepts (e.g. "that category", "the same score", "those \
signals") using the previous SQL and question. Make sure to use the context of both the \
previous and follow up questions to determine wether a concept in the follow up is actually a reference.
- Carry forward table names, column names, formulas, tresholds and conditions that the follow-up \
references or implicitly depends on, unless requested otherwise by the follow up. Carry forward\
exact numeric values, if they exist.
- If the follow-up reuses or extends the previous query's full structure, incorporate it. \
If it only borrows part of it, incorporate only that part.
- For any concept or metric in the follow-up that does not clearly map 1:1 to a term \
in the previous SQL, do NOT assign it a table or column, or new name — leave it unresolved so \
the SQL generator can discover it from the schema. Only carry forward table/column \
assignments for concepts explicitly present in the previous SQL.
- if the question indicates only a minor change to the question (e.g a short sentence \
starting with "also"), closely preserve the previous question structure.
- If the follow-up explicitly signals that part of the previous SQL's structure should \
change, narrow, or drop this time (e.g. a different formula, a restricted table/join scope,\
an output change), do not carry forward that part.
"""


def merge_follow_up_question(
    p1_question: str,
    p1_sql: str,
    p2_question: str,
) -> str:
    """Rewrite a raw follow-up question into a self-contained question with column
    names and conditions drawn from the Phase 1 SQL."""
    prompt = _FOLLOW_UP_MERGE_PROMPT.format(
        p1_question=p1_question,
        p1_sql=p1_sql[:800],
        p2_question=p2_question,
    )
    merged = safe_invoke_text_nr(prompt).strip()
    if not merged:
        logger.warning(
            "Follow-up merge returned empty; falling back to raw follow-up question"
        )
        return p2_question
    logger.info("Follow-up merged question: %s", merged)
    return merged
