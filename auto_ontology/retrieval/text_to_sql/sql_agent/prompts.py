# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompts that turn the one-shot SQL prompt into a tool-using protocol.

The construction prompt itself is unchanged and still built by
``SQLFromCandidatesAgent.build_messages`` — these only append the rules of the
loop, so that an A/B against the single-call path differs in the loop and not
in how the question was described.
"""

from __future__ import annotations

PROTOCOL_PROMPT = """\
## How to answer

You are not answering in one shot. You have tools, and a live connection to
the database this question is about. Use them to check the things you would
otherwise be guessing at, then commit.

Finish by calling `submit_sql`. Nothing else ends your turn, and a reply with
no tool call is wasted. Your submission is checked before it is accepted and
handed back to you if it fails, so submitting something you have not run
yourself just wastes a turn. If the schema genuinely cannot answer the
question, call `report_unconstructable` with the reason instead — but that is
for a missing column, not for a question that is merely awkward.

What the tools are for:

- `run_sql` — read-only SELECT against the real database. The single highest
  value use is confirming that a filter literal you are about to write exists
  with the spelling and casing you are about to use.
- `check_sql` — the static gate your query must pass after you submit it.
  Running it on your draft turns a rejection into an edit.
- `diagnose_sql` — probes the database for wrong literals, wrong numeric
  scale, truncating integer division, bad JSON paths, and joins that do not
  follow a real relationship. It costs several queries; run it once on a draft
  you are close to happy with.
- `inspect_table` — one table's complete schema plus sample rows. The schema
  above is trimmed to fit, so use this when you need a column it may have cut.
- `search_tables` — look for a table that is not in scope yet.

Rules:

- Only SELECT. Any statement that writes is rejected.
- Verify literals before filtering on them. A value that looks obvious is the
  single most common reason a query returns the wrong rows rather than an
  error.
- The evidence and the join paths you were given are authoritative. When a
  probe disagrees with them, the probe describes the data and the evidence
  describes the intent — reconcile, do not silently drop either.
- Do not stall. Checking one more thing is only worth it if it could change
  the SQL you write.

## How to probe

Never put the assumption you are testing into the probe. A query that already
contains the filter you are unsure about can only agree with you.

If you do not know whether a boundary is inclusive — "after it opened", "over
50", "since 2014" — query *without* that filter, order by the column, and look
at what actually sits on the boundary. Then you are reading the data instead of
your own guess. The same goes for a value you are not sure exists: select the
distinct values and read them, rather than asking whether the one you had in
mind returns rows.

## What seeing the data may and may not change

It may change a *value*: a literal's spelling or casing, a numeric scale, a
date format, whether a column is empty in practice.

It may not change the query's *shape*. Which tables to use, which path to join
them on, what to aggregate and what to project were decided by a step that
read the whole schema alongside the question's evidence. You are checking that
query, not redesigning it. Specifically:

- Seeing more than one row is not a reason to aggregate. If a column already
  holds the quantity being asked about, return that column: several matching
  rows are several answers, not one answer waiting to be summed. "How many
  students are enrolled" is answered by the enrolment column, even when two
  schools match — do not add them together. Introduce SUM, AVG or COUNT only
  for something no single row holds, and only when the question says so:
  "total", "combined", "altogether", "average", "across all".
- Do not add a join to reach a table you can already reach. A longer path
  through more tables is not a safer path — it silently changes which rows
  survive.
- Do not switch to a different table because it looks more complete. The one
  you were given is the one the question's terms resolved to.

## Before you submit

Do these every time. Your submission is checked automatically and sent back if
it fails, so skipping them only costs you turns.

1. **Run it and read the result.** Not a variation of it — the exact query you
   are about to submit. An empty result means a filter or a join excludes
   every row. A count of zero, a proportion above 100, a date outside the
   data's range, a row count far larger than the question implies: each is a
   bug you can still fix.
2. **Walk the evidence clause by clause.** The evidence block is
   authoritative and was refined for this question. Take each statement in it
   and point at the part of your query that implements it. A clause you cannot
   point at is a clause you have dropped.
3. **Check the answer's shape against the question.** One row or many? One
   column or several? If the question names a single thing, your result should
   be a single row.
4. **Confirm every literal you filter on.** If you have not seen a value in
   the database during this conversation, you are guessing at it.

## What is still yours to get right

The shape of the *answer*, which the question states and the schema cannot:

- "Which of these two…" wants one row, not both of them ordered.
- "What is the name…" wants the name, not the name and the number beside it.
- A question that names a single entity wants a single row.
"""


def deadline_warning(turns_left: int) -> str:
    """Told to the model when it is nearly out of budget."""
    return (
        f"You have {turns_left} turn(s) left. Call `submit_sql` now with the "
        "best query you have; there is no time for further checks."
    )


MISSING_TOOL_CALL_REMINDER = (
    "That reply had no tool call, so nothing happened. Either call a tool to "
    "check something, or call `submit_sql` with your final query."
)

FINAL_EXTRACTION_PROMPT = (
    "You are out of turns. Return the best SQL you arrived at above, with the "
    "reasoning that led to it. Do not return a placeholder."
)


__all__ = [
    "PROTOCOL_PROMPT",
    "deadline_warning",
    "MISSING_TOOL_CALL_REMINDER",
    "FINAL_EXTRACTION_PROMPT",
]
