# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from auto_ontology.utils.llm_invoke import safe_invoke_text

logger = logging.getLogger(__name__)

_COMPLETENESS_PROMPT = """\
You are managing a list of formulas and conditions that are still incompletely specified \
for SQL generation purposes.

Working question (for context on what SQL operations are needed):
{working_question}

Previously flagged gaps (may be empty):
{prior_gaps}

User's answer to the last clarification question:
Q: {last_q}
A: {last_a}

Relevant external knowledge (may contain partial or ambiguous definitions):
{relevant_knowledge}

Schema columns already resolved by VDB (drop a gap only if the description directly resolves it):
{resolved_schema_terms}

Your task — produce a FRESH updated list:
1. DROP any prior gap that is now fully resolved by the user's answer above \
   (exact operators and constants given for every part of it, all column names considered resolved).\
2. UPDATE a prior gap if the user's answer, relevant knowledge, or the working question provides \
    information which requires a change in the gap's description (e.g, narrowing the gap in light \
    of additional information).
2. KEEP any prior gap that the answer did not fully resolve and does not require an update.
4. ADD any new gap introduced by the answer or the external knowledge that would \
   prevent writing correct SQL — for example:
   - A business label, population, or cutoff the question depends on that is not \
     stated in the user's answer or the external knowledge
   - A threshold or classification condition expressed as a choice between meanings, \
     where the knowledge does not say which meaning applies
   - Which event orders a sequence when two different times would change which \
     rows qualify, and neither the question nor the answer decides it
   - Whether the result is one row per reading or one row per entity, when that \
     choice changes the grain and neither the question nor the answer decides it
   - A KB condition or user answer using hedged language (e.g. "typically", "often",\
    "approximately", "things like") that leaves which cases qualify undecided
 
Do NOT flag:
- A missing formula, equation, operator, constant, or column binding. Those are \
  taken from external knowledge and the schema at SQL generation time. Do not ask \
  the user for them, even when the formula names concepts rather than schema columns.
- Business context or motivation that does not affect which rows or values qualify
- Terms fully defined by an explicit formula in the answer or KB
- Minor stylistic ambiguities a SQL generator can resolve on its own (e.g. sign \
  handling, boundary operators)

Output one line per remaining gap, ordered from most to least critical for SQL correctness, \
by considering the working question (e.g. a missing core formula blocks SQL entirely; a missing \
secondary threshold is lower priority):
  INCOMPLETE: <term> | <what is still missing>

If no gaps remain, output:
  COMPLETE"""

_COMPLETENESS_PROMPT_KB_ONLY = """\
You are scanning for formulas or conditions that are missing or ambiguously defined \
and would prevent writing correct SQL.

Working question (for context on what SQL operations are needed):
{working_question}

Relevant external knowledge:
{relevant_knowledge}

Schema entities found in the database but with NO formula in the external knowledge base \
(the question may or may not require computing a formula for these — check each one):
{vdb_only_section}

Schema columns already resolved by VDB (drop a gap only if the description directly resolves it):
{resolved_schema_terms}

Identify gaps in either of the following categories:
1. A business label, population, or cutoff the working question depends on that \
   external knowledge does not already decide:
   - Expressed with hedged language ("typically", "often", "approximately", "things like") \
     so which cases qualify is undecided
   - Two documented meanings and no indication which one applies
   - Two events that could order a sequence, when the choice changes which rows qualify
   - One row per reading versus one row per entity, when the choice changes the grain
2. Do NOT flag a schema entity just because its formula is not written next to a \
   column name. A formula stated anywhere in external knowledge is complete. \
   Standard aggregations (SUM, AVG, COUNT) are not gaps.

A gap is only significant if the user must choose a meaning, population, cutoff, \
ordering event, or row grain that the knowledge does not decide.

Do NOT flag:
- A missing formula, equation, operator, or column name. Column binding is not a \
  question for the user.
- Terms fully defined with an explicit formula in external knowledge, even when \
  that formula uses business names rather than schema column names.
- Terms already resolved to a specific column above (schema columns already resolved by VDB).
- Business context that does not affect which rows or values qualify
- Minor ambiguities a SQL generator can resolve on its own

Output one line per gap found, ordered from most to least critical:
  INCOMPLETE: <term> | <what is still missing>

If everything is clearly defined for SQL generation, output:
  COMPLETE"""


def _parse_gaps(
    response: str, current_gaps: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    if response.upper().startswith("COMPLETE"):
        return []
    gaps: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in response.splitlines():
        if line.upper().startswith("INCOMPLETE:"):
            body = line[len("INCOMPLETE:") :].strip()
            if "|" in body:
                term, missing = body.split("|", 1)
                term = term.strip()
            else:
                term, missing = body.strip(), "exact specification missing"
            if term not in seen:
                seen.add(term)
                gaps.append((term, missing.strip()))
    if not gaps:
        # Response was neither "COMPLETE" nor any recognizable "INCOMPLETE:"
        # line — malformed LLM output, not a genuine completeness verdict.
        # Preserve whatever was already tracked rather than silently wiping it.
        logger.warning(
            "Completeness — unparseable response, keeping prior gaps: %s",
            response[:200],
        )
        return list(current_gaps)
    return gaps


def detect_incomplete_formulas(
    working_question: str,
    last_turn: dict | None,
    relevant_knowledge: str,
    current_gaps: list[tuple[str, str]],
    llm,
    vdb_only_entities: list[str] | None = None,
    resolved_schema_terms: str = "",
) -> list[tuple[str, str]]:
    """Return an updated (term, what_is_missing) gap list.

    When last_turn is None (turn 0): scans the KB and any VDB-only entities for
    missing formulas — no user answer to evaluate. resolved_schema_terms is also
    used here to drop gaps for terms already resolved by the same VDB pass.
    When last_turn is provided: evaluates the answer against prior gaps and KB,
    dropping resolved terms, keeping unresolved ones, adding new gaps.
    resolved_schema_terms: pre-formatted VDB hit descriptions (entity → description)
    used to close gaps whose column semantics are already known from the schema.
    """
    if last_turn is None:
        vdb_section = (
            "\n".join(f"- {e}" for e in vdb_only_entities)
            if vdb_only_entities
            else "None"
        )
        prompt = _COMPLETENESS_PROMPT_KB_ONLY.format(
            working_question=working_question,
            relevant_knowledge=relevant_knowledge or "None",
            vdb_only_section=vdb_section,
            resolved_schema_terms=resolved_schema_terms or "None",
        )
    else:
        prior_text = (
            "\n".join(f"- {t}: {m}" for t, m in current_gaps)
            if current_gaps
            else "None"
        )
        prompt = _COMPLETENESS_PROMPT.format(
            working_question=working_question,
            prior_gaps=prior_text,
            last_q=last_turn["q"],
            last_a=last_turn["a"],
            relevant_knowledge=relevant_knowledge or "None",
            resolved_schema_terms=resolved_schema_terms or "None",
        )

    response = safe_invoke_text(llm, prompt).strip()
    logger.debug("Completeness — raw response: %s", response[:600])
    gaps = _parse_gaps(response, current_gaps)
    logger.info("Completeness — gaps found: %s", gaps)
    return gaps
