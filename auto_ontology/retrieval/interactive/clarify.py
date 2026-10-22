# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clarification-turn orchestration.

Decides whether to ask the user a clarification question before writing SQL,
using entity resolution (entity_resolution.py) and external-knowledge coverage
(kb_coverage.py) as inputs. Those two modules own "what does this term resolve
to"; this module owns "given what's resolved, should we ask, and what".
"""

from __future__ import annotations

import logging
import re
from typing import Optional, TYPE_CHECKING

from auto_ontology.utils.llm_invoke import safe_invoke_text

from .entity_resolution import (
    _find_unresolvable_entities,
    _normalize_entity,
    _FILLER,
    _CONNECTIVES,
    _KB_VDB_DISAMBIG_THRESHOLD,
)
from .kb_coverage import _parse_kb_entries, _compact_schema

if TYPE_CHECKING:
    from .state import InteractiveSessionState

logger = logging.getLogger(__name__)

# Regex for detecting calculation-context queries.
# Covers: calculate/calculated/calculation, compute/computed/computation, derive/derived/derivation.
_CALC_TRIGGER = re.compile(r"\b(calculat|comput|deriv)", re.IGNORECASE)

# Shared tokenizer for the "was this term already asked about" check below.
# Splits on any run of non-word characters (whitespace, slashes, hyphens, etc.),
# so "X-to-Y join" and "X to Y join" tokenize identically. Used on both the
# incomplete-formula term and past clarify-history question text — they must
# share one normalization pipeline, or terms containing punctuation the two
# sides split differently on can never register as "already asked".
_TOKEN_SPLIT = re.compile(r"[^\w]+")

_STUCK_PHRASES = frozenset(
    [
        "out of scope",
        "not certain",
        "uncertain",
        "cannot answer",
        "can't answer",
        "unable to answer",
        "not able to answer",
        # The user left the computation unspecified. That is a refusal, not a
        # formula: do not merge it into the question and do not ask it again.
        "leave the specifics",
        "leave that to you",
        "best judgment",
        "don't have the specific",
        "do not have the specific",
    ]
)

# The user simulator (and the product prompt) will not answer these. A question
# that asks for one of them burns a clarify turn and comes back unanswered.
_UNANSWERABLE_MECHANICS = re.compile(
    r"\b(formulas?|equations?|columns?|tables?|schemas?|sql)\b", re.IGNORECASE
)
# Gap descriptions often mention a column while the real choice is a business
# cutoff ("no mapping to a column value"). Only a formula/SQL gap is skipped
# before a question is generated; the question itself is still screened.
_FORMULA_GAP = re.compile(r"\b(formulas?|equations?|sql)\b", re.IGNORECASE)

_EMPTY_EXTERNAL_KB = frozenset({"", "[]", "none", "null"})


def external_kb_text(external_kb: str) -> str:
    """Return external knowledge text, or "" when the session has none."""
    text = (external_kb or "").strip()
    if text.lower() in _EMPTY_EXTERNAL_KB:
        return ""
    return text


def knowledge_for_user_decisions(grounded: str, external_kb: str) -> str:
    """Grounded slice plus the full external knowledge, for completeness checks.

    Coverage still decides which entries are about this question. The full
    text is appended so a formula coverage did not attach is not reported as
    missing and then asked of the user.
    """
    parts: list[str] = []
    grounded_text = (grounded or "").strip()
    if grounded_text:
        parts.append(grounded_text)
    kb = external_kb_text(external_kb)
    if kb:
        parts.append(
            "Full external knowledge (already known; do not treat a definition "
            "or formula that appears here as missing):\n" + kb
        )
    return "\n\n".join(parts)


def asks_unanswerable_mechanics(text: str) -> bool:
    """True when text asks for a formula, column, table, or SQL structure."""
    return bool(_UNANSWERABLE_MECHANICS.search(text or ""))


_CLARIFY_PROMPT = """\
You are deciding whether to ask the user a clarification question before writing SQL.

Database schema:
{db_schema}

Relevant external knowledge:
{relevant_knowledge}

Full external knowledge (authoritative; never ask the user to repeat a definition or formula from here):
{external_knowledge}

Schema mappings already resolved (use these directly — do NOT ask where these terms are stored):
{resolved_schema_terms}

User question: {question}

Prior clarifications (Q&A):
{history}

Topics already asked about that went UNANSWERED:
{unanswered_topics}

Terms not found in the database schema or external knowledge (ask the user to define these):
{unresolvable_terms}

Business choices still open, ranked most-critical first. Ask when one of these is a meaning, a population, a cutoff, which event orders a sequence, or whether the result is one row per reading or one row per entity. Do not ask the user to restate a formula that external knowledge already contains, and do not ask them to name a column or table:
{incomplete_formulas_note}

{sort_direction_note}

STRICT RULES — follow every one of these exactly:
1. NEVER ask where data is stored, and never ask the user to name a column, table, or SQL structure. If external knowledge already states a formula, use that text — do not ask the user to restate it. BAD: "Which column stores quality X?" BAD: "What is the exact formula for quality X?" GOOD: "When you say health crisis, which documented risk condition do you mean?" GOOD: "Should consecutive matches be ordered by when the match was created, or by the match time?" GOOD: "Should each reading be its own row, or one row per robot?"
2. Ask about a business choice the schema, external knowledge, and history do not already decide: which of two documented meanings applies, which population is in scope, a cutoff the knowledge does not state, which event orders a sequence, or whether one reading is one row. If the formula is not in external knowledge, ask what the metric means. The user will often state the calculation in that answer. Do not phrase the question as a request for a formula, a column, or SQL.
3. If the question is vague about what to output (e.g., "show relevant metrics", "summarize your findings", "show the top results"), prioritize asking the user which specific metrics or fields they want in the output. Ask for the business names, not for columns or formulas.
4. Do not ask the exact same question about a topic the user could not answer (listed under "Topics already asked about that went UNANSWERED"). Do not rephrase an unanswered question into a request for a formula, a column, or SQL. You MAY ask a follow-up about a business choice the user already started to answer, such as which of two meanings they confirmed or which event orders the rows.
5. If there are potentially unresolvable terms which do not have satisfactory definitions in the prior clarifications, relevant knowledge, full external knowledge, or db_schema, ask what the business term refers to, one term at a time. Do not ask how to calculate it when external knowledge already states the formula.
6. Suggested format: "What do you mean by [TERM]?" Name the exact term. For ordering or grain, name the two business alternatives. Do not ask where a value is stored.
7. Pick the column yourself when the candidates mean the same thing. If two candidates are different events or different grains, and the choice changes which rows qualify, ask which one the user means.
8. If anything else in the user's question seems unclear, you may ask about it — for example an ambiguous grouping term, a threshold the knowledge does not state, an unspecified limit, which event orders a sequence, one row per reading versus one row per entity, or a preferred output format.
9. Output a single focused question only — never two questions joined with "and" or "or".
10. Output PROCEED when the meaning, the population, the ordering event, and the row grain are decided by the question, the external knowledge, or the clarifications. A formula that external knowledge already states is not a reason to ask. An undecided order or an undecided row grain is a reason to ASK. If "Business choices still open" lists only a formula that external knowledge already states, output PROCEED.

Output PROCEED or ASK: <question>:"""

_FORCED_QUESTION_PROMPT = """\
The following output specification is required for a database query but cannot be resolved \
from the schema or external knowledge:

Term: {term}
Why it's needed: {description}

Generate a single, focused question to ask the user to clarify this specification.
Ask which business meaning, population, cutoff, ordering event, or row grain they intend.
Name the two business alternatives when the choice is an order or a grain.
Do not ask for a formula, equation, column, table, or SQL structure.
If the only unresolved piece is a formula or where a value is stored, output exactly SKIP.
Output only the question text, or SKIP, nothing else.\
"""


_SORT_DIRECTION = re.compile(
    r"\b(asc\b|desc\b|ascending|descending|"
    # bare high/low etc. only count when followed by "first" — otherwise they're adjectives
    r"high(est)?\s+first|low(est)?\s+first|"
    r"larg(est)?\s+first|small(est)?\s+first|"
    r"best\s+first|worst\s+first|"
    r"increasing order|decreasing order|"
    r"top \d|bottom \d|"
    r"from (high(est)?|low(est)?|larg(est)?|small(est)?|best|worst)|"
    r"(high(est)?|low(est)?|larg(est)?|small(est)?|worst|best)\s+to\s+"
    r"(high(est)?|low(est)?|small(est)?|larg(est)?|worst|best))\b",
    re.IGNORECASE,
)

_DEFAULT_SORT_HINT = (
    "DefaultSort: a primary output metric is the metric most central to the query, "
    "typically the one used in a filter condition or explicitly requested as the "
    "main output value. Identify if such a metric exists for this Question. If it "
    "exists and is itself a computed score or metric (not a categorical dimension"
    "like a name, tier, or group/bucket label), prefer ORDER BY that metric DESC — "
    "unless the question explicitly suggests ascending order, or the metric is one "
    "where a lower value is the better or more important result (e.g. a cost, "
    "error rate, time, or other quantity where smaller is preferable), in which "
    "case use ASC instead. If no single metric is clearly primary, or the primary "
    "output is a categorical dimension rather than a computed value, do not add an "
    "ORDER BY."
)


def should_inject_default_sort(question: str) -> bool:
    """Inject the default DESC sort hint unless the question already specifies a direction.

    When an explicit direction is present (ascending/descending/highest first/etc.) the hint
    would create a contradictory signal, so we suppress it. In all other cases the hint is
    safe — its text is self-limiting ("when results include a computed score or metric",
    "does not suggest ascending order", "if no single metric is clearly primary, do not add
    an ORDER BY").
    """
    return not bool(_SORT_DIRECTION.search(question))


def _last_answer_is_stuck(history: list[dict]) -> bool:
    if not history:
        return False
    last_answer = history[-1].get("a", "").lower()
    return any(phrase in last_answer for phrase in _STUCK_PHRASES)


def _format_resolved_schema_terms(
    resolved_hits: list[tuple[str, str, float, str]],
) -> str:
    """Format resolved VDB hits as concise one-liners for the decide-LLM prompt."""
    lines = []
    for entity, hit_text, *_ in resolved_hits:
        body = re.sub(r"^ColumnAttribute:[^.]+\.\s*", "", hit_text or "").rstrip()
        lines.append(f'- "{entity}" → {body}' if body else f'- "{entity}" → {hit_text}')
    return "\n".join(lines)


def _cache_resolved_hits(
    session: "InteractiveSessionState",
    resolved_hits: list[tuple[str, str, float, str]],
) -> None:
    """Persist resolved hits to session for reuse across this turn (see
    _cached_resolved_hits' consumers: clarify prompt building and
    build_grounded_terms_hint in evidence.py)."""
    session._cached_resolved_hits = resolved_hits


_PRUNE_RESOLVED_PROMPT = """\
Clarification question asked: {question}
User's answer: {answer}

The following terms could not be resolved from the database schema or external knowledge.
For each term, decide whether the user's answer now fully resolves it — i.e., the answer
explicitly defines what the term means, what data to show, or how to compute it.

Terms:
{terms}

Output one line per term, exactly:
<term>: RESOLVED
<term>: UNRESOLVED\
"""


def prune_resolved_terms(
    persistent: list[str],
    last_turn: dict,
    llm,
) -> list[str]:
    """Remove terms from persistent_unresolved that the latest user answer resolved.

    Returns the pruned list (terms still unresolved after this answer).
    """
    if not persistent or not last_turn:
        return persistent
    term_list = "\n".join(f"- {t}" for t in persistent)
    prompt = _PRUNE_RESOLVED_PROMPT.format(
        question=last_turn["q"],
        answer=last_turn["a"],
        terms=term_list,
    )
    response = safe_invoke_text(llm, prompt).strip()
    logger.debug("Clarify — prune_resolved_terms raw response: %s", response[:200])
    resolved: set[str] = set()
    for line in response.splitlines():
        if ": RESOLVED" in line.upper():
            term = line.split(":")[0].strip().lstrip("- ").lower()
            resolved.add(term)
    remaining = [t for t in persistent if t.lower() not in resolved]
    if resolved:
        logger.info("Clarify — persistent terms resolved: %s", resolved)
    return remaining


def should_clarify(
    session: "InteractiveSessionState",
    llm,
) -> tuple[bool, Optional[str]]:
    """Return (True, question) to ask, or (False, None) to proceed to SQL."""

    def _fmt_turn(h: dict) -> str:
        a = h["a"]
        stuck = any(phrase in a.lower() for phrase in _STUCK_PHRASES)
        label = " [UNANSWERED — user could not clarify]" if stuck else ""
        return f"Q: {h['q']}\nA: {a}{label}"

    history_text = "\n".join(_fmt_turn(h) for h in session.clarify_history) or "None"

    unanswered = [
        h["q"]
        for h in session.clarify_history
        if any(phrase in h["a"].lower() for phrase in _STUCK_PHRASES)
    ]
    unanswered_topics_text = (
        "\n".join(f"- {q}" for q in unanswered) if unanswered else "None"
    )

    if session._cached_unresolvable_for != session.working_question:
        (
            unresolvable,
            resolved_hits,
            relevant_knowledge,
            extracted_norms,
            entry_to_original_terms,
            vdb_only_norms,
            json_shared_notes,
        ) = _find_unresolvable_entities(
            session.working_question,
            session.semantic_retriever,
            session.db_name,
            session.external_kb,
            session.external_kb_children_map,
            session._collision_hit_verdicts,
            session._last_ambiguous_hits,
            session._kb_vdb_adjudicated_norms,
        )
        session._cached_unresolvable = unresolvable
        session._cached_unresolvable_for = session.working_question
        session._cached_vdb_only_norms = vdb_only_norms
        # Accumulate, never replace — a note from an earlier turn (e.g. a
        # collision resolved before the entity phrasing drifted) must survive
        # even if this turn's fresh extraction has no collision to report.
        for note in json_shared_notes:
            if note not in session._collision_resolution_notes:
                session._collision_resolution_notes.append(note)
        # Always replace with the fresh KB result — never carry stale content forward.
        # An empty result is valid (entities not covered by KB this turn).
        session._grounded_knowledge = relevant_knowledge
        session._grounded_knowledge_for = session.working_question
        # Accumulate across turns: union of all KB entries seen this phase.
        # Annotate each new entry with the original natural-language terms that matched
        # it so the evidence builder can bridge phrasing gaps (e.g. "significant compliance
        # issues" → "High Audit Compliance Pressure") even when later turns stop linking them.
        if relevant_knowledge:
            existing = _parse_kb_entries(session.cumulative_grounded_knowledge)
            for name, text in _parse_kb_entries(relevant_knowledge).items():
                if name not in existing:
                    matched_from = entry_to_original_terms.get(name, [])
                    if matched_from:
                        first_nl = text.index("\n") if "\n" in text else len(text)
                        text = (
                            text[:first_nl]
                            + f"\n# matched from: {', '.join(matched_from)}"
                            + text[first_nl:]
                        )
                    session.cumulative_grounded_knowledge = (
                        session.cumulative_grounded_knowledge + "\n" + text
                        if session.cumulative_grounded_knowledge
                        else text
                    )
        # For terms covered by BOTH KB and VDB (score < 0.62), inject a
        # disambiguation note so the evidence LLM can choose between the KB
        # formula and the direct schema column rather than blindly applying both.
        #
        # Accumulated ACROSS turns, not just this turn's snapshot: the coverage-LLM
        # classification (KB side) and the VDB retrieval (schema side) are both
        # re-run fresh every turn against the evolving working_question, so the two
        # signals for the same normalized term frequently land on different turns
        # (e.g. KB confirms it turn 2, VDB only finds the column turn 3). Requiring
        # both to be true within a single turn's local resolved_hits/kb_covered_norms
        # silently missed every term whose two signals arrived a turn apart.
        kb_entries_parsed = _parse_kb_entries(relevant_knowledge)
        for entry_name, matched_terms in entry_to_original_terms.items():
            entry_text = next(
                (
                    v
                    for k, v in kb_entries_parsed.items()
                    if k.startswith(entry_name) or entry_name.startswith(k)
                ),
                "",
            )
            if not entry_text:
                continue
            for t in matched_terms:
                # Normalize the key — the coverage LLM's raw phrasing (e.g. "event
                # count") doesn't always equal the normalized form (e.g. "event"),
                # and a mismatch here silently dropped an otherwise-qualifying term.
                key = _normalize_entity(t) or t.lower().strip()
                session._ever_kb_covered_norms.add(key)
                session._ever_term_to_kb_entry.setdefault(key, (entry_name, entry_text))
        for norm, col_text, score, *_ in resolved_hits:
            session._ever_vdb_hit_norms.setdefault(norm, (col_text, score))

        kb_covered_hits = [
            (norm, col_text, score)
            for norm, (col_text, score) in session._ever_vdb_hit_norms.items()
            if norm in session._ever_kb_covered_norms
            and score < _KB_VDB_DISAMBIG_THRESHOLD
            # Already adjudicated by _resolve_collisions' merged KB-formula-
            # vs-column decision (possibly off a fresher score than the one
            # frozen here by setdefault) — don't re-decide independently and
            # risk contradicting it.
            and norm not in session._kb_vdb_adjudicated_norms
        ]
        for norm, col_text, score in kb_covered_hits:
            entry_name, kb_text = session._ever_term_to_kb_entry.get(norm, ("", ""))
            if not kb_text:
                continue
            # Dedup by KB entry name — stable across turns unlike norm phrasing
            marker = f"[DISAMBIGUATION for KB:'{entry_name}'"
            if marker in session.cumulative_grounded_knowledge:
                continue
            note = (
                f"\n{marker}: "
                f"KB defines it as: {kb_text[:200].strip()} "
                f"— but schema also has a direct column: {col_text[:120].strip()}. "
                f"In evidence, choose whichever fits the question domain — not both.]"
            )
            session.cumulative_grounded_knowledge += note
            logger.info(
                "Clarify — KB+VDB disambiguation note added for KB entry %r (VDB score=%.3f)",
                entry_name,
                score,
            )

        # Capture all extracted entities on the very first clarify call (turn 0).
        if not session.initial_extracted_entities:
            session.initial_extracted_entities = sorted(extracted_norms)
        # Cache resolved hits for reuse this turn (clarify prompt + evidence generation).
        _cache_resolved_hits(session, resolved_hits)
        # Prune persistent terms that were extracted this turn but are no longer
        # unresolvable — they were resolved via KB coverage or VDB this turn.
        current_unresolvable_names = {e for e, _ in unresolvable}
        resolved_by_kb_or_vdb = extracted_norms - current_unresolvable_names
        if resolved_by_kb_or_vdb:
            logger.info(
                "Clarify — persistent terms resolved by KB/VDB: %s",
                resolved_by_kb_or_vdb,
            )
        session.persistent_unresolved = [
            t for t in session.persistent_unresolved if t not in resolved_by_kb_or_vdb
        ]
        # Accumulate new unresolvable terms into the persistent list,
        # skipping any already resolved in a prior turn.
        existing = set(session.persistent_unresolved)
        for name, _ in unresolvable:
            if name not in existing and name not in session.resolved_persistent:
                session.persistent_unresolved.append(name)
                existing.add(name)
    else:
        resolved_hits = []
        for entity, hit_text, score, hit_id in session._cached_resolved_hits or []:
            resolved_hits.append((entity, hit_text, score, hit_id))
    unresolvable = session._cached_unresolvable or []

    # Merge current unresolvable with any persistently-tracked terms that dropped
    # out of entity extraction in later turns.
    current_names = [e for e, _ in unresolvable]
    current_set = set(current_names)
    persistent_extra = [
        t for t in session.persistent_unresolved if t not in current_set
    ]
    all_unresolvable = current_names + persistent_extra
    unresolvable_text = (
        "\n".join(f"- {t}" for t in all_unresolvable) if all_unresolvable else "None"
    )

    resolved_schema_text = _format_resolved_schema_terms(resolved_hits) or "None"

    sort_direction_note = (
        ""  # Sort direction is handled by the default DESC hint at SQL gen time.
    )

    grounded_knowledge_for_prompt = session._grounded_knowledge or "None"
    logger.debug(
        "Clarify — feeding to decide-LLM | relevant_knowledge (%d chars): %r",
        len(grounded_knowledge_for_prompt),
        grounded_knowledge_for_prompt[:300],
    )

    # Turn-0 scan: run completeness before any Q&A to surface missing formulas.
    # Fires when KB has content OR when the question implies a calculation and there
    # are VDB-only entities (schema hits with no KB formula).
    has_calc_vdb = _CALC_TRIGGER.search(session.working_question) and bool(
        session._cached_vdb_only_norms
    )
    if (
        not session.clarify_history
        and not session.incomplete_formula_terms
        and (session._grounded_knowledge or has_calc_vdb)
    ):
        from .completeness import detect_incomplete_formulas

        if has_calc_vdb:
            hits_map = {e: t for e, t, *_ in resolved_hits}
            vdb_only = []
            for e in sorted(session._cached_vdb_only_norms):
                if e in hits_map:
                    desc = re.sub(
                        r"^ColumnAttribute:[^.]+\.\s*", "", hits_map[e]
                    ).rstrip()
                    vdb_only.append(f"{e}: {desc}" if desc else e)
                else:
                    vdb_only.append(e)
        else:
            vdb_only = []
        gaps = detect_incomplete_formulas(
            session.working_question,
            last_turn=None,
            relevant_knowledge=knowledge_for_user_decisions(
                session._grounded_knowledge or "", session.external_kb
            ),
            current_gaps=[],
            llm=llm,
            vdb_only_entities=vdb_only,
            resolved_schema_terms=resolved_schema_text,
        )
        if gaps:
            session.incomplete_formula_terms = gaps
            logger.info("Completeness (turn-0 scan) — gaps: %s", gaps)

    if session.incomplete_formula_terms:
        incomplete_formulas_note = "\n".join(
            f"- {term}: {missing}" for term, missing in session.incomplete_formula_terms
        )
    else:
        incomplete_formulas_note = "None"

    turns_used = len(session.clarify_history)
    turns_remaining = session.max_clarify_turns - turns_used

    prompt = _CLARIFY_PROMPT.format(
        db_schema=_compact_schema(session.db_schema),
        relevant_knowledge=grounded_knowledge_for_prompt,
        external_knowledge=external_kb_text(session.external_kb) or "None",
        resolved_schema_terms=resolved_schema_text,
        question=session.working_question,
        history=history_text,
        unanswered_topics=unanswered_topics_text,
        unresolvable_terms=unresolvable_text,
        incomplete_formulas_note=incomplete_formulas_note,
        sort_direction_note=sort_direction_note,
    )
    response = safe_invoke_text(llm, prompt).strip()

    if response.upper().startswith("ASK:"):
        question = response[4:].strip()
        if asks_unanswerable_mechanics(question):
            # A formula or column question will be refused. Drop it and still
            # allow a never-asked business choice below to use this turn.
            logger.info(
                "Clarify — suppressed formula/column/SQL question: %r",
                question,
            )
            response = "PROCEED"
        else:
            logger.info(
                "Clarify — DECISION: ASK  (history len=%d)",
                len(session.clarify_history),
            )
            return True, question

    # Guard: override PROCEED if any incomplete term has never been asked about.
    # The decide-LLM may rationalize PROCEED when it can see VDB hints, but a term
    # that hasn't appeared in any past question is genuinely unasked and needs a turn.
    if (
        not response.upper().startswith("ASK:")
        and session.incomplete_formula_terms
        and turns_remaining > 0
    ):
        # Same split regex on both sides (unlike an earlier version of this check,
        # which split the term on whitespace/slash only and the asked-history text
        # on whitespace/slash/any-non-word-char — that asymmetry mashed hyphenated
        # terms like "X-to-Y join" into one token that could never match how the
        # words appear, split apart, in a real sentence).
        asked_tokens = {
            tok.lower()
            for h in session.clarify_history
            for tok in _TOKEN_SPLIT.split(h["q"])
            if tok
        }
        for term, description in session.incomplete_formula_terms:
            term_key = term.strip().lower()
            if term_key in session._forced_terms_asked:
                # Already force-asked this exact term in a prior turn — the LLM's
                # generated question deliberately paraphrases the term into business
                # language (see _FORCED_QUESTION_PROMPT), so lexical overlap with
                # that question is not a reliable "already asked" signal on its own.
                continue
            term_tokens = {
                tok
                for tok in _TOKEN_SPLIT.split(term.lower())
                if tok and tok not in _FILLER and tok not in _CONNECTIVES
            }
            if term_tokens and not (term_tokens & asked_tokens):
                if _FORMULA_GAP.search(term) or _FORMULA_GAP.search(description):
                    # A missing formula or column is resolved from external
                    # knowledge at SQL time. Forcing a user question here only
                    # spends a turn the user will refuse.
                    session._forced_terms_asked.add(term_key)
                    logger.info(
                        "Clarify — skip forced question for mechanics gap %r",
                        term,
                    )
                    continue
                question = _generate_forced_question(term, description, llm)
                session._forced_terms_asked.add(term_key)
                if (
                    question.strip().upper() == "SKIP"
                    or asks_unanswerable_mechanics(question)
                    or "?" not in question
                ):
                    logger.info(
                        "Clarify — forced question for %r is not a business "
                        "choice (%r)",
                        term,
                        question,
                    )
                    continue
                logger.info(
                    "Clarify — DECISION override: ASK (never-asked incomplete term %r)",
                    term,
                )
                return True, question

    # Malformed-response fallback: the decide-LLM is instructed to return exactly
    # "ASK: <question>" or "PROCEED", so anything else is a parsing failure, not
    # a real verdict. Defaulting to PROCEED on malformed output risks silently
    # skipping a real clarifying question, so treat a short response containing
    # a "?" as a dropped "ASK:" prefix and ask it verbatim. The length cap avoids
    # misfiring on longer off-format text (e.g. stray reasoning) that happens to
    # contain a question mark without actually being a question to ask.
    if "?" in response and len(response) < 300:
        if asks_unanswerable_mechanics(response):
            logger.info(
                "Clarify — DECISION: PROCEED (suppressed malformed "
                "formula/column/SQL question: %r)",
                response,
            )
            return False, None
        logger.warning(
            "Clarify — DECISION: malformed response treated as ASK (%r)", response
        )
        return True, response

    logger.info(
        "Clarify — DECISION: PROCEED  (history len=%d)", len(session.clarify_history)
    )
    return False, None


def _generate_forced_question(term: str, description: str, llm) -> str:
    """Generate a targeted clarification question for a term that is both incomplete and unresolvable."""
    prompt = _FORCED_QUESTION_PROMPT.format(term=term, description=description)
    return safe_invoke_text(llm, prompt).strip()


def refresh_grounded_knowledge(session: "InteractiveSessionState") -> None:
    """Run entity extraction and KB coverage without making a clarification decision.

    Updates session._grounded_knowledge and session.cumulative_grounded_knowledge so Evidence
    generation has current KB context. Used in Phase 2 where clarification questions
    are not allowed but KB grounding is still needed.
    """
    if session._cached_unresolvable_for == session.working_question:
        logger.info("Clarify — KB already current for Phase 2 question (cached)")
        return
    (
        unresolvable,
        resolved_hits,
        relevant_knowledge,
        _,
        entry_to_original_terms,
        _vdb_only,
        json_shared_notes,
    ) = _find_unresolvable_entities(
        session.working_question,
        session.semantic_retriever,
        session.db_name,
        session.external_kb,
        session.external_kb_children_map,
        session._collision_hit_verdicts,
        session._last_ambiguous_hits,
        session._kb_vdb_adjudicated_norms,
    )
    session._cached_unresolvable = unresolvable
    session._cached_unresolvable_for = session.working_question
    # Accumulate, never replace — see the sibling call site above.
    for note in json_shared_notes:
        if note not in session._collision_resolution_notes:
            session._collision_resolution_notes.append(note)
    session._grounded_knowledge = relevant_knowledge
    session._grounded_knowledge_for = session.working_question
    _cache_resolved_hits(session, resolved_hits)
    if relevant_knowledge:
        existing = _parse_kb_entries(session.cumulative_grounded_knowledge)
        for name, text in _parse_kb_entries(relevant_knowledge).items():
            if name not in existing:
                matched_from = entry_to_original_terms.get(name, [])
                if matched_from:
                    first_nl = text.index("\n") if "\n" in text else len(text)
                    text = (
                        text[:first_nl]
                        + f"\n# matched from: {', '.join(matched_from)}"
                        + text[first_nl:]
                    )
                session.cumulative_grounded_knowledge = (
                    session.cumulative_grounded_knowledge + "\n" + text
                    if session.cumulative_grounded_knowledge
                    else text
                )
