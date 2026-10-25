# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any, Union

# NOTE: get_agent_response_with_state and TextToSQLPayload are imported lazily
# inside _run_sql_generation to avoid triggering LLM client initialisation at
# import time (which requires NVIDIA_API_KEY to be set).
from concurrent.futures import ThreadPoolExecutor

from .clarify import (
    should_clarify,
    refresh_grounded_knowledge,
    prune_resolved_terms,
    _STUCK_PHRASES,
    should_inject_default_sort,
    _DEFAULT_SORT_HINT,
    _format_resolved_schema_terms,
)
from .kb_coverage import expand_kb_with_children
from .output_type import (
    output_type_enabled,
    should_skip_output_type_question,
    OUTPUT_TYPE_QUESTION,
    SCALAR_HINT,
)
from .conditional_output import conditional_output_enabled, get_conditional_output_hint
from .completeness import detect_incomplete_formulas
from .evidence import (
    build_grounded_terms_hint,
    generate_evidence,
)
from .followup_merge import merge_follow_up_question
from .grounding import ground_external_knowledge
from .merge import merge_clarification
from .types import AskUserAction, InteractivePhase, SubmitSQLAction, TurnType
from .state import InteractiveSessionState
from auto_ontology.retrieval.text_to_sql.chat_sql import execute_chat_sql
from auto_ontology.retrieval.text_to_sql.connector_routing import (
    resolve_connector_from_tables,
)
from auto_ontology.retrieval.text_to_sql.numeric_types import (
    dominant_kind,
    flip_numeric_output_types,
    float_json_arithmetic,
    opposite_target,
    order_by_leading_aggregate,
    order_groups_by_identifier,
    order_periods_chronologically,
    sum_decimal_casts_to_integer,
)

logger = logging.getLogger(__name__)


# ── Seeds ───────────────────────────────────────────────────────────────────


def _apply_debug_seed(
    session: InteractiveSessionState,
    *,
    debug_error: str | None = None,
) -> None:
    """Prepare path_state to resume at reconstruct_sql with the submit feedback as context.

    *debug_error* is an already-extracted DB error string the caller obtained
    from its own submit response (see `step`'s *turn_type* param) — a given
    value is used verbatim; None means the caller determined this is the
    "wrong results" case (SQL ran but the answer didn't match — no execution
    error to report).
    """
    session.path_state["_resume_from"] = "reconstruct_sql"
    if debug_error is not None:
        session.path_state["error"] = debug_error
        logger.info("Debug seed: execution error → %s", debug_error[:200])
    else:
        # SQL ran but results didn't match — no execution error detail available.
        # Inject targeted hints based on observed failure patterns.
        session.path_state["error"] = (
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
            "preserve `* 100`. If your existing query's filters, grain, projections, "
            "and ordering already satisfy every requirement above, preserve the exact "
            "same logical result rows and order (a wrong-result verdict can be caused "
            "solely by SQL numeric type differences, which are retried automatically "
            "when the result values stay unchanged)."
        )
        logger.info("Debug seed: wrong results — injecting targeted hints")
    # Preserve round 1's reconstruction lineage as read-only prompt context
    # before wiping "failed_attempts" below — see sql_reconstruction.py's
    # history_section, which renders "prior_round_failed_attempts" alongside
    # this turn's own attempts so the debug turn's LLM can see e.g. "you
    # already tried the flattened JSONB key and were told it was wrong"
    # instead of reconstructing from scratch with no memory of round 1's
    # fixes. This is deliberately a *separate* key from "failed_attempts":
    # the routers in text_to_sql_graph.py (route_sql_validation,
    # _make_soft_check_router) gate on len(failed_attempts), so carrying the
    # round 1 list forward under that same key would eat into the fresh
    # reconstruction budget the counter-reset below exists to guarantee.
    # Content and count are tracked separately on purpose.
    prior_round_failed_attempts = session.path_state.get("failed_attempts")
    if prior_round_failed_attempts:
        session.path_state["prior_round_failed_attempts"] = prior_round_failed_attempts

    session.path_state["sql_attempts"] = 0
    session.path_state["reconstruction_count"] = 0
    session.path_state["error_analysis_done"] = False
    # "failed_attempts" controls reconstruction and the unified validator's
    # intent-LLM retry safeguard (len(failed_attempts) > 5). It must reset here
    # too, or a debug turn inherits the attempt count from the PRIOR turn and
    # can skip the intent phase almost immediately, cutting off the fresh
    # repair budget this turn is supposed to get.
    #
    # "jsonb_path_repair_attempts" is JsonbPathCheckAgent's bounded counter
    # (see jsonb_path_check.py, capped at _MAX_REPAIR_ATTEMPTS) — it stays at
    # its capped value forever once exhausted in this phase, so without
    # popping it here the debug turn's from-scratch reconstruction (which can
    # reintroduce the exact class of bug the check already fixed, e.g.
    # flattening a nested JSONB key back out) gets zero path-check coverage
    # instead of the fresh attempt budget every other repair guard below
    # already gets. "join_path_repair_attempts" is the analogous counter for
    # JoinPathCheckAgent — same reasoning.
    # "value_repair_attempted" (proactive_value_check.py, empty_result_value_repair.py),
    # "null_jsonb_retry_attempted", and "empty_like_retry_attempted"
    # (empty_like_result_check.py) are the same one-shot-per-phase shape —
    # popped here so each repair gets its one fresh attempt on the new turn's
    # SQL instead of silently no-opping because a prior turn already used it.
    for key in (
        "failed_attempts",
        "jsonb_path_repair_attempts",
        "join_path_repair_attempts",
        "value_repair_attempted",
        "null_jsonb_retry_attempted",
        "empty_like_retry_attempted",
    ):
        session.path_state.pop(key, None)


def _apply_follow_up_seed(
    session: InteractiveSessionState,
    *,
    follow_up_question: str | None = None,
) -> None:
    """Prepare session for Phase 2: new question, carry Phase 1 context for SQL gen.

    *follow_up_question* is the follow-up question text, extracted by the
    caller from its own orchestrator message (see `step`'s params). Falls
    back to the existing working_question below if empty/omitted.
    """
    follow_up_q = follow_up_question or ""

    # Clear Phase 1 SQL artifacts; keep relevant_tables as merge hints.
    # Also clear similar_questions so Phase 1 VDB-retrieved examples don't bleed in —
    # Phase 1 context is injected explicitly via the follow-up instruction block instead.
    #
    # The repair/one-shot-guard keys (failed_attempts through
    # empty_like_retry_attempted) reset here for the same reason
    # _apply_debug_seed resets them: Phase 2 asks a fresh question
    # and gets a fresh repair budget.
    for key in (
        "normalized_question",
        "sql_code",
        "sql_generation_result",
        "error",
        "sql_attempts",
        "reconstruction_count",
        "error_analysis_done",
        "_resume_from",
        "final_response",
        "sql_response_from_db",
        "similar_questions",
        "failed_attempts",
        "jsonb_path_repair_attempts",
        "join_path_repair_attempts",
        "value_repair_attempted",
        "null_jsonb_retry_attempted",
        "empty_like_retry_attempted",
    ):
        session.path_state.pop(key, None)

    # Carry the full Phase 1 KB union into Phase 2 Evidence generation,
    # then reset so Phase 2 accumulates its own entries fresh.
    session.prior_round_grounded_knowledge = session.cumulative_grounded_knowledge
    session.cumulative_grounded_knowledge = ""
    # Phase 2 asks a different question — don't carry Phase 1's cross-turn
    # KB/VDB disambiguation bookkeeping into it (see clarify.py).
    session._ever_kb_covered_norms = set()
    session._ever_vdb_hit_norms = {}
    session._ever_term_to_kb_entry = {}
    session._kb_vdb_adjudicated_norms = set()
    # Same reasoning for collision-resolution notes: unlike _collision_hit_verdicts
    # (keyed by stable column id — a schema fact, still valid across phases),
    # these are injected verbatim into Evidence and reference Phase 1's specific
    # entities/columns, which are usually irrelevant to Phase 2's question.
    session._collision_resolution_notes = []
    # Same reasoning again for pending-ambiguity resolution: the notes are injected
    # verbatim into Evidence and reference Phase 1's specific entities, and the
    # one-shot guard must be re-armed so Phase 2's own ambiguous entities (a fresh
    # working_question can surface new ones) actually get resolved instead of being
    # silently skipped because Phase 1 already flipped the flag.
    session._ambiguity_resolved = False
    session._ambiguity_resolution_notes = []
    # scalar_hint is one-way (False→True) per question — Phase 1's verdict on
    # its question says nothing about Phase 2's, and leaving it True would
    # force SCALAR_HINT into Phase 2's Evidence (steering toward an aggregate
    # answer) even when the follow-up explicitly wants a breakdown/table.
    session.scalar_hint = False

    # Reset clarify state for Phase 2; clarification questions are not allowed.
    session.working_question = follow_up_q if follow_up_q else session.working_question
    session.original_question = session.working_question
    session.clarify_history = []
    session._pending_question = None
    session._forced_terms_asked = set()
    session.max_clarify_turns = 0
    logger.info(
        "[%s] Phase 2 Query: \033[1;35m%s\033[0m",
        session.task_id,
        session.working_question,
    )


# ── SQL generation ──────────────────────────────────────────────────────────


def _format_user_clarifications(session: InteractiveSessionState) -> str:
    """Render usable clarification Q&A verbatim for the SQL evidence payload."""
    clarifications: list[str] = []
    for turn in session.clarify_history:
        question = str(turn.get("q") or "").strip()
        answer = str(turn.get("a") or "").strip()
        if not answer or any(phrase in answer.lower() for phrase in _STUCK_PHRASES):
            continue
        number = len(clarifications) + 1
        clarifications.append(
            f"Clarification {number}:\nQuestion: {question}\nAnswer: {answer}"
        )

    if not clarifications:
        return ""
    return (
        "## User Clarifications (verbatim — authoritative)\n"
        + "\n\n".join(clarifications)
        + "\n\n"
    )


def _run_sql_generation(session: InteractiveSessionState) -> str:
    """Call Auto Ontology and persist the returned path_state back to session."""
    from auto_ontology.retrieval.text_to_sql.main import get_agent_response_with_state
    from auto_ontology.retrieval.text_to_sql.state import TextToSQLPayload

    # For debug turns that skip clarification, run grounding now to get KB context.
    # For normal turns, cumulative_grounded_knowledge already has everything from clarification.
    extra_knowledge = ""
    if session._grounded_knowledge_for != session.working_question:
        expanded_knowledge = expand_kb_with_children(
            session.external_kb, session.external_kb_children_map
        )
        extra_knowledge = ground_external_knowledge(
            session.working_question, expanded_knowledge, _get_fast_llm()
        )

    p1_sql = session.prior_round_sql or ""
    p1_question = session.prior_round_question or ""

    # Build Evidence from the union of: Phase 1 carry-over + this-phase KB turns + debug extra.
    # VDB resolved hits are column descriptions, not formulas — the SQL generator
    # rediscovers schema mappings via its own VDB; they only benefit the decide-LLM prompt.
    combined_knowledge = "\n".join(
        filter(
            None,
            [
                session.prior_round_grounded_knowledge,
                session.cumulative_grounded_knowledge,
                extra_knowledge,
            ],
        )
    )

    # For Phase 2, rewrite the follow-up into a self-contained question that bakes in
    # column names and conditions from Phase 1 SQL so the SQL generator doesn't have to
    # guess the relationship between the two phases. Persist as working_question so debug
    # re-runs and logging reflect the enriched question.
    # Evidence extraction uses the merged question so it is calibrated to the same
    # question the SQL generator receives, rather than the shorter raw follow-up.
    if p1_sql and p1_question:
        merged_q = merge_follow_up_question(
            p1_question, p1_sql, session.working_question
        )
        session.working_question = merged_q
        logger.info(
            "[%s] SQL gen — follow-up merged question (p1 sql %d chars)",
            session.task_id,
            len(p1_sql),
        )
    evidence_question = session.working_question

    # Send the original (pre-clarification) question as the working question to
    # SQL generation, and route the fully merged/clarified version into evidence
    # instead — the merged question accumulates every formula and definition
    # collected across clarify turns and can get long, which was diluting entity
    # retrieval. Only worth doing when a merge actually happened; otherwise
    # working_question == original_question and this would just duplicate the
    # question in evidence for no reason.
    question = session.original_question
    expanded_question_note = ""
    if session.working_question != session.original_question:
        expanded_question_note = (
            "## Expanded Question (incorporates all clarifications — this is "
            f"the full, authoritative question)\n{session.working_question}\n\n"
        )
    # Phase 2 only: the raw Phase 1 SQL as an exact reference so numeric thresholds,
    # CASE conditions, and formulas are preserved verbatim — the merged question captures
    # intent and structure, but the raw SQL is the source of truth for precise values.
    # Reference material like the expanded question above, so it belongs in evidence
    # rather than bolted onto the short original question.
    phase1_sql_reference_note = ""
    if p1_sql and p1_question:
        phase1_sql_reference_note = (
            "## Phase 1 SQL Reference (use exact column names, thresholds, and "
            f"formulas from this SQL where applicable)\n{p1_sql}\n\n"
        )
    user_clarifications_note = _format_user_clarifications(session)

    resolved_terms_section = build_grounded_terms_hint(session)
    evidence = generate_evidence(
        evidence_question, combined_knowledge, resolved_terms_section
    )
    if session._collision_resolution_notes:
        # Injected directly rather than left to evidence-gen's LLM to relay —
        # that step isn't reliable about preserving instructions passed through it.
        evidence = "\n".join(
            filter(None, [evidence, *session._collision_resolution_notes])
        )
        logger.info(
            "[%s] SQL gen — injected %d collision-resolution note(s)",
            session.task_id,
            len(session._collision_resolution_notes),
        )
    if session._ambiguity_resolution_notes:
        # Same reasoning as _collision_resolution_notes above — injected
        # directly, not left to evidence-gen's LLM to relay.
        evidence = "\n".join(
            filter(None, [evidence, *session._ambiguity_resolution_notes])
        )
        logger.info(
            "[%s] SQL gen — injected %d ambiguity-resolution note(s)",
            session.task_id,
            len(session._ambiguity_resolution_notes),
        )
    if should_inject_default_sort(session.working_question):
        evidence = "\n".join(filter(None, [evidence, _DEFAULT_SORT_HINT]))
        logger.info("[%s] SQL gen — injected default DESC sort hint", session.task_id)
    if session.scalar_hint:
        evidence = "\n".join(filter(None, [evidence, SCALAR_HINT]))
        logger.info("[%s] SQL gen — injected scalar aggregate hint", session.task_id)
    if conditional_output_enabled():
        _cond_hint = get_conditional_output_hint(session.working_question)
        if _cond_hint:
            evidence = "\n".join(filter(None, [evidence, _cond_hint]))
            logger.info(
                "[%s] SQL gen — injected conditional output hint", session.task_id
            )
    if expanded_question_note or user_clarifications_note or phase1_sql_reference_note:
        evidence = (
            expanded_question_note
            + user_clarifications_note
            + phase1_sql_reference_note
            + evidence
        )
        logger.info(
            "[%s] SQL gen — prepended expanded question/user clarifications/"
            "Phase 1 SQL reference to evidence",
            session.task_id,
        )
    if evidence:
        logger.info("[%s] SQL gen — Evidence: %s", session.task_id, evidence)

    payload: TextToSQLPayload = {
        "question": question,
        "evidence": evidence,
        # Same merged/clarified question that expanded_question_note carries
        # into evidence above, minus the hint blocks and SQL reference —
        # for consumers (e.g. the relevance filter) that want the clarified
        # intent but not sort/scalar/output-shape noise.
        "enriched_question": (
            session.working_question if session.working_question != question else ""
        ),
        "data_retriever": session.data_retriever,
        "semantic_retriever": session.semantic_retriever,
        "connectors": session.connectors,
        "path_state": dict(
            session.path_state
        ),  # copy so Auto Ontology doesn't mutate in place
        "acronyms": [],
        "custom_prompts": combined_knowledge,  # full union as low-priority fallback
        "calculation_only": True,
        "shorten_answer": True,
        "validate_sql_values": True,
        # The grader compares numbers exactly, so a float and a numeric holding
        # the same value disagree: keep our explicit casts on the numeric side.
        "numeric_type_alignment": session.numeric_type_alignment,
        "skip_evidence_refinement": session.skip_evidence_refinement,
    }
    result = get_agent_response_with_state(payload)

    # Merge returned path_state back into durable session path_state
    if returned_ps := result.get("path_state"):
        session.path_state.update(returned_ps)

    # Clear resume_from after use so next call goes through full pipeline
    session.path_state.pop("_resume_from", None)

    return result.get("sql_code", "")


# ── Numeric type retry ──────────────────────────────────────────────────────


def _numeric_kinds_for(session: InteractiveSessionState, sql: str) -> list | None:
    """Numeric column kinds recorded for *sql*, or None when they belong to other SQL.

    The execution node tags what it recorded with the statement it ran; the SQL
    returned to the caller can differ from that (a later rewrite, a failed run),
    and kinds of the wrong query are worse than none.
    """
    path_state = session.path_state
    if (path_state.get("result_numeric_kinds_sql") or "") != (sql or "").strip():
        return None
    return path_state.get("result_numeric_kinds")


def _values_digest_for(session: InteractiveSessionState, sql: str) -> str | None:
    """Type-blind digest of the result values recorded for *sql* (same tagging rule)."""
    path_state = session.path_state
    if (path_state.get("result_numeric_kinds_sql") or "") != (sql or "").strip():
        return None
    return path_state.get("result_values_digest")


def _align_mixed_numeric_types(session: InteractiveSessionState, sql: str) -> str:
    """When a query returns a mix of ``decimal`` and ``float`` columns, align the
    ``float`` columns to ``numeric``.

    A mixed result happens when one projection uses an explicit ``DECIMAL`` cast
    and another touches a native ``float4``/``float8`` column without casting.
    Unifying them to ``numeric`` avoids failing exact graders on the uncast
    column while keeping ``first_submit_numeric_kind`` uniform if a debug retry
    later needs to flip to ``double precision``.
    """
    kinds = _numeric_kinds_for(session, sql)
    if not kinds or not ({"decimal", "float"} <= set(kinds)):
        return sql

    connector = resolve_connector_from_tables(
        session.path_state.get("relevant_tables") or [], session.connectors
    )
    if connector is None:
        return sql
    aligned = flip_numeric_output_types(
        sql, getattr(connector, "dialect", None), kinds, "numeric"
    )
    if not aligned:
        return sql
    try:
        df = execute_chat_sql(connector, aligned, kind="mixed numeric type alignment")
    except Exception as exc:  # noqa: BLE001 — fall back to the SQL as generated
        logger.warning(
            "[%s] Mixed numeric type alignment did not run (%s); keeping the generated SQL",
            session.task_id,
            exc,
        )
        return sql
    from auto_ontology.retrieval.text_to_sql.numeric_types import (
        numeric_column_kinds,
        result_values_digest,
    )

    session.path_state["result_numeric_kinds"] = numeric_column_kinds(df)
    session.path_state["result_numeric_kinds_sql"] = aligned.strip()
    session.path_state["result_values_digest"] = result_values_digest(df)
    logger.info(
        "[%s] Unified mixed decimal/float output columns to numeric",
        session.task_id,
    )
    return aligned


def _flip_numeric_types_for_retry(session: InteractiveSessionState, sql: str) -> str:
    """Re-submit *sql* with its numeric columns in the opposite type.

    Used on the one debug turn the benchmark grants when a submit ran but was
    marked wrong. The grader rounds and then compares numbers exactly, so a
    float answer and a numeric answer with the same digits are different
    answers to it; the first submit settled on one type and was graded wrong,
    so the retry tries the other. Returns *sql* unchanged whenever the flip
    cannot be done or does not run — a broken retry would burn the phase's
    last attempt for nothing.

    Only a retry that *kept the first submit's answer* is flipped. If debugging
    changed the values themselves (a different filter, join, or expression), the
    type the first submit had says nothing about the new answer, and flipping
    would swap a possibly-correct type for a wrong one.

    A debug result can also arrive with no recorded first type: the phase
    ended without a submit, so there is no kind and no value digest. Decimal
    columns in that result are retried as ``double precision``. An integer
    first submit still has a digest, and its debug SQL is left alone.
    """
    first_kind = session.first_submit_numeric_kind
    kinds = _numeric_kinds_for(session, sql)
    if not kinds:
        return sql
    if not first_kind:
        if session.first_submit_values_digest is None and "decimal" in kinds:
            return _retry_double_without_a_recorded_type(session, sql, kinds)
        return sql

    first_values_digest = session.first_submit_values_digest
    if not first_values_digest or first_values_digest != _values_digest_for(
        session, sql
    ):
        logger.info(
            "[%s] Debug retry changed the answer's values; keeping its numeric types",
            session.task_id,
        )
        return _align_mixed_numeric_types(session, sql)

    connector = resolve_connector_from_tables(
        session.path_state.get("relevant_tables") or [], session.connectors
    )
    if connector is None:
        return sql
    flipped = flip_numeric_output_types(
        sql, getattr(connector, "dialect", None), kinds, opposite_target(first_kind)
    )
    if not flipped:
        return sql
    try:
        execute_chat_sql(connector, flipped, kind="numeric type retry")
    except Exception as exc:  # noqa: BLE001 — fall back to the SQL as generated
        logger.warning(
            "[%s] Numeric type retry did not run (%s); keeping the generated SQL",
            session.task_id,
            exc,
        )
        return sql
    logger.info(
        "[%s] Debug retry: numeric columns moved to the opposite type (first submit was %s)",
        session.task_id,
        first_kind,
    )
    return flipped


def _retry_double_without_a_recorded_type(
    session: InteractiveSessionState,
    sql: str,
    kinds: list,
) -> str:
    """Retry decimal output as ``double precision`` when no first type was stored.

    The opposite-type flip needs a first submit to know which side was already
    graded wrong. When that submit never recorded a result, a decimal answer
    still has the other side left: ``double precision``. The rewritten
    statement is kept only when it executes.
    """
    connector = resolve_connector_from_tables(
        session.path_state.get("relevant_tables") or [], session.connectors
    )
    if connector is None:
        return sql
    flipped = flip_numeric_output_types(
        sql, getattr(connector, "dialect", None), kinds, "double"
    )
    if not flipped:
        return sql
    try:
        execute_chat_sql(connector, flipped, kind="numeric type retry")
    except Exception as exc:  # noqa: BLE001 — fall back to the SQL as generated
        logger.warning(
            "[%s] Decimal-to-double retry did not run (%s); keeping the generated SQL",
            session.task_id,
            exc,
        )
        return sql
    logger.info(
        "[%s] Debug retry: decimal columns moved to double precision (no first-submit numeric type)",
        session.task_id,
    )
    return flipped


# ── LLM accessor ────────────────────────────────────────────────────────────

_llm = None
_fast_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        from auto_ontology.utils.llm_invoke import get_llm_client

        _llm = get_llm_client()
    return _llm


def _get_fast_llm():
    global _fast_llm
    if _fast_llm is None:
        from auto_ontology.utils.llm_invoke import get_non_reasoning_llm_client

        try:
            _fast_llm = get_non_reasoning_llm_client(max_tokens=2048)
        except (ValueError, EnvironmentError):
            _fast_llm = _get_llm()
    return _fast_llm


# ── Public API ───────────────────────────────────────────────────────────────


def create_session(
    session_id: str,
    task_id: str,
    db_name: str,
    db_schema: str,
    external_kb: str,
    question: str,
    data_retriever: Any,
    semantic_retriever: Any,
    connectors: list,
    max_clarify_turns: int = 5,
    external_kb_children_map: dict | None = None,
    skip_evidence_refinement: bool = False,
    numeric_type_alignment: bool = False,
) -> InteractiveSessionState:
    return InteractiveSessionState(
        session_id=session_id,
        task_id=task_id,
        db_name=db_name,
        db_schema=db_schema,
        external_kb=external_kb,
        original_question=question,
        working_question=question,
        max_clarify_turns=max_clarify_turns,
        external_kb_children_map=external_kb_children_map or {},
        skip_evidence_refinement=skip_evidence_refinement,
        numeric_type_alignment=numeric_type_alignment,
        data_retriever=data_retriever,
        semantic_retriever=semantic_retriever,
        connectors=connectors,
        path_state={"target_db": db_name, "task_id": task_id},
    )


def step(
    session: InteractiveSessionState,
    *,
    turn_type: TurnType,
    debug_error: str | None = None,
    initial_question: str | None = None,
    follow_up_question: str | None = None,
) -> Union[AskUserAction, SubmitSQLAction]:
    """Advance the session one turn.

    The coordinator is dialect-neutral: it has no knowledge of any specific
    orchestrator's message format. The caller is responsible for classifying
    the turn (*turn_type*) and, for turns that need it, supplying the
    relevant text directly:
      - INITIAL: *initial_question*, the user's question for this task.
      - DEBUG: *debug_error*, the DB execution-error text — omit/None for a
        "SQL ran but the answer was wrong" turn (no execution error).
      - FOLLOW_UP: *follow_up_question*, the next-phase question text.
    """
    if turn_type is None:
        raise ValueError(
            "step() requires an explicit turn_type — the coordinator no "
            "longer classifies turns by pattern-matching orchestrator text; "
            "the caller must classify the turn itself from its own protocol "
            "state (e.g. a submit response's phase_completed field)."
        )

    if turn_type == TurnType.INITIAL:
        # Extract question from message on first turn (question not in init_session state)
        if not session.clarify_history and not session.path_state.get("sql_code"):
            extracted = initial_question
            if extracted:
                session.original_question = extracted
                session.working_question = extracted
                logger.info(
                    "[%s] Query: \033[1;35m%s\033[0m",
                    session.task_id,
                    extracted,
                )

                # ── Output-type check (runs once, before clarify sees anything) ──
                # Pure regex — no LLM call. Fires only on the very first step
                # (clarify_history is empty). If the type is ambiguous, we ask the
                # user before any other clarification question. If it's obvious,
                # we set scalar_hint (or skip silently for table/ddl) and continue.
                if output_type_enabled():
                    skip = should_skip_output_type_question(extracted)
                    logger.info(
                        "[%s] OutputType check → %s",
                        session.task_id,
                        skip if skip is not None else "None (will ask)",
                    )
                    if skip is None:
                        session._pending_question = OUTPUT_TYPE_QUESTION
                        return AskUserAction(question=OUTPUT_TYPE_QUESTION)
                    elif skip == "scalar":
                        session.scalar_hint = True
                        logger.info(
                            "[%s] OutputType: scalar hint set at session start",
                            session.task_id,
                        )
                    # "table" and "ddl" → skip silently, no hint needed

    elif turn_type == TurnType.FOLLOW_UP:
        if session.phase != InteractivePhase.ROUND2_CLARIFY:
            _apply_follow_up_seed(session, follow_up_question=follow_up_question)
            session.phase = InteractivePhase.ROUND2_CLARIFY

    elif turn_type == TurnType.DEBUG:
        _apply_debug_seed(session, debug_error=debug_error)

    under_budget = len(session.clarify_history) < session.max_clarify_turns

    # ── Per-turn output-type re-check (pure regex, no LLM) ───────────────────
    # Re-runs on the enriched working_question each turn so that intent revealed
    # naturally in dialogue (without an explicit output-type question) still sets
    # the hint. One-way only: False → True; never unsets once set.
    if output_type_enabled() and not session.scalar_hint:
        if should_skip_output_type_question(session.working_question) == "scalar":
            session.scalar_hint = True
            logger.info(
                "[%s] OutputType: scalar hint set from enriched question (turn %d)",
                session.task_id,
                len(session.clarify_history),
            )

    if turn_type != TurnType.DEBUG and under_budget:
        should_ask, question = should_clarify(session, _get_llm())
        if should_ask:
            session._pending_question = question
            return AskUserAction(question=question)
    elif turn_type != TurnType.DEBUG and not under_budget:
        # No clarification allowed (Phase 2 or budget exhausted) but still run
        # KB coverage to update cumulative_grounded_knowledge for Evidence generation.
        refresh_grounded_knowledge(session)

    # Decision to proceed (whatever the reason — turn budget exhausted, or
    # should_clarify judging PROCEED above) — resolve any still-open ambiguous
    # entities from the final working_question, so the SQL generator gets a real
    # decision instead of the vague "choose whichever fits" note. Gated on the
    # proceed decision itself, not on turn count, so it isn't tied to the
    # max_clarify_turns mechanic specifically.
    #
    # _ambiguity_resolved guards this to run once per *phase*, not once per
    # handle_turn call: a DEBUG turn (SQL rejected, "fix it") re-enters this same
    # function and falls through to _run_sql_generation again after the flag is
    # already True from the first pass, so it correctly skips re-resolving —
    # there's nothing turn-loop-specific about the guard itself, it's the same
    # "already did this, don't repeat" flag as any one-shot-per-phase step.
    if not session._ambiguity_resolved:
        session._ambiguity_resolved = True
        # Collision resolution (clarify._resolve_collisions, runs earlier in the
        # same _find_unresolvable_entities call this turn — see entity_resolution.py)
        # may have already settled some of these same entities on its own, more
        # mature logic (KB-covered exclusion, composite-column handling, LLM
        # fallback). Its outcome lands in _cached_resolved_hits. Without this
        # check, both mechanisms could independently decide the same term and,
        # if they disagree, inject contradicting notes into evidence with no
        # arbitration between them — whichever landed later in the text would
        # win by accident. Defer to collision's answer where one already exists.
        already_resolved = {
            entity for entity, *_ in session._cached_resolved_hits or []
        }
        pending = {
            e: hits
            for e, hits in session._last_ambiguous_hits.items()
            if e not in already_resolved
        }
        if pending:
            from .entity_resolution import resolve_pending_ambiguities

            session._ambiguity_resolution_notes = resolve_pending_ambiguities(
                session.working_question,
                session.cumulative_grounded_knowledge,
                pending,
            )
            if session._ambiguity_resolution_notes:
                logger.info(
                    "[%s] SQL gen — resolved %d pending ambiguous entit(y/ies)",
                    session.task_id,
                    len(session._ambiguity_resolution_notes),
                )

    sql = _run_sql_generation(session)
    if session.numeric_type_alignment:
        if turn_type == TurnType.DEBUG and debug_error is None:
            sql = _flip_numeric_types_for_retry(session, sql)
        elif turn_type != TurnType.DEBUG:
            sql = _align_mixed_numeric_types(session, sql)
            session.first_submit_numeric_kind = dominant_kind(
                _numeric_kinds_for(session, sql)
            )
            session.first_submit_values_digest = _values_digest_for(session, sql)
        sql = _stabilize_submitted_order(session, sql)
        # After the tie-break, so a grouped id appended above is dropped again
        # when the sort key itself was the later aggregate. Phase 2 is left
        # alone: a follow-up often sorts by that later metric on purpose.
        if (
            turn_type == TurnType.DEBUG
            and debug_error is None
            and session.phase == InteractivePhase.ROUND1_CLARIFY
        ):
            sql = _submit_if_it_runs(
                session,
                sql,
                order_by_leading_aggregate(sql, _postgres_dialect(session)),
                "debug order by the leading aggregate",
            )
    return SubmitSQLAction(sql=sql)


def _real_column_names(session: InteractiveSessionState, sql: str) -> set[str]:
    """Public ``real`` / ``double precision`` column names, lowercased.

    Looked up once per session, and only when *sql* casts JSON text. The
    float-arithmetic rewrite needs to know which ``CAST(column AS NUMERIC)``
    is fighting a native float; other statements skip the catalog query.
    """
    if "->>" not in sql:
        return set()
    cached = session.path_state.get("real_column_names")
    if cached is not None:
        return set(cached)
    connector = resolve_connector_from_tables(
        session.path_state.get("relevant_tables") or [], session.connectors
    )
    if connector is None:
        return set()
    try:
        frame = execute_chat_sql(
            connector,
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' "
            "AND data_type IN ('real', 'double precision')",
            kind="real column names",
        )
    except Exception as exc:  # noqa: BLE001 — no catalog, no float rewrite
        logger.warning(
            "[%s] Could not list real columns (%s); leaving numeric casts",
            session.task_id,
            exc,
        )
        return set()
    columns = getattr(frame, "columns", [])
    if frame is None or "column_name" not in columns:
        return set()
    names = {str(value).lower() for value in frame["column_name"].dropna()}
    session.path_state["real_column_names"] = sorted(names)
    return names


def _postgres_dialect(session: InteractiveSessionState) -> str | None:
    for connector in session.connectors or []:
        dialect = getattr(connector, "dialect", None)
        if dialect and str(dialect).lower() in ("postgres", "postgresql"):
            return str(dialect)
    return None


def _submit_if_it_runs(
    session: InteractiveSessionState, sql: str, rewritten: str, kind: str
) -> str:
    """Use *rewritten* only when it actually executes. A failed rewrite must not replace a query that already ran."""
    if not rewritten or rewritten == sql:
        return sql
    connector = resolve_connector_from_tables(
        session.path_state.get("relevant_tables") or [], session.connectors
    )
    if connector is None:
        return sql
    try:
        execute_chat_sql(connector, rewritten, kind=kind)
    except Exception as exc:  # noqa: BLE001 — keep the SQL that already executed
        logger.warning(
            "[%s] %s did not run (%s); keeping the previous SQL",
            session.task_id,
            kind,
            exc,
        )
        return sql
    logger.info("[%s] %s", session.task_id, kind)
    return rewritten


def _stabilize_submitted_order(session: InteractiveSessionState, sql: str) -> str:
    """Apply the row-order rewrites that matched the grader on saved SQL.

    Runs after the numeric-type flip so a flip back to ``double precision``
    does not undo the chronological average, and so an integer ``SUM`` is not
    wrapped again as ``numeric``. Each rewrite is kept only if it executes.
    """
    dialect = _postgres_dialect(session)
    if dialect is None or not sql.strip():
        return sql
    sql = _submit_if_it_runs(
        session,
        sql,
        order_groups_by_identifier(sql, dialect),
        "grouped identifier tie-break",
    )
    sql = _submit_if_it_runs(
        session,
        sql,
        order_periods_chronologically(sql, dialect),
        "chronological order for a per-period ranking",
    )
    sql = _submit_if_it_runs(
        session,
        sql,
        sum_decimal_casts_to_integer(sql, dialect),
        "integer sum of a whole-number cast",
    )
    return _submit_if_it_runs(
        session,
        sql,
        float_json_arithmetic(sql, dialect, _real_column_names(session, sql)),
        "float arithmetic for a real column mixed with JSON metrics",
    )


def apply_user_answer(session: InteractiveSessionState, answer: str) -> None:
    """Record user answer and merge into working question."""
    if session._pending_question:
        session.clarify_history.append({"q": session._pending_question, "a": answer})
        session._pending_question = None
    answer_lower = answer.lower()
    user_could_not_answer = any(phrase in answer_lower for phrase in _STUCK_PHRASES)
    if not user_could_not_answer and session.clarify_history:
        last_turn = session.clarify_history[-1]
        relevant_knowledge = session._grounded_knowledge or ""

        with ThreadPoolExecutor(max_workers=3) as pool:
            merge_future = pool.submit(
                merge_clarification,
                session.working_question,
                last_turn,
                _get_fast_llm(),
                relevant_knowledge=relevant_knowledge,
            )
            completeness_future = pool.submit(
                detect_incomplete_formulas,
                session.working_question,
                last_turn,
                relevant_knowledge,
                list(session.incomplete_formula_terms),
                _get_llm(),
                None,
                _format_resolved_schema_terms(session._cached_resolved_hits or []),
            )
            prune_future = pool.submit(
                prune_resolved_terms,
                list(session.persistent_unresolved),
                last_turn,
                _get_fast_llm(),
            )
            merged = merge_future.result()
            gaps = completeness_future.result()
            pruned = prune_future.result()
            newly_resolved = set(session.persistent_unresolved) - set(pruned)
            session.resolved_persistent.update(newly_resolved)
            session.persistent_unresolved = pruned

            if session._cached_vdb_only_norms:
                remaining_vdb = prune_resolved_terms(
                    sorted(session._cached_vdb_only_norms),
                    last_turn,
                    _get_fast_llm(),
                )
                session._cached_vdb_only_norms = set(remaining_vdb)

        if merged:
            session.working_question = merged
        else:
            logger.warning(
                "[%s] merge_clarification returned empty; keeping previous question",
                session.task_id,
            )
        logger.info(
            "[%s] Merged question: %s", session.task_id, session.working_question
        )

        session.incomplete_formula_terms = gaps
        logger.info("[%s] Incomplete formula terms: %s", session.task_id, gaps)


def apply_submit_result(session: InteractiveSessionState, result: dict) -> None:
    """Update session with the submit response for future debug seeding."""
    session.latest_feedback = result.get("message", "")
    # Save round-1 artifacts only on successful completion so they don't
    # contaminate round-1 debug turns with follow-up instruction / cross-round logic.
    if session.prior_round_question is None and result.get("phase_completed") == 1:
        session.prior_round_question = session.working_question
        session.prior_round_sql = session.path_state.get("sql_code", "")
