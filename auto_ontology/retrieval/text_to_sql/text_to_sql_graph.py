# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Any, Callable, Dict

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from auto_ontology.retrieval.text_to_sql.state import (
    TextToSQLPayload,
    AgentState,
    get_question_for_processing,
)
from auto_ontology.retrieval.text_to_sql.agents.candidates_preparation import (
    CandidatePreparationAgent,
)
from auto_ontology.retrieval.text_to_sql.agents.candidates_retrieval import (
    CandidateRetrievalAgent,
)
from auto_ontology.retrieval.entity_coverage.agents.question_extraction import (
    QuestionExtractionAgent,
)
from auto_ontology.retrieval.text_to_sql.agents.empty_result_value_repair import (
    EmptyResultValueRepairAgent,
)
from auto_ontology.retrieval.text_to_sql.agents.evidence_refinement import (
    EvidenceRefinementAgent,
)
from auto_ontology.retrieval.text_to_sql.agents.combined_precheck import (
    CombinedPrecheckAgent,
)
from auto_ontology.retrieval.text_to_sql.agents.question_intent import (
    QuestionIntentAgent,
)
from auto_ontology.retrieval.text_to_sql.agents.information_agent import (
    InformationAgent,
)
from auto_ontology.retrieval.text_to_sql.agents.prediction_graph import (
    PredictionGraphAgent,
)
from auto_ontology.retrieval.text_to_sql.agents.kumo_prediction import (
    KumoPredictionAgent,
)
from auto_ontology.retrieval.text_to_sql.agents.empty_like_result_check import (
    EmptyLikeResultCheckAgent,
)
from auto_ontology.retrieval.text_to_sql.agents.response import ResponseAgent
from auto_ontology.retrieval.text_to_sql.agents.sql_execution import SQLExecutionAgent
from auto_ontology.retrieval.text_to_sql.agents.sql_from_semantic import (
    SQLFromCandidatesAgent,
)
from auto_ontology.retrieval.text_to_sql.agents.sql_reconstruction import (
    SQLReconstructionAgent,
)
from auto_ontology.retrieval.text_to_sql.agents.sql_unconstructable import (
    SQLUnconstructableAgent,
)
from auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation import (
    INTENT_VALIDATION_SKIPPED_AFTER,
    SQLValidationAgent,
)
from auto_ontology.retrieval.text_to_sql.base import agent_wrapper
from auto_ontology.retrieval.text_to_sql.db_probe.config import (
    is_db_probe_proactive,
    is_db_probe_jsonb_path_check,
    is_db_probe_join_path_check,
)

logger = logging.getLogger(__name__)

# Tag on the payload a node writes to the custom stream channel as it starts.
# ``stream_agent_response`` turns it into the client's "this step is running
# now" event; keep the two in step.
NODE_START_EVENT = "step_start"

# Stop after seven reconstruction calls.
MAX_RECONSTRUCTION_ATTEMPTS = 7


def route_sql_validation(state: AgentState) -> str:
    """
    Route based on SQL validation result.

    Handles unified SQL validation attempts and retry logic:
    - "valid_sql" if parse/static/intent validation succeeds
    - "invalid_sql" if invalid (with retry logic)
    - "unconstructable" after 7 reconstructions, or when a node already gave up
      (e.g. an unreachable database, which no rewrite can fix)

    Args:
        state: Current agent state

    Returns:
        Routing decision based on validation result and attempt count
    """
    if state["decision"] == "unconstructable":
        # Passed through rather than folded into the else branch below, which
        # assumes anything that is not "invalid_sql" is usable SQL and would
        # send an already-abandoned run on toward execution.
        return "unconstructable"

    path_state = state["path_state"]
    failed_attempt_count = len(path_state.get("failed_attempts") or [])

    if state["decision"] == "invalid_sql":
        logger.info("SQL reconstruction attempts completed: %s", failed_attempt_count)
        if failed_attempt_count >= MAX_RECONSTRUCTION_ATTEMPTS:
            logger.error(
                "SQL construction failed after %s reconstructions",
                failed_attempt_count,
            )
            return "unconstructable"
        return "invalid_sql"

    return "valid_sql"


def _make_soft_check_router(check_name: str):
    """Build a router for a post-construction "soft" check (jsonb/join/value
    prechecks, empty-LIKE check, value-repair check).

    These send SQL back to ``reconstruct_sql`` on failure but do not route
    through :func:`route_sql_validation`, which enforces the hard reconstruction
    cap. A persistently failing check could otherwise loop until the graph's
    global recursion limit aborts the run. Once enough failed attempts have
    accumulated, stop blocking on this check and let the SQL through as-is.

    Args:
        check_name: Node name, used only for the log message when the cap fires.

    Returns:
        A router function usable in ``add_conditional_edges``.
    """

    def _route(state: AgentState) -> str:
        decision = state.get("decision", "") or ""
        if decision != "invalid_sql":
            return "valid_sql"
        failed_attempt_count = len(state["path_state"].get("failed_attempts") or [])
        if failed_attempt_count > INTENT_VALIDATION_SKIPPED_AFTER:
            logger.info(
                f"Skipping {check_name} after {failed_attempt_count} reconstructions"
            )
            return "valid_sql"
        return "invalid_sql"

    return _route


def route_translation(state: AgentState) -> str:
    """
    Route to translation or graph END based on target language and context.

    This function does NOT modify state. It only decides whether we need
    to translate the final_response or end the graph.

    Returns:
        - "translate": if translation is needed
        - "end": if no translation is needed
    """
    language = state.get("language", "english") or "english"
    if language.lower() != "english":
        # Non-English target language → request translation
        return "translate"

    # English (or unknown) → no translation
    return "end"


def _prediction_enabled() -> bool:
    """Whether the KumoRFM prediction branch should be built into the graph.

    Evaluated ONCE at graph-creation (startup), not per request: when
    ``KUMO_RFM_API_URL`` is unset the prediction nodes/edges are never added, so
    the classify → prepare-graph → predict path simply does not exist.

    The URL is what makes prediction possible: the SDK targets a Universal TFM
    NIM, and NIMs are unauthenticated by contract, so ``KUMO_RFM_API_KEY`` is
    optional and only carries a gateway credential when a deployment adds one.
    """
    return bool(os.environ.get("KUMO_RFM_API_URL"))


def route_decision(state: AgentState) -> str:
    """
    Generic router — returns the current ``decision`` value from state,
    applying optional aliases first.

    Each caller's ``add_conditional_edges`` edge-map defines which values
    are legal; LangGraph will raise if the returned string isn't a key in
    that map, so no extra guard-rail set is needed here.

    Alias mappings (agent-specific convenience):
    - "constructable" → "validate_sql_query"
    """
    decision = state.get("decision", "") or ""

    decision_mapping = {
        "constructable": "validate_sql_query",
    }

    mapped = decision_mapping.get(decision, decision)

    if decision != mapped:
        logger.debug("Mapped decision '%s' → '%s'", decision, mapped)

    return mapped


def route_evidence_refinement(state: AgentState) -> str:
    """Run schema-aware evidence refinement only when evidence is present."""
    if (state.get("evidence") or "").strip():
        return "refine_evidence"
    return "construct_sql_from_candidates"


def route_question_type_or_evidence(state: AgentState) -> str:
    """Route the early intent decision after candidates scope prediction inputs."""
    if state.get("path_state", {}).get("question_type") == "prediction":
        return "prediction"
    return route_evidence_refinement(state)


def route_after_candidate_retrieval(state: AgentState) -> str:
    """Send information requests to metadata answering before SQL preparation."""
    if state.get("path_state", {}).get("question_type") == "information":
        return "information"
    return "prepare_candidates"


def _make_node(name, fn):
    """
    Create a node with logging wrapper.

    For agents, use agent_wrapper instead.
    For simple functions, use this wrapper.
    """
    return RunnableLambda(wrap_node_with_logging(name, fn))


def log_node_visit(state, node_name: str):
    """
    Track how many times each graph node was visited during a run.
    """
    path_state = state.get("path_state", {})
    counts = path_state.get("node_visit_counts", {})
    counts[node_name] = counts.get(node_name, 0) + 1
    path_state["node_visit_counts"] = counts
    state["path_state"] = path_state
    total = sum(counts.values())
    logger.info(f"🔁 Node visits: {counts} | Total visits this run: {total}")


def announce_node_start(node_name: str) -> None:
    """Tell the stream this node is starting, before it does its work.

    ``app.stream()`` only yields a node's update once it has *finished*, so a
    client driven by updates alone shows the previous node's label — a 20s
    reconstruction appears as "Validating intent" hanging. The custom channel
    is streamed the moment it is written, which is what makes the label track
    the work in progress. See ``stream_agent_response``.

    Best-effort: outside a streaming context there is no writer to get, and a
    missing progress event must not take the run down with it.
    """
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()({"type": NODE_START_EVENT, "node": node_name})
    except Exception:  # noqa: BLE001 — progress reporting is best-effort
        logger.debug("No stream writer for node %s", node_name, exc_info=True)


def wrap_node_with_logging(
    node_name: str,
    fn: Callable[[AgentState], Dict[str, Any]],
) -> Callable[[AgentState], Dict[str, Any]]:
    """
    Wrap a node callable so it logs node visits automatically.
    """

    def wrapped(state: AgentState) -> Dict[str, Any]:
        announce_node_start(node_name)
        log_node_visit(state, node_name)
        return fn(state)

    return wrapped


def _entry_router_fn(state):
    if state["path_state"].get("_resume_from") == "reconstruct_sql":
        return "reconstruct_sql"
    return "question_intent"


def create_graph():

    # KumoRFM prediction is wired in only when configured — decided once here at
    # graph creation (startup), never per request.
    prediction_enabled = _prediction_enabled()
    logger.info("Text-to-SQL graph: prediction branch %s", prediction_enabled)

    # ==================== CREATE AGENT INSTANCES ====================

    # Routing agents
    question_intent_agent = QuestionIntentAgent()
    question_extraction_agent = QuestionExtractionAgent()
    retrieval_agent = CandidateRetrievalAgent()
    information_agent = InformationAgent()
    candidate_preparation_agent = CandidatePreparationAgent()
    evidence_refinement_agent = EvidenceRefinementAgent()
    sql_from_candidates_agent = SQLFromCandidatesAgent()
    sql_reconstruction_agent = SQLReconstructionAgent()
    sql_validation_agent = SQLValidationAgent()
    sql_execution_agent = SQLExecutionAgent()
    empty_like_result_check_agent = EmptyLikeResultCheckAgent()
    response_agent = ResponseAgent()
    sql_unconstructable_agent = SQLUnconstructableAgent()

    # ==================== CREATE NODES ====================

    # Routing nodes (using agent_wrapper)

    question_intent_node = _make_node(
        "question_intent", agent_wrapper(question_intent_agent)
    )
    question_extraction_node = _make_node(
        "question_extraction", agent_wrapper(question_extraction_agent)
    )
    retrieve_candidates_node = _make_node(
        "retrieve_candidates", agent_wrapper(retrieval_agent)
    )
    information_node = _make_node("information_agent", agent_wrapper(information_agent))
    prepare_candidates_node = _make_node(
        "prepare_candidates", agent_wrapper(candidate_preparation_agent)
    )
    refine_evidence_node = _make_node(
        "refine_evidence", agent_wrapper(evidence_refinement_agent)
    )
    # Live DB grounding is a repair signal, not always-on context: the
    # value-repair node only runs after an empty execution result.
    value_repair_node = _make_node(
        "check_value_repair", agent_wrapper(EmptyResultValueRepairAgent())
    )
    # Optional proactive (pre-execution) checks — literal/value, join-path, and
    # JSONB key-path — merged into one node (CombinedPrecheckAgent) so a query
    # failing more than one of them costs a single reconstruction round-trip
    # instead of one per check. The node is only added if at least one of the
    # three flags is on; internally, each sub-check still only runs if its own
    # flag is enabled. Opt-in via DB_PROBE_PROACTIVE, DB_PROBE_JOIN_PATH_CHECK,
    # DB_PROBE_JSONB_PATH_CHECK.
    proactive_enabled = is_db_probe_proactive()
    join_path_enabled = is_db_probe_join_path_check()
    jsonb_path_enabled = is_db_probe_jsonb_path_check()
    logger.info(
        "Text-to-SQL graph: db-probe proactive=%s join_path=%s jsonb_path=%s",
        proactive_enabled,
        join_path_enabled,
        jsonb_path_enabled,
    )
    combined_precheck_enabled = (
        proactive_enabled or join_path_enabled or jsonb_path_enabled
    )
    combined_precheck_node = (
        _make_node("precheck_combined", agent_wrapper(CombinedPrecheckAgent()))
        if combined_precheck_enabled
        else None
    )
    construct_sql_from_candidates_node = _make_node(
        "construct_sql_from_candidates",
        agent_wrapper(sql_from_candidates_agent),
    )
    reconstruct_sql_node = _make_node(
        "reconstruct_sql", agent_wrapper(sql_reconstruction_agent)
    )

    validate_sql_query_node = _make_node(
        "validate_sql_query", agent_wrapper(sql_validation_agent)
    )
    execute_sql_query_node = _make_node(
        "execute_sql_query", agent_wrapper(sql_execution_agent)
    )
    check_empty_like_result_node = _make_node(
        "check_empty_like_result", agent_wrapper(empty_like_result_check_agent)
    )
    format_and_respond_node = _make_node(
        "format_and_respond", agent_wrapper(response_agent)
    )
    unconstructable_sql_response_node = _make_node(
        "unconstructable_sql_response", agent_wrapper(sql_unconstructable_agent)
    )

    # ==================== CREATE GRAPH ====================

    graph = StateGraph(AgentState)

    # -----------------    ENTRY POINT   ------------------
    graph.add_node("_entry_router", lambda state: state)
    graph.set_entry_point("_entry_router")
    graph.add_conditional_edges(
        "_entry_router",
        _entry_router_fn,
        {
            "question_intent": "question_intent",
            "reconstruct_sql": "reconstruct_sql",
        },
    )

    # Add only nodes instantiated above.
    graph.add_node("question_intent", question_intent_node)
    graph.add_node("question_extraction", question_extraction_node)
    graph.add_node("retrieve_candidates", retrieve_candidates_node)
    graph.add_node("information_agent", information_node)
    graph.add_node("prepare_candidates", prepare_candidates_node)
    graph.add_node("refine_evidence", refine_evidence_node)
    graph.add_node("check_value_repair", value_repair_node)
    if combined_precheck_node is not None:
        graph.add_node("precheck_combined", combined_precheck_node)
    graph.add_node("construct_sql_from_candidates", construct_sql_from_candidates_node)
    graph.add_node("reconstruct_sql", reconstruct_sql_node)
    graph.add_node("validate_sql_query", validate_sql_query_node)
    graph.add_node("execute_sql_query", execute_sql_query_node)
    graph.add_node("check_empty_like_result", check_empty_like_result_node)
    graph.add_node("format_and_respond", format_and_respond_node)
    graph.add_node("unconstructable_sql_response", unconstructable_sql_response_node)

    # Minimal flow using only the defined nodes.
    graph.add_edge("question_intent", "question_extraction")
    graph.add_edge("question_extraction", "retrieve_candidates")
    graph.add_conditional_edges(
        "retrieve_candidates",
        route_after_candidate_retrieval,
        {
            "information": "information_agent",
            "prepare_candidates": "prepare_candidates",
        },
    )
    graph.add_edge("information_agent", END)

    if prediction_enabled:
        # The intent node classified the question before extraction. Routing waits
        # until candidate preparation so KumoRFM still receives the relevant table
        # scope. The prediction path itself is two nodes:
        # ``prepare_prediction_graph`` (build the graph/model) →
        # ``kumo_predict`` (generate PQL + predict), so the slow build step streams
        # its own progress. Built only when KumoRFM is configured.
        graph.add_node(
            "prepare_prediction_graph",
            _make_node(
                "prepare_prediction_graph", agent_wrapper(PredictionGraphAgent())
            ),
        )
        graph.add_node(
            "kumo_predict",
            _make_node("kumo_predict", agent_wrapper(KumoPredictionAgent())),
        )

        graph.add_conditional_edges(
            "prepare_candidates",
            route_question_type_or_evidence,
            {
                "prediction": "prepare_prediction_graph",
                "refine_evidence": "refine_evidence",
                "construct_sql_from_candidates": "construct_sql_from_candidates",
            },
        )
        graph.add_conditional_edges(
            "prepare_prediction_graph",
            route_decision,
            {
                "predict_ready": "kumo_predict",
                "predict_failed": END,
            },
        )
        graph.add_edge("kumo_predict", END)
    else:
        graph.add_conditional_edges(
            "prepare_candidates",
            route_evidence_refinement,
            {
                "refine_evidence": "refine_evidence",
                "construct_sql_from_candidates": "construct_sql_from_candidates",
            },
        )

    graph.add_edge("refine_evidence", "construct_sql_from_candidates")
    graph.add_conditional_edges(
        "construct_sql_from_candidates",
        route_decision,
        {
            "validate_sql_query": "validate_sql_query",
            "unconstructable": "unconstructable_sql_response",
        },
    )

    # When any pre-execution probe check is enabled, every route that would
    # otherwise go straight to execution is funnelled through the merged
    # precheck node first (literal/value, join-path, and JSONB-path checks —
    # see combined_precheck.py for why they're combined and how their
    # internal ordering/gating works).
    pre_execute_target = (
        "precheck_combined"
        if combined_precheck_node is not None
        else "execute_sql_query"
    )

    # SQL validation → route
    graph.add_conditional_edges(
        "validate_sql_query",
        route_sql_validation,
        {
            "valid_sql": pre_execute_target,
            "invalid_sql": "reconstruct_sql",
            "unconstructable": "unconstructable_sql_response",
        },
    )

    if combined_precheck_node is not None:
        graph.add_conditional_edges(
            "precheck_combined",
            _make_soft_check_router("precheck_combined"),
            {
                "valid_sql": "execute_sql_query",
                "invalid_sql": "reconstruct_sql",
            },
        )

    # SQL execution → route (use route_sql_validation to enforce attempt limits)
    graph.add_conditional_edges(
        "execute_sql_query",
        route_sql_validation,
        {
            "valid_sql": "check_empty_like_result",
            "invalid_sql": "reconstruct_sql",
            "unconstructable": "unconstructable_sql_response",
        },
    )

    # After the empty-LIKE check, run the value-repair check (also gated on an
    # empty result at run time).
    graph.add_conditional_edges(
        "check_empty_like_result",
        _make_soft_check_router("check_empty_like_result"),
        {
            "valid_sql": "check_value_repair",
            "invalid_sql": "reconstruct_sql",
        },
    )
    graph.add_conditional_edges(
        "check_value_repair",
        _make_soft_check_router("check_value_repair"),
        {
            "valid_sql": "format_and_respond",
            "invalid_sql": "reconstruct_sql",
        },
    )

    graph.add_edge("reconstruct_sql", "validate_sql_query")

    graph.add_edge("unconstructable_sql_response", END)

    graph.add_edge("format_and_respond", END)

    return graph


__all__ = [
    "INTENT_VALIDATION_SKIPPED_AFTER",
    "NODE_START_EVENT",
    "TextToSQLPayload",
    "AgentState",
    "create_graph",
    "get_question_for_processing",
    "route_after_candidate_retrieval",
]
