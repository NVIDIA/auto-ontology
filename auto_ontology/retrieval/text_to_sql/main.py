# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from datetime import datetime
from typing import Generator, cast

from langchain_core.messages import HumanMessage, SystemMessage

from auto_ontology.retrieval.text_to_sql.text_to_sql_graph import (
    NODE_START_EVENT,
    _prediction_enabled,
    create_graph,
)
from auto_ontology.retrieval.text_to_sql.connector_routing import (
    resolve_target_database_name,
)
from auto_ontology.retrieval.text_to_sql.node_labels import NODE_LABELS
from auto_ontology.retrieval.text_to_sql.state import (
    AgentState,
    RetryRestoreState,
    TextToSQLPayload,
)
from auto_ontology.retrieval.text_to_sql.prompts import main_system_prompt_template
from auto_ontology.utils.llm_invoke import (
    get_llm_client,
    get_non_reasoning_llm_client,
)

logger = logging.getLogger(__name__)


class AgentRunError(RuntimeError):
    """Graph failure carrying the last node and recoverable partial answer."""

    def __init__(
        self,
        message: str,
        *,
        node: str | None = None,
        partial_answer: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.node = node
        self.partial_answer = partial_answer or {}


try:
    llm_client = get_non_reasoning_llm_client()
except (ValueError, EnvironmentError) as e:
    logger.error("Failed to initialize non-reasoning LLM client: %s", e)
    llm_client = None

try:
    reasoning_llm_client = get_llm_client()
except (ValueError, EnvironmentError) as e:
    logger.error("Failed to initialize reasoning LLM client: %s", e)
    reasoning_llm_client = None

graph = create_graph()
app = graph.compile()

# Whether the combined precheck sits between validation and execution.
# Read off the graph that was actually built rather than re-reading
# the probe flags: ``create_graph`` evaluates them once at import, so a later
# change would leave the two disagreeing about which node is the last gate
# before execution. See ``_sql_about_to_run``.
_COMBINED_PRECHECK_IN_GRAPH = "precheck_combined" in graph.nodes
_TRANSPARENT_NODES = frozenset({"_entry_router"})
_MAX_FULL_PIPELINE_ATTEMPTS = 2
_UNCONSTRUCTABLE_NODE = "unconstructable_sql_response"

# Keys created or consumed within one question-to-SQL attempt. A full retry
# starts before intent classification, so none of these may leak into it from
# an interactive resume seed or the failed attempt.
_FULL_RETRY_PATH_KEYS_TO_CLEAR = frozenset(
    {
        "_resume_from",
        "normalized_question",
        "question_type",
        "calculation_subtype",
        "sql_template",
        "extracted_evidence",
        "entities",
        "subject",
        "retrieved_column_attributes",
        "retrieved_custom_analyses",
        "retrieved_sql_attributes",
        "retrieved_subject_term",
        "candidates",
        "relevant_tables",
        "relevant_queries",
        "similar_questions",
        "custom_analyses",
        "custom_analyses_str",
        "sql_attributes",
        "sql_attributes_str",
        "primary_attribute",
        "attribute_join_paths",
        "table_relevance_reasoning",
        "evidence_repairs_applied",
        "sql_code",
        "sql_generation_result",
        "error",
        "failed_attempts",
        "prior_round_failed_attempts",
        "interpretation_history",
        "error_type",
        "error_analysis_done",
        "error_known_fixable",
        "last_intent_rejected_sql",
        "returned_deterministic_fallback",
        "unconstructable_explanation",
        "final_response",
        "formatted_response",
        "sql_response_from_db",
        "sql_columns",
        "sql_tables",
        "custom_analyses_used",
        "sql_attempts",
        "reconstruction_count",
        "jsonb_path_repair_attempts",
        "join_path_repair_attempts",
        "value_repair_attempted",
        "null_jsonb_retry_attempted",
        "empty_like_retry_attempted",
        "thoughts_log",
        "node_visit_counts",
    }
)


def _build_state(payload: TextToSQLPayload) -> AgentState:
    custom_prompts = payload.get("custom_prompts", "")
    acronyms = payload.get("acronyms", [])
    connectors = payload.get("connectors", [])
    if not connectors:
        raise ValueError(
            "TextToSQLPayload is missing required 'connectors'. "
            "Provide a non-empty list of database connectors, each with a valid 'dialect' attribute."
        )

    data_retriever = payload.get("data_retriever")
    if data_retriever is None:
        raise ValueError(
            "TextToSQLPayload is missing required 'data_retriever' (nemo_retriever.retriever.Retriever "
            "instance). Construct a Retriever once at startup and pass it in the payload."
        )
    semantic_retriever = payload.get("semantic_retriever")
    if semantic_retriever is None:
        logger.warning(
            "No 'semantic_retriever' in payload — "
            "ColumnAttribute, CustomAnalysis, and SqlAttribute searches will be skipped."
        )

    custom_prompts_text = f"{custom_prompts}\n\n" if custom_prompts else ""
    domain_rules = list(acronyms or [])

    calculation_only = bool(payload.get("calculation_only", False))

    # ``prediction=True`` only means something when the KumoRFM branch was built
    # into the graph at startup; without KUMO_RFM_API_KEY the classify node does
    # not exist, so honouring the override is impossible. Fail loudly rather than
    # silently answering with SQL. Calculation-only questions intentionally
    # ignore this separate interactive-mode selector.
    prediction_override = None if calculation_only else payload.get("prediction")
    if prediction_override is True and not _prediction_enabled():
        raise ValueError(
            "prediction=true was requested but the prediction flow is not "
            "configured on this deployment (KUMO_RFM_API_KEY is unset)."
        )

    initial_path_state = dict(payload.get("path_state") or {})

    target_db = payload.get("target_db")
    if target_db:
        initial_path_state["target_db"] = resolve_target_database_name(
            target_db, connectors
        )
    elif len(connectors) == 1:
        connector_db = getattr(connectors[0], "database_name", None)
        if connector_db:
            initial_path_state["target_db"] = connector_db

    submitted_question = payload["question"].strip()
    processing_question = (
        payload.get("processing_question") or ""
    ).strip() or submitted_question
    # Keep the exact submitted turn separate from a standalone follow-up rewrite.
    # Question extraction may replace normalized_question later, while intent
    # validation must continue to see both representations.
    initial_path_state["processing_question"] = processing_question

    main_system_prompt = main_system_prompt_template.format(
        date=datetime.now(),
        custom_prompts=custom_prompts_text,
    )
    messages = [
        SystemMessage(content=main_system_prompt),
        HumanMessage(content=processing_question),
    ]
    initial_evidence = payload.get("evidence") or ""
    initial_sql_examples = list(payload.get("sql_examples") or [])
    initial_value_anchors = list(payload.get("value_anchors") or [])
    initial_glossary = list(acronyms or [])
    restore_from: RetryRestoreState = {
        "initial_question": submitted_question,
        "evidence": initial_evidence,
        "messages": list(messages),
        "sql_examples": list(initial_sql_examples),
        "value_anchors": list(initial_value_anchors),
        "glossary": list(initial_glossary),
        "path_state": dict(initial_path_state),
    }

    state: dict = {
        "llm": llm_client,
        "reasoning_llm": reasoning_llm_client,
        "initial_question": submitted_question,
        "evidence": initial_evidence,
        "sql_examples": initial_sql_examples,
        "value_anchors": initial_value_anchors,
        "calculation_only": calculation_only,
        "shorten_answer": payload.get("shorten_answer", False),
        "validate_sql_values": payload.get("validate_sql_values", False),
        "sql_value_validation_cache": {},
        "enriched_question": payload.get("enriched_question") or "",
        "connectors": connectors,
        "messages": messages,
        "path_state": initial_path_state,
        "data_retriever": data_retriever,
        "semantic_retriever": semantic_retriever,
        "decision": "",
        "domain_rules": domain_rules,
        "glossary": initial_glossary,
        "prediction_override": prediction_override,
        "restore_from": restore_from,
        "full_pipeline_attempt": 1,
    }
    return state


def _extract_answer(final_state: dict) -> dict:
    path_state = final_state.get("path_state", {})
    final_response = path_state.get("final_response")

    if final_response is None:
        messages_out = final_state.get("messages") or []
        if isinstance(messages_out, list) and messages_out:
            final_response = messages_out[-1]
        else:
            final_response = ""

    if isinstance(final_response, dict):
        return final_response
    return {"response": str(final_response)}


def _restore_for_full_pipeline_retry(state: AgentState) -> AgentState:
    """Rebuild one attempt from the immutable request boundary."""
    restore_from = state["restore_from"]
    restored_path_state = dict(restore_from["path_state"])
    for key in _FULL_RETRY_PATH_KEYS_TO_CLEAR:
        restored_path_state.pop(key, None)

    restored = dict(state)
    restored.update(
        {
            "initial_question": restore_from["initial_question"],
            "evidence": restore_from["evidence"],
            "messages": list(restore_from["messages"]),
            "sql_examples": list(restore_from["sql_examples"]),
            "value_anchors": list(restore_from["value_anchors"]),
            "glossary": list(restore_from["glossary"]),
            "path_state": restored_path_state,
            "decision": "",
            "full_pipeline_attempt": state.get("full_pipeline_attempt", 1) + 1,
        }
    )
    return cast(AgentState, restored)


def _should_retry_full_pipeline(
    final_state: AgentState,
    terminal_node: str | None,
) -> bool:
    """Whether an unconstructable SQL attempt should restart from intent."""
    if terminal_node != _UNCONSTRUCTABLE_NODE:
        return False
    if "restore_from" not in final_state:
        return False
    if final_state.get("full_pipeline_attempt", 1) >= _MAX_FULL_PIPELINE_ATTEMPTS:
        return False
    return not str(_extract_answer(final_state).get("sql_code") or "").strip()


def _apply_last_deterministic_sql_fallback(
    final_state: dict,
    terminal_node: str | None,
) -> bool:
    """Return the last static-valid SQL after the final intent retry is exhausted."""
    if (
        terminal_node != _UNCONSTRUCTABLE_NODE
        or final_state.get("full_pipeline_attempt", 1) < _MAX_FULL_PIPELINE_ATTEMPTS
    ):
        return False

    path_state = final_state.get("path_state") or {}
    fallback_sql = str(path_state.get("last_intent_rejected_sql") or "").strip()
    if (
        not fallback_sql
        or str(_extract_answer(final_state).get("sql_code") or "").strip()
    ):
        return False

    path_state["sql_code"] = fallback_sql
    path_state["returned_deterministic_fallback"] = True
    final_response = path_state.get("final_response")
    if isinstance(final_response, dict):
        final_response = dict(final_response)
        final_response["sql_code"] = fallback_sql
    else:
        final_response = {
            "response": str(final_response or "SQL intent validation was exhausted."),
            "sql_code": fallback_sql,
        }
    path_state["final_response"] = final_response
    final_state["path_state"] = path_state
    logger.warning(
        "Returning the last deterministically valid SQL after two full-pipeline "
        "attempts exhausted intent validation"
    )
    return True


def _sql_about_to_run(node_name: str, node_output: dict, node_path_state: dict) -> str:
    """The SQL this node just cleared for execution, or ``""``.

    Deliberately not emitted at generation time: unified validation routinely
    sends a query back for reconstruction, so a draft is frequently not what
    runs. The consequence is that a run which never clears its final gate
    (``unconstructable`` after 8 attempts) shows no SQL at all.

    Which node *is* the final gate is decided by ``create_graph`` and is not
    visible here: the proactive value check, when built in, sits after intent
    validation and can still bounce a query to reconstruction.
    """
    decision = (node_output or {}).get("decision") or ""

    if _COMBINED_PRECHECK_IN_GRAPH:
        cleared = node_name == "precheck_combined" and decision == "valid_sql"
    else:
        cleared = node_name == "validate_sql_query" and decision == "valid_sql"
    if not cleared:
        return ""

    # ``sql_code`` is what ``SQLExecutionAgent`` runs; keep the generation
    # result as a defensive fallback for partial/mocked node updates.
    sql = (node_path_state.get("sql_code") or "").strip()
    if sql:
        return sql
    generated = node_path_state.get("sql_generation_result")
    return (getattr(generated, "sql_code", "") or "").strip()


def _merge_node_output(final_state: dict, node_output: dict | None) -> None:
    """Fold one graph node's output into the accumulated state, in place.

    ``path_state`` is merged key-by-key (nodes only ever return the subset
    they touched); every other top-level key is overwritten outright.
    """
    if not node_output:
        return
    if "path_state" in node_output:
        final_state.setdefault("path_state", {})
        final_state["path_state"].update(node_output["path_state"])
    for key, value in node_output.items():
        if key != "path_state":
            final_state[key] = value


def _build_thoughts_summary(thoughts_log: list[dict]) -> str:
    """Concatenate the run's per-node thought entries into one summary string.

    Deterministic (no extra LLM call): one bullet per entry, labelled with the
    same human-readable name the live step events use, in the order the nodes
    actually ran (a node visited more than once — e.g. during reconstruction
    retries — contributes one bullet per visit).
    """
    lines = [
        f"- {NODE_LABELS.get(entry['node'], entry['node'])}: {entry['text']}"
        for entry in thoughts_log
        if entry.get("text")
    ]
    return "\n".join(lines)


def _extract_partial_answer(final_state: dict) -> dict:
    """Return generated SQL/response that existed before a downstream failure."""
    path_state = final_state.get("path_state", {})
    generation = path_state.get("sql_generation_result")
    sql_code = path_state.get("sql_code") or getattr(generation, "sql_code", "")
    response = getattr(generation, "response", "")
    thought = getattr(generation, "thought", "")
    if not sql_code and not response:
        return {}
    return {
        "sql_code": str(sql_code or ""),
        "response": str(response or ""),
        "thought": str(thought or ""),
    }


def stream_agent_response(
    payload: TextToSQLPayload,
) -> Generator[dict, None, None]:
    """Yield two ``{"type": "step", "node": ..., "phase": ...}`` events per
    graph node — ``"start"`` as it begins (so a client can label the work in
    progress) and ``"end"`` when it returns, carrying its ``thought`` — plus
    ``{"type": "sql", "node": ..., "sql": ...}`` once a query has cleared
    validation and is about to run (see ``_sql_about_to_run``), then
    ``{"type": "result", "answer": ...}`` with the final answer (its
    ``thoughts`` key summarizes every ``thought`` collected along the way).
    On error yields ``{"type": "error", "message": ...}``."""
    t0 = time.perf_counter()

    logger.info("Text-to-SQL agent started for question: %s", payload["question"])

    state = _build_state(payload)
    final_state = dict(state)
    # Last SQL surfaced to the client. A query can clear its final gate more
    # than once (an empty result sends it back through validation unchanged),
    # so dedupe rather than re-emitting the same query.
    streamed_sql: str | None = None

    last_node: str | None = None
    try:
        while True:
            final_state = dict(state)
            terminal_node: str | None = None
            # ``custom`` payloads stream the instant a node writes one (as it
            # begins); ``updates`` only arrive once it has returned. Reading
            # updates alone would label the screen with the previously finished
            # node, so a slow reconstruction looks like a hung validation.
            for mode, chunk in app.stream(
                state,
                stream_mode=["updates", "custom"],
                config={"recursion_limit": 45},
            ):
                if mode == "custom":
                    if (chunk or {}).get("type") == NODE_START_EVENT:
                        started = chunk.get("node")
                        if started and started not in _TRANSPARENT_NODES:
                            # A node that raises produces no update, so tracking
                            # completions alone would blame the node before it.
                            last_node = started
                            yield {
                                "type": "step",
                                "phase": "start",
                                "node": started,
                                "thought": None,
                            }
                    continue

                logger.info("--- AGENT STEP ---")
                for node_name, node_output in chunk.items():
                    if node_name in _TRANSPARENT_NODES:
                        _merge_node_output(final_state, node_output)
                        continue

                    last_node = node_name
                    terminal_node = node_name
                    logger.info("Node: %s", node_name)

                    # A node records its own thought (if any) at the tail of
                    # path_state["thoughts_log"] — see BaseAgent.record_thought.
                    # Only surface it here when this node is the one that just
                    # added it, so a step event never shows a stale entry left
                    # over from an earlier node.
                    thought = None
                    node_path_state = (node_output or {}).get("path_state") or {}
                    thoughts_log = node_path_state.get("thoughts_log") or []
                    if thoughts_log and thoughts_log[-1].get("node") == node_name:
                        thought = thoughts_log[-1].get("text")

                    # Only place a thought can be attached: the node has to finish
                    # before it has one to report.
                    yield {
                        "type": "step",
                        "phase": "end",
                        "node": node_name,
                        "thought": thought,
                    }

                    # Surface the SQL once a node has cleared it for execution,
                    # so it is on screen while the database runs it rather than
                    # only landing with the final answer. Drafts that validation
                    # is about to send back for reconstruction are deliberately
                    # not shown — see ``_sql_about_to_run``.
                    node_sql = _sql_about_to_run(
                        node_name, node_output, node_path_state
                    )
                    if node_sql and node_sql != streamed_sql:
                        streamed_sql = node_sql
                        yield {"type": "sql", "node": node_name, "sql": node_sql}

                    _merge_node_output(final_state, node_output)

            if not _should_retry_full_pipeline(final_state, terminal_node):
                break
            logger.warning(
                "Full SQL pipeline attempt %d returned no SQL; restarting from "
                "question intent.",
                final_state.get("full_pipeline_attempt", 1),
            )
            state = _restore_for_full_pipeline_retry(final_state)
            streamed_sql = None
            last_node = None

        _apply_last_deterministic_sql_fallback(final_state, terminal_node)
        answer = _extract_answer(final_state)
        thoughts_log = final_state.get("path_state", {}).get("thoughts_log") or []
        thoughts_summary = _build_thoughts_summary(thoughts_log)
        if isinstance(answer, dict) and thoughts_summary:
            answer["thoughts"] = thoughts_summary
        elapsed = time.perf_counter() - t0
        logger.debug("Final answer (%.2fs):\n%s", elapsed, answer)
        yield {"type": "result", "answer": answer}

    except Exception as exc:
        logger.exception("Error during agent stream")
        yield {
            "type": "error",
            "message": f"Agent failed after {last_node or 'graph_start'}: {exc}",
            "node": last_node,
            "error_type": type(exc).__name__,
            "partial_answer": _extract_partial_answer(final_state),
        }


def get_agent_response(payload: TextToSQLPayload) -> dict:
    """Non-streaming convenience wrapper around ``stream_agent_response``."""
    for event in stream_agent_response(payload):
        if event["type"] == "result":
            return event["answer"]
        if event["type"] == "error":
            raise AgentRunError(
                event["message"],
                node=event.get("node"),
                partial_answer=event.get("partial_answer"),
            )
    return {"response": "SQL can't be constructed.", "sql_code": "", "result": None}


def get_agent_response_with_state(payload: TextToSQLPayload) -> dict:
    """Like get_agent_response but also returns path_state in the result under key 'path_state'."""
    # Required by auto_ontology/retrieval/interactive/coordinator.py to persist path_state across turns/phases.
    state = _build_state(payload)
    final_state = dict(state)

    try:
        while True:
            final_state = dict(state)
            terminal_node: str | None = None
            for step in app.stream(state, config={"recursion_limit": 45}):
                for node_name, node_output in step.items():
                    if node_name not in _TRANSPARENT_NODES:
                        terminal_node = node_name
                    _merge_node_output(final_state, node_output)
            if not _should_retry_full_pipeline(final_state, terminal_node):
                break
            logger.warning(
                "Full SQL pipeline attempt %d returned no SQL; restarting from "
                "question intent.",
                final_state.get("full_pipeline_attempt", 1),
            )
            state = _restore_for_full_pipeline_retry(final_state)
        _apply_last_deterministic_sql_fallback(final_state, terminal_node)
    except Exception as exc:
        logger.exception("Error during agent stream in get_agent_response_with_state")
        # The stream may have already produced a valid, executed SQL query
        # (e.g. several reconstruction rounds succeeded) before a later node
        # raised — most commonly GraphRecursionError from an intent-validation
        # <-> reconstruction oscillation. Fall back to whatever SQL is already
        # sitting in path_state instead of discarding it and submitting blank
        # SQL, which is a guaranteed Phase 1 failure even when the last known
        # SQL was correct.
        interrupted_path_state = final_state.get("path_state") or {}
        fallback_sql = interrupted_path_state.get("sql_code", "") or ""
        return {
            "response": f"Agent failed: {exc}",
            "sql_code": fallback_sql,
            "path_state": interrupted_path_state,
        }

    # merge path_state: start with initial, overlay final accumulated
    merged_path_state = dict(state.get("path_state") or {})
    if "path_state" in final_state:
        merged_path_state.update(final_state["path_state"])

    answer = _extract_answer(final_state)
    if isinstance(answer, dict):
        result = dict(answer)
    else:
        result = {"response": str(answer)}
    result["path_state"] = merged_path_state
    return result


__all__ = [
    "get_agent_response",
    "get_agent_response_with_state",
    "stream_agent_response",
    "app",
    "graph",
    "llm_client",
    "reasoning_llm_client",
    "AgentRunError",
]
