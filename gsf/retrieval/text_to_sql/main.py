# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from datetime import datetime
from typing import Generator

from langchain_core.messages import HumanMessage, SystemMessage

from gsf.retrieval.text_to_sql.text_to_sql_graph import (
    INTENT_VALIDATION_SKIPPED_AFTER,
    NODE_START_EVENT,
    _prediction_enabled,
    create_graph,
)
from gsf.retrieval.text_to_sql.connector_routing import (
    resolve_target_database_name,
)
from gsf.retrieval.text_to_sql.node_labels import NODE_LABELS
from gsf.retrieval.text_to_sql.state import AgentState, TextToSQLPayload
from gsf.retrieval.text_to_sql.prompts import main_system_prompt_template
from gsf.retrieval.data_access.custom_analyses import fetch_custom_analyses
from gsf.utils.llm_invoke import get_llm_client

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
    llm_client = get_llm_client()
except ValueError as e:
    logger.error("Failed to initialize LLM client: %s", e)
    llm_client = None

graph = create_graph()
app = graph.compile()

# Whether the proactive value check sits between validation and execution.
# Read off the graph that was actually built rather than re-reading
# ``DB_PROBE_PROACTIVE``: ``create_graph`` evaluates that env var once at
# import, so a later change to it would leave the two disagreeing about which
# node is the last gate before execution. See ``_sql_about_to_run``.
_PROACTIVE_VALUE_CHECK_IN_GRAPH = "precheck_value_repair" in graph.nodes


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
    domain_rules = fetch_custom_analyses() + list(acronyms or [])

    # ``prediction=True`` only means something when the KumoRFM branch was built
    # into the graph at startup; without KUMO_RFM_API_KEY the classify node does
    # not exist, so honouring the override is impossible. Fail loudly rather than
    # silently answering with SQL.
    prediction_override = payload.get("prediction")
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

    processing_question = (
        payload.get("processing_question") or payload["question"]
    ).strip()

    main_system_prompt = main_system_prompt_template.format(
        date=datetime.now(),
        custom_prompts=custom_prompts_text,
    )
    messages = [
        SystemMessage(content=main_system_prompt),
        HumanMessage(content=processing_question),
    ]

    state: dict = {
        "llm": llm_client,
        "initial_question": processing_question,
        "connectors": connectors,
        "messages": messages,
        "path_state": initial_path_state,
        "data_retriever": data_retriever,
        "semantic_retriever": semantic_retriever,
        "decision": "",
        "domain_rules": domain_rules,
        "glossary": list(acronyms or []),
        "prediction_override": prediction_override,
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


def _sql_about_to_run(node_name: str, node_output: dict, node_path_state: dict) -> str:
    """The SQL this node just cleared for execution, or ``""``.

    Deliberately not emitted at generation time: both validations routinely
    send a query back for reconstruction, so a draft is frequently not what
    runs. The consequence is that a run which never clears its final gate
    (``unconstructable`` after 8 attempts) shows no SQL at all.

    Which node *is* the final gate is decided by ``create_graph`` and is not
    visible here: the proactive value check, when built in, sits after intent
    validation and can still bounce a query to reconstruction, and past
    ``INTENT_VALIDATION_SKIPPED_AFTER`` reconstructions intent validation is
    skipped entirely.
    """
    decision = (node_output or {}).get("decision") or ""

    if _PROACTIVE_VALUE_CHECK_IN_GRAPH:
        cleared = node_name == "precheck_value_repair" and decision == "valid_sql"
    else:
        cleared = decision == "intent_valid" or (
            node_name == "validate_sql_query"
            and decision == "valid_sql"
            and (node_path_state.get("reconstruction_count") or 0)
            > INTENT_VALIDATION_SKIPPED_AFTER
        )
    if not cleared:
        return ""

    # ``sql_code`` is what ``SQLExecutionAgent`` runs; the generation result
    # only covers intent validation's early return when there is no SQL.
    sql = (node_path_state.get("sql_code") or "").strip()
    if sql:
        return sql
    generated = node_path_state.get("sql_generation_result")
    return (getattr(generated, "sql_code", "") or "").strip()


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
                    if started:
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
                last_node = node_name
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
                node_sql = _sql_about_to_run(node_name, node_output, node_path_state)
                if node_sql and node_sql != streamed_sql:
                    streamed_sql = node_sql
                    yield {"type": "sql", "node": node_name, "sql": node_sql}

                if node_output:
                    if "path_state" in node_output:
                        if "path_state" not in final_state:
                            final_state["path_state"] = {}
                        final_state["path_state"].update(node_output["path_state"])
                    for key, value in node_output.items():
                        if key != "path_state":
                            final_state[key] = value

        answer = _extract_answer(final_state)
        thoughts_log = final_state.get("path_state", {}).get("thoughts_log") or []
        thoughts_summary = _build_thoughts_summary(thoughts_log)
        if isinstance(answer, dict) and thoughts_summary:
            answer["thoughts"] = thoughts_summary
        elapsed = time.perf_counter() - t0
        logger.info("Final answer (%.2fs):\n%s", elapsed, answer)
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


__all__ = [
    "get_agent_response",
    "stream_agent_response",
    "app",
    "graph",
    "llm_client",
    "AgentRunError",
]
