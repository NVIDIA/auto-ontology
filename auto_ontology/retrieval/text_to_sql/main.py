# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from typing import Generator

from langchain_core.messages import HumanMessage

from auto_ontology.retrieval.text_to_sql.pipeline import create_steps, run_steps
from auto_ontology.retrieval.text_to_sql.connector_routing import (
    resolve_target_database_name,
)
from auto_ontology.retrieval.text_to_sql.node_labels import NODE_LABELS
from auto_ontology.retrieval.text_to_sql.react.agent import STEP_NAME as REACT_STEP
from auto_ontology.retrieval.text_to_sql.react.config import prediction_enabled
from auto_ontology.retrieval.text_to_sql.react.prompts import custom_prompts_section
from auto_ontology.retrieval.text_to_sql.state import (
    AgentState,
    TextToSQLPayload,
)
from auto_ontology.utils.llm_invoke import (
    get_llm_client,
    get_non_reasoning_llm_client,
)

logger = logging.getLogger(__name__)


class AgentRunError(RuntimeError):
    """Pipeline failure carrying the last node and recoverable partial answer."""

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

steps = create_steps()


def _build_state(payload: TextToSQLPayload) -> AgentState:
    connectors = payload.get("connectors", [])
    if not connectors:
        raise ValueError(
            "TextToSQLPayload is missing required 'connectors'. "
            "Provide a non-empty list of database connectors, each with a valid 'dialect' attribute."
        )

    data_retriever = payload.get("data_retriever")
    if data_retriever is None:
        raise ValueError(
            "TextToSQLPayload is missing required 'data_retriever' (auto_ontology.utils.retriever.Retriever "
            "instance). Construct a Retriever once at startup and pass it in the payload."
        )
    semantic_retriever = payload.get("semantic_retriever")
    if semantic_retriever is None:
        logger.warning(
            "No 'semantic_retriever' in payload — "
            "ColumnAttribute, CustomAnalysis, and SqlAttribute searches will be skipped."
        )

    prediction_override = payload.get("prediction")
    if prediction_override is True and not prediction_enabled():
        raise ValueError(
            "prediction=true was requested but the prediction flow is not "
            "configured on this deployment (KUMO_RFM_API_URL is unset)."
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
    initial_path_state["processing_question"] = processing_question

    state: dict = {
        "llm": llm_client,
        "reasoning_llm": reasoning_llm_client,
        "initial_question": submitted_question,
        "evidence": payload.get("evidence") or "",
        "numeric_type_alignment": payload.get("numeric_type_alignment", False),
        "connectors": connectors,
        "messages": [HumanMessage(content=processing_question)],
        "path_state": initial_path_state,
        "data_retriever": data_retriever,
        "semantic_retriever": semantic_retriever,
        "decision": "",
        "prediction_override": prediction_override,
        "custom_prompts": custom_prompts_section(
            payload.get("custom_prompts"), payload.get("acronyms")
        ),
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


def _submitted_sql(step_name: str, step_output: dict, step_path_state: dict) -> str:
    """The SQL the agent just submitted as its answer, or ``""``."""
    if step_name != REACT_STEP or (step_output or {}).get("decision") != "valid_sql":
        return ""
    return (step_path_state.get("sql_code") or "").strip()


def _merge_step_output(final_state: dict, step_output: dict | None) -> None:
    """Fold one step's output into the accumulated state, in place.

    ``path_state`` is merged key-by-key (nodes only ever return the subset
    they touched); every other top-level key is overwritten outright.
    """
    if not step_output:
        return
    if "path_state" in step_output:
        final_state.setdefault("path_state", {})
        final_state["path_state"].update(step_output["path_state"])
    for key, value in step_output.items():
        if key != "path_state":
            final_state[key] = value


def _build_thoughts_summary(thoughts_log: list[dict]) -> str:
    """Concatenate the run's per-node thought entries into one summary string.

    Deterministic (no extra LLM call): one bullet per entry, labelled with the
    same human-readable name the live step events use, in the order they were
    recorded.
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
    node — ``"start"`` as it begins (so a client can label the work in
    progress) and ``"end"`` when it returns, carrying its ``thought`` — plus
    ``{"type": "sql", "node": ..., "sql": ...}`` once the agent has submitted
    its query, then ``{"type": "result", "answer": ...}`` with the final answer
    (its ``thoughts`` key summarizes every ``thought`` collected along the way).
    On error yields ``{"type": "error", "message": ...}``."""
    t0 = time.perf_counter()

    logger.info("Text-to-SQL agent started for question: %s", payload["question"])

    state = _build_state(payload)
    final_state = dict(state)
    last_step: str | None = None
    try:
        for phase, step_name, step_output in run_steps(dict(state), steps):
            # A node that raises never reaches "end", so the start is what
            # names the node that failed.
            last_step = step_name
            if phase == "start":
                yield {
                    "type": "step",
                    "phase": "start",
                    "node": step_name,
                    "thought": None,
                }
                continue

            logger.info("--- AGENT STEP ---")
            logger.info("Step: %s", step_name)

            # A node records its thoughts at the tail of
            # path_state["thoughts_log"] — see BaseAgent.record_thought.
            # Only surface the last one when this node added it, so a step
            # event never shows a stale entry left by an earlier node.
            thought = None
            step_path_state = (step_output or {}).get("path_state") or {}
            thoughts_log = step_path_state.get("thoughts_log") or []
            if thoughts_log and thoughts_log[-1].get("node") == step_name:
                thought = thoughts_log[-1].get("text")

            yield {
                "type": "step",
                "phase": "end",
                "node": step_name,
                "thought": thought,
            }

            step_sql = _submitted_sql(step_name, step_output, step_path_state)
            if step_sql:
                yield {"type": "sql", "node": step_name, "sql": step_sql}

            _merge_step_output(final_state, step_output)

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
            "message": f"Agent failed after {last_step or 'start'}: {exc}",
            "node": last_step,
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
        for phase, _step_name, step_output in run_steps(dict(state), steps):
            if phase == "end":
                _merge_step_output(final_state, step_output)
    except Exception as exc:
        logger.exception("Error during agent stream in get_agent_response_with_state")
        # Keep whatever SQL the agent had already submitted when a later node
        # raised, rather than discarding it and returning blank SQL.
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
    "steps",
    "llm_client",
    "reasoning_llm_client",
    "AgentRunError",
]
