# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import time
from datetime import datetime
from typing import Generator

from langchain_core.messages import HumanMessage, SystemMessage

from gsf.retrieval.text_to_sql.text_to_sql_graph import create_graph
from gsf.retrieval.text_to_sql.state import AgentState, TextToSQLPayload
from gsf.retrieval.text_to_sql.prompts import main_system_prompt_template
from gsf.retrieval.data_access.custom_analyses import fetch_custom_analyses
from gsf.utils.llm_invoke import get_llm_client

logger = logging.getLogger(__name__)

_ENTITY_MODEL = os.environ.get("ENTITY_EXTRACTION_MODEL")

try:
    llm_client = get_llm_client()
except ValueError as e:
    logger.error("Failed to initialize LLM client: %s", e)
    llm_client = None

entity_llm_client = None
if _ENTITY_MODEL:
    try:
        entity_llm_client = get_llm_client(model=_ENTITY_MODEL, max_tokens=512)
        logger.info("Entity extraction will use model: %s", _ENTITY_MODEL)
    except ValueError as e:
        logger.warning("Failed to init entity LLM (%s): %s", _ENTITY_MODEL, e)


graph = create_graph()
app = graph.compile()


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

    initial_path_state = dict(payload.get("path_state") or {})

    main_system_prompt = main_system_prompt_template.format(
        date=datetime.now(),
        custom_prompts=custom_prompts_text,
    )
    messages = [
        SystemMessage(content=main_system_prompt),
        HumanMessage(content=payload["question"]),
    ]

    state: dict = {
        "llm": llm_client,
        "initial_question": payload["question"],
        "connectors": connectors,
        "messages": messages,
        "path_state": initial_path_state,
        "data_retriever": data_retriever,
        "semantic_retriever": semantic_retriever,
        "decision": "",
        "domain_rules": domain_rules,
    }
    if entity_llm_client is not None:
        state["entity_llm"] = entity_llm_client
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


def stream_agent_response(
    payload: TextToSQLPayload,
) -> Generator[dict, None, None]:
    """Yield ``{"type": "step", "node": ...}`` for each graph node,
    then ``{"type": "result", "answer": ...}`` with the final answer.
    On error yields ``{"type": "error", "message": ...}``."""
    t0 = time.perf_counter()

    state = _build_state(payload)
    final_state = dict(state)

    try:
        for step in app.stream(state, config={"recursion_limit": 45}):
            logger.info("--- AGENT STEP ---")
            for node_name, node_output in step.items():
                logger.info("Node: %s", node_name)
                yield {"type": "step", "node": node_name}

                if node_output:
                    if "path_state" in node_output:
                        if "path_state" not in final_state:
                            final_state["path_state"] = {}
                        final_state["path_state"].update(node_output["path_state"])
                    for key, value in node_output.items():
                        if key != "path_state":
                            final_state[key] = value

        answer = _extract_answer(final_state)
        elapsed = time.perf_counter() - t0
        logger.info("Final answer (%.2fs):\n%s", elapsed, answer)
        yield {"type": "result", "answer": answer}

    except Exception as exc:
        logger.exception("Error during agent stream")
        yield {"type": "error", "message": f"Agent failed: {exc}"}


def get_agent_response(payload: TextToSQLPayload) -> dict:
    """Non-streaming convenience wrapper around ``stream_agent_response``."""
    for event in stream_agent_response(payload):
        if event["type"] == "result":
            return event["answer"]
        if event["type"] == "error":
            raise RuntimeError(event["message"])
    return {"response": "SQL can't be constructed.", "sql_code": "", "result": None}


__all__ = [
    "get_agent_response",
    "stream_agent_response",
    "app",
    "graph",
    "llm_client",
]
