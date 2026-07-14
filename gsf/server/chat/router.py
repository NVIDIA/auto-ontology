# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat streaming endpoint — wraps the LangGraph text-to-SQL pipeline.

Each chat request runs in a prewarmed agent subprocess from
``WarmPool``. See ``worker.py`` for why subprocess isolation is
necessary (cancellation of sync, C-blocked agent code) and how the warm
pool keeps the cold-start cost off the request path.

Concurrency model
-----------------
The agent pipeline is not safe to run in parallel (shared retriever/connector
state, single LLM rate budget), so only one stream is allowed at a time.

* If the slot is empty → request runs.
* If the slot is held by a stream whose **client is still connected**
  (typical case: a second browser window/tab posting concurrently) →
  the new request is rejected with HTTP 409.
* If the slot is held by an **orphaned** stream (its client navigated away
  and the TCP connection died) → the new request preempts: the orphan
  worker is killed-and-replaced via the pool, and the new request takes
  the standby. This is what makes "navigate away → come back → ask again"
  work without a 409.

Disconnect detection
--------------------
A relying-on-write-failure ("the next heartbeat will fail") detection is too
slow in practice — the kernel buffers small writes and disconnect-triggered
GeneratorExit only fires when the response object is garbage-collected,
which can be tens of seconds. We instead spawn a per-request async watchdog
that polls ASGI ``http.disconnect`` and clears ``client_alive`` within one
poll interval (~100 ms), freeing the slot for any follow-up request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass
from typing import Generator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from gsf.server.chat.helpers import NODE_LABELS, ChatRequest, ChatRequestWithEvidence
from gsf.server.chat.worker import PrewarmedWorker, get_pool

logger = logging.getLogger(__name__)

router = APIRouter()

# How often the watchdog polls the ASGI disconnect signal. Small enough that
# a navigate-away → come-back-and-ask flow always finds the slot free.
_DISCONNECT_POLL_S = 0.1


@dataclass
class _Slot:
    """In-flight stream descriptor stored in ``_active_slot``."""

    worker: PrewarmedWorker
    client_alive: threading.Event
    # Set if the watchdog detected a client disconnect. Tells the stream's
    # finally whether to ``replace`` the worker (cancel) or ``return_alive``
    # it to the warm pool (natural completion).
    cancelled: threading.Event
    # Latches the moment any path (watchdog or stream finally) has called
    # back into the pool, so the second path doesn't double-release.
    released: threading.Event


_active_slot: _Slot | None = None
_slot_lock = threading.Lock()


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _try_claim_slot(new_slot: _Slot) -> _Slot | None:
    """Install ``new_slot`` if the previous one's client is gone.

    Returns the displaced slot (caller must replace its worker via the pool)
    on success. Raises HTTPException(409) if the previous slot still has a
    live client.
    """

    global _active_slot
    with _slot_lock:
        prev = _active_slot
        if prev is not None and prev.client_alive.is_set():
            raise HTTPException(
                status_code=409,
                detail="Conversation in progress",
            )
        _active_slot = new_slot
        return prev


def _release(slot: _Slot) -> None:
    """Free the slot and hand the worker back to the pool.

    Idempotent. ``slot.cancelled`` decides whether the worker is killed
    (cancel) or returned alive (natural end). The first caller wins; the
    second is a no-op via ``slot.released``.
    """

    if slot.released.is_set():
        return
    slot.released.set()
    slot.client_alive.clear()

    global _active_slot
    with _slot_lock:
        if _active_slot is slot:
            _active_slot = None

    pool = get_pool()
    if slot.cancelled.is_set():
        pool.replace(slot.worker)
    else:
        pool.return_alive(slot.worker)


async def _watch_disconnect(http_request: Request, slot: _Slot) -> None:
    """Free the slot the moment the ASGI client disconnects.

    Without this, the sync streaming generator only learns the client is
    gone when its next write to the socket fails — and the kernel happily
    buffers small heartbeat writes, so that signal can lag by tens of
    seconds, holding the slot and forcing follow-up requests into 409.
    """

    while slot.client_alive.is_set():
        try:
            disconnected = await http_request.is_disconnected()
        except Exception:  # noqa: BLE001 — defensive; never want to leak
            logger.exception("Disconnect watchdog failed")
            return
        if disconnected:
            slot.cancelled.set()
            _release(slot)
            return
        await asyncio.sleep(_DISCONNECT_POLL_S)


def _stream_chat(worker: PrewarmedWorker) -> Generator[str, None, None]:
    try:
        for item in worker.events():
            if item is None:
                # SSE comment — keeps the connection alive against any
                # intermediary (proxy/load balancer) that drops idle
                # streams. Clients ignore non-"data:" lines.
                yield ": ping\n\n"
                continue
            event = item
            if event.get("type") == "step":
                node_name = event.get("node", "")
                event = {**event, "label": NODE_LABELS.get(node_name, node_name)}
            yield _sse(event)
    except RuntimeError as exc:
        logger.exception("Agent stream failed")
        yield _sse({"type": "error", "message": f"Agent stream failed: {exc}"})

    yield "data: [DONE]\n\n"


def _stream_with_slot(slot: _Slot) -> Generator[str, None, None]:
    try:
        yield from _stream_chat(slot.worker)
    finally:
        _release(slot)


@router.post("/chat/completions")
async def chat_completions(
    request: ChatRequest, http_request: Request
) -> StreamingResponse:
    pool = get_pool()
    worker = pool.acquire()
    slot = _Slot(
        worker=worker,
        client_alive=threading.Event(),
        cancelled=threading.Event(),
        released=threading.Event(),
    )
    slot.client_alive.set()

    try:
        displaced = _try_claim_slot(slot)
    except HTTPException:
        # Lost the race against a still-connected sibling — return the
        # acquired worker to the pool so we don't waste its warm state.
        pool.return_alive(worker)
        raise

    if displaced is not None:
        # Previous client was gone — replace the orphan via the pool so a
        # fresh standby is on its way for the next request.
        displaced.cancelled.set()
        _release(displaced)

    worker.submit(request.question)
    asyncio.create_task(_watch_disconnect(http_request, slot))

    return StreamingResponse(
        _stream_with_slot(slot),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/bird")
async def chat_bird(request: ChatRequestWithEvidence) -> dict:
    """Non-streaming BIRD benchmark endpoint.

    Folds ``evidence`` into the question, calls the agent directly
    (no warm-pool), and returns the result dict synchronously.
    """
    from gsf.connectors import get_connectors
    from gsf.retrieval.text_to_sql.main import get_agent_response
    from gsf.retrieval.text_to_sql.state import TextToSQLPayload
    from gsf.server.chat.settings_dal import fetch_acronyms, fetch_custom_prompts
    from gsf.utils import get_data_objects_retriever, get_semantic_objects_retriever

    question = request.question
    if request.evidence:
        question = f"{question}\n\nEvidence: {request.evidence}"

    payload: TextToSQLPayload = {
        "question": question,
        "data_retriever": get_data_objects_retriever(),
        "semantic_retriever": get_semantic_objects_retriever(),
        "connectors": get_connectors(),
        "acronyms": fetch_acronyms(),
        "custom_prompts": fetch_custom_prompts(),
        "target_db": request.database,
    }
    result = get_agent_response(payload)
    return {
        "database": request.database,
        "question": request.question,
        "evidence": request.evidence,
        **result,
    }
