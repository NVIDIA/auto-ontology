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
Each conversation gets its own warm agent subprocess and may only have one
in-flight stream at a time; different conversations run fully independently
(each `WarmPool.acquire()` cold-starts a fresh subprocess if no standby is
free, so there's no cap on how many conversations can run concurrently).
Slots are keyed by ``request.conversation_id`` — callers that omit it (e.g.
direct API/NAT plugin usage) get a fresh key per request and never collide.

* If the conversation's slot is empty → request runs.
* If the slot is held by a stream whose **client is still connected**
  (typical case: a second browser window/tab posting concurrently into the
  same conversation) → the new request is rejected with HTTP 409.
* If the slot is held by an **orphaned** stream (its ASGI client disconnected
  and ``client_alive`` was cleared) → the new request preempts: the orphan
  worker is killed-and-replaced via the pool, and the new request takes the
  standby.

  **Web-app caveat:** the Next.js ``/api/chat/completions`` proxy tees the
  upstream SSE stream and keeps the FastAPI connection alive until the run
  finishes (so ``after()`` can persist the assistant turn). Browser
  navigate-away therefore does **not** free the slot for web-app requests —
  a second submit into the same conversation still gets 409 until the run
  completes. Use ``/chat/watch`` to reattach instead.

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
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Generator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from gsf.dal.terms import semantic_layer_calculated
from gsf.retrieval.text_to_sql.visualization import analyze_and_visualize
from gsf.server.chat.helpers import NODE_LABELS, ChatRequest, VisualizeRequest
from gsf.server.chat.worker import PrewarmedWorker, get_pool
from gsf.utils.llm_invoke import get_llm_client, get_non_reasoning_llm_client

logger = logging.getLogger(__name__)

router = APIRouter()

# Built once at import time, same fallback order as the main agent pipeline
# (gsf.retrieval.text_to_sql.main): prefer the cheaper non-reasoning model,
# fall back to the main reasoning model. Neither client does I/O at
# construction time, so this is safe to build eagerly; a missing API key
# just leaves the client as ``None`` and the endpoint soft-fails per request.
try:
    _non_reasoning_llm = get_non_reasoning_llm_client(max_tokens=2048)
except (ValueError, EnvironmentError) as e:
    logger.warning("Chat visualize: failed to init non-reasoning LLM: %s", e)
    _non_reasoning_llm = None

try:
    _reasoning_llm = get_llm_client()
except (ValueError, EnvironmentError) as e:
    logger.warning("Chat visualize: failed to init reasoning LLM: %s", e)
    _reasoning_llm = None

# Returned as the 409 detail when a chat is attempted before the semantic
# layer has been built.
_SEMANTIC_MISSING_MSG = (
    "The semantic layer hasn't been created yet, so I can't answer questions. "
    "Enable semantic compilation to build it."
)

# How often the watchdog polls the ASGI disconnect signal. Small enough that
# a navigate-away → come-back-and-ask flow always finds the slot free.
_DISCONNECT_POLL_S = 0.1

# How often `_stream_slot` re-checks `slot.buffer` for new events when it
# has caught up to the live tail. Also the SSE heartbeat cadence.
_BUFFER_POLL_S = 0.25


@dataclass
class _Slot:
    """In-flight stream descriptor stored in ``_active_slots``."""

    key: str
    worker: PrewarmedWorker
    client_alive: threading.Event
    # Set if the watchdog detected a client disconnect. Tells the stream's
    # finally whether to ``replace`` the worker (cancel) or ``return_alive``
    # it to the warm pool (natural completion).
    cancelled: threading.Event
    # Latches the moment any path (watchdog or pump) has called back into
    # the pool, so the second path doesn't double-release.
    released: threading.Event
    # Append-only log of every SSE event this run has produced so far,
    # filled by ``_pump`` independently of any HTTP connection. Any number
    # of ``_stream_slot`` consumers (the original submitter, plus any
    # ``/chat/watch`` reconnects) replay it from index 0 and then keep
    # tailing it live — this is what lets a reloaded browser tab resume
    # watching a run that started before the reload.
    buffer: list[dict] = field(default_factory=list)
    buffer_lock: threading.Lock = field(default_factory=threading.Lock)
    # Set once the worker's event generator ends (naturally or via error),
    # i.e. once ``buffer`` is complete and no more events will ever be
    # appended.
    finished: threading.Event = field(default_factory=threading.Event)


# Keyed by conversation_id (or a generated per-request key when absent) so
# distinct conversations never contend for the same slot.
_active_slots: dict[str, _Slot] = {}
_slot_lock = threading.Lock()


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _try_claim_slot(new_slot: _Slot) -> _Slot | None:
    """Install ``new_slot`` under its key if the previous holder is gone.

    Returns the displaced slot (caller must replace its worker via the pool)
    on success. Raises HTTPException(409) if the previous slot for this key
    still has a live client.
    """

    with _slot_lock:
        prev = _active_slots.get(new_slot.key)
        if prev is not None and prev.client_alive.is_set():
            raise HTTPException(
                status_code=409,
                detail="Conversation in progress",
            )
        _active_slots[new_slot.key] = new_slot
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

    with _slot_lock:
        if _active_slots.get(slot.key) is slot:
            del _active_slots[slot.key]

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


def _pump(slot: _Slot) -> None:
    """Drain ``slot.worker``'s event queue into ``slot.buffer``.

    Runs in its own thread, independent of any HTTP request/response — this
    decoupling is what lets a reloaded/reopened browser tab "reconnect" via
    ``/chat/watch`` and see every step of a run from the start, not just
    whatever happens to arrive after it joins. Releases the slot back to
    the pool once the run ends, naturally or via error, regardless of
    whether anyone is currently watching.
    """

    try:
        for item in slot.worker.events():
            if item is None:
                # Just a queue-poll heartbeat from the worker side; the
                # heartbeat clients actually see is generated per-consumer
                # in `_stream_slot` instead, so there's nothing to buffer.
                continue
            event = item
            if event.get("type") == "step":
                node_name = event.get("node", "")
                event = {**event, "label": NODE_LABELS.get(node_name, node_name)}
            with slot.buffer_lock:
                slot.buffer.append(event)
    except RuntimeError as exc:
        logger.exception("Agent stream failed")
        with slot.buffer_lock:
            slot.buffer.append(
                {"type": "error", "message": f"Agent stream failed: {exc}"}
            )
    finally:
        slot.finished.set()
        _release(slot)


def _stream_slot(slot: _Slot) -> Generator[str, None, None]:
    """Replay ``slot.buffer`` from the start, then tail it live.

    Shared by the submitting request and any later ``/chat/watch``
    reconnects — every consumer just reads the same growing buffer, so a
    client that reconnects mid-run sees every step from the beginning
    before catching up to the live tail.
    """

    index = 0
    while True:
        with slot.buffer_lock:
            new_events = slot.buffer[index:]
            index = len(slot.buffer)
        for event in new_events:
            yield _sse(event)
        if new_events:
            continue
        if slot.finished.is_set():
            break
        # SSE comment — keeps the connection alive against any intermediary
        # (proxy/load balancer) that drops idle streams. Clients ignore
        # non-"data:" lines.
        yield ": ping\n\n"
        time.sleep(_BUFFER_POLL_S)

    yield "data: [DONE]\n\n"


@router.post("/chat/completions")
async def chat_completions(
    request: ChatRequest, http_request: Request
) -> StreamingResponse:
    # Block chat when the semantic layer hasn't been built — no connectors/graph
    # run happens. The chat page gates on the status API; this is the backstop.
    if not semantic_layer_calculated():
        raise HTTPException(status_code=409, detail=_SEMANTIC_MISSING_MSG)

    # Callers without a conversation_id (direct API/NAT plugin usage) get a
    # fresh key per request so they never contend with each other or with
    # conversations that do send one.
    key = request.conversation_id or str(uuid.uuid4())

    pool = get_pool()
    worker = pool.acquire()
    slot = _Slot(
        key=key,
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

    worker.submit(request.question, prediction=request.prediction)
    threading.Thread(target=_pump, args=(slot,), daemon=True).start()
    asyncio.create_task(_watch_disconnect(http_request, slot))

    return StreamingResponse(
        _stream_slot(slot),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/chat/watch")
async def chat_watch(conversation_id: str) -> StreamingResponse:
    """Reattach to an in-flight run for ``conversation_id``, if any.

    Purely a passive observer: it never submits a question, never touches
    ``client_alive``/``cancelled`` (so it can't trigger or block the 409
    "Conversation in progress" logic in `_try_claim_slot`), and has no
    effect on the run if it disconnects. This is what lets a reloaded or
    reopened browser tab recover the live progress UI (and, once it lands,
    the final answer) for a run it didn't itself start — replaying every
    buffered step from the beginning, then tailing new ones as they arrive.

    If nothing is running for this conversation, the stream ends
    immediately with `[DONE]` and no events, which the frontend treats
    identically to "nothing to resume" — no separate status endpoint
    needed.
    """

    with _slot_lock:
        slot = _active_slots.get(conversation_id)

    if slot is None:
        return StreamingResponse(
            iter(["data: [DONE]\n\n"]),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return StreamingResponse(
        _stream_slot(slot),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/cancel")
async def chat_cancel(conversation_id: str) -> dict[str, bool]:
    """Abort the in-flight run for ``conversation_id``, if any.

    Backs the chat UI's Stop button. Without this, stopping only detached
    the browser while the agent kept running and holding the slot, so the
    very next question in the same conversation was rejected with 409.

    Killing the worker makes ``PrewarmedWorker.events()`` return rather
    than raise, so ``_pump`` appends no error event: the run's buffer ends
    with whatever steps completed and the stream closes with ``[DONE]``.
    Consumers therefore see a run that stopped without an answer, and the
    completions proxy persists no assistant turn for it.
    """

    with _slot_lock:
        slot = _active_slots.get(conversation_id)

    if slot is None:
        return {"cancelled": False}

    slot.cancelled.set()
    _release(slot)
    return {"cancelled": True}


@router.post("/chat/visualize")
async def chat_visualize(request: VisualizeRequest) -> dict[str, Any]:
    """Second step: recommend a chart for an already-executed SQL result.

    Runs outside the ``WarmPool`` subprocess — unlike the main agent
    pipeline, this is one or two short LLM calls with no DB/retriever
    dependency, so it doesn't need process-level cancellation isolation.
    The blocking LLM calls are offloaded to a thread so they don't block
    the event loop.
    """
    llm = _non_reasoning_llm or _reasoning_llm
    if llm is None:
        logger.warning("No LLM available for chat visualize — skipping")
        return {"charts": None}

    specs = await asyncio.to_thread(
        analyze_and_visualize,
        llm=llm,
        question=request.question,
        sql=request.sql,
        sql_response_from_db=request.result,
    )
    return {"charts": specs}
