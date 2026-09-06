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
Slots are keyed by the authenticated user plus ``conversation_id``. Stateless
calls (no ``conversation_id``, see ``ChatRequest``) get a fresh, unique key
per request instead and so never contend with anything.

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

Disconnecting never cuts the answer short
------------------------------------------
A disconnected client (navigate-away, switch to another conversation, closed
tab) only clears ``client_alive`` — it does **not** cancel the run. ``_pump``
keeps draining the worker and persisting the result independently of any HTTP
connection, so the agent always finishes and the full answer always lands in
the conversation, whether or not anyone is still watching. The only things
that actually kill an in-flight worker are an explicit ``POST /chat/cancel``
(Stop button) or a genuinely new question preempting an orphaned slot above.
Reattach with ``/chat/watch`` to see a run that kept going after you left.

The chart/table bubble is guaranteed too
-----------------------------------------
Once the SQL answer ("Message 1") lands, ``_pump`` immediately generates and
persists the chart-or-table bubble ("Message 2") itself — see
``_build_charts_event``. It streams back as its own ``charts`` SSE event once
ready, so a still-connected client renders it with no extra request, and
``/chat/watch`` replays it for anyone who reattaches later. Same guarantee as
the SQL answer: it exists whether or not a browser tab is still around to ask
for it — provided the call had a ``conversation_id`` to persist into.
Stateless calls (no ``conversation_id``) don't get a chart step at all, same
as they never got the old client-driven visualize step without an
authenticated conversation to save into.

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
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from gsf.dal import datasources as datasources_dal
from gsf.dal.terms import semantic_layer_calculated
from gsf.retrieval.text_to_sql.visualization import analyze_and_visualize
from gsf.server.chat.helpers import (
    NODE_LABELS,
    ChatRequest,
    build_result_message,
)
from gsf.server.chat.conversation_dal import (
    ConversationAccessError,
    create_stateless_analytics,
    persist_analytics_result,
    persist_assistant_result,
    persist_result_message,
    prepare_conversation,
)
from gsf.server.chat.identity import resolve_internal_user
from gsf.server.chat.settings_dal import is_visualization_enabled
from gsf.server.chat.worker import PrewarmedWorker, get_pool
from gsf.utils.llm_invoke import get_llm_client, get_non_reasoning_llm_client
from gsf.server.responses import ChatCancelResponse

logger = logging.getLogger(__name__)

router = APIRouter()


class EventStreamResponse(StreamingResponse):
    """Declares the SSE media type so the spec doesn't also claim JSON."""

    media_type = "text/event-stream"


# The two streaming routes emit OpenAI-style SSE frames rather than a JSON
# body, so they document a media type instead of a response_model.
SSE_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "content": {"text/event-stream": {"schema": {"type": "string"}}},
        "description": "Server-sent events; the stream ends with `data: [DONE]`",
    }
}

# Built once at import time, same fallback order as the main agent pipeline
# (gsf.retrieval.text_to_sql.main): prefer the cheaper non-reasoning model,
# fall back to the main reasoning model. Neither client does I/O at
# construction time, so this is safe to build eagerly; a missing API key
# just leaves the client as ``None`` and ``_build_charts_event`` soft-fails
# per run.
try:
    _non_reasoning_llm = get_non_reasoning_llm_client(max_tokens=2048)
except (ValueError, EnvironmentError) as e:
    logger.warning("Chart generation: failed to init non-reasoning LLM: %s", e)
    _non_reasoning_llm = None

try:
    _reasoning_llm = get_llm_client()
except (ValueError, EnvironmentError) as e:
    logger.warning("Chart generation: failed to init reasoning LLM: %s", e)
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

# Headers every SSE response here carries. Shared so the two endpoints cannot
# drift apart — ``/chat/watch`` is a GET, so it is the one a shared cache would
# actually be willing to store.
#
# ``no-store``, not ``no-cache``: the latter permits storing the response and
# only requires revalidation before reuse, and these bodies carry the user's
# SQL and the rows it returned. ``X-Accel-Buffering`` is what stops nginx from
# buffering the stream; the cache directive has no bearing on that.
_SSE_HEADERS = {
    "Cache-Control": "no-store",
    "X-Accel-Buffering": "no",
}


def _resolve_chat_target_db(target_db: str | None) -> str | None:
    """Resolve a catalog database UUID or name to its canonical name."""
    if target_db is None or not target_db.strip():
        return None

    requested = target_db.strip()
    databases = datasources_dal.fetch_databases(zone_ids=None)
    for database in databases:
        database_id = str(database.get("id") or "").strip()
        database_name = str(database.get("name") or "").strip()
        if database_name and (
            requested == database_id or requested.casefold() == database_name.casefold()
        ):
            return database_name

    raise HTTPException(
        status_code=422,
        detail=(
            f"target_db {target_db!r} does not match a catalog database UUID or name."
        ),
    )


@dataclass
class _Slot:
    """In-flight stream descriptor stored in ``_active_slots``."""

    key: str
    worker: PrewarmedWorker
    user_id: str | None
    conversation_id: UUID | None
    analytics_id: str | None
    # The question this run answers — needed by ``_pump`` to generate the
    # chart step itself (mirrors what the client used to pass to
    # ``POST /chat/visualize``).
    question: str
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
    # Last SQL the run streamed out (from a ``sql`` event). A failure after
    # the query was validated — execution errors that exhaust the retry
    # budget, or a hard error — produces an answer with no ``sql_code``, so
    # without remembering it here the query the user just watched fail would
    # vanish from the persisted turn the moment the page reloaded. A run that
    # never got a query past validation has no ``sql`` event and so persists
    # none. See ``_pump``.
    last_sql: str | None = None


# Keyed by authenticated user + conversation_id (or a per-request unique key
# for stateless calls), so distinct users and conversations never contend.
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
    """Mark the slot orphaned the moment the ASGI client disconnects.

    Without this, the sync streaming generator only learns the client is
    gone when its next write to the socket fails — and the kernel happily
    buffers small heartbeat writes, so that signal can lag by tens of
    seconds, holding the slot and forcing follow-up requests into 409.

    Crucially, this only clears ``client_alive`` — it never cancels the
    run or releases the slot itself. A disconnected client (navigated
    away, switched to another conversation, closed the tab) must not cut
    the agent off mid-answer: ``_pump`` keeps draining the worker and
    persisting the result regardless of whether anyone is watching, so the
    run always finishes and the full answer always lands in the
    conversation. Clearing ``client_alive`` only makes the slot eligible
    for preemption — i.e. a genuinely new question posted into the same
    conversation (see ``_try_claim_slot``) is what actually cancels an
    orphaned run, not the mere act of disconnecting.
    """

    while slot.client_alive.is_set():
        try:
            disconnected = await http_request.is_disconnected()
        except Exception:  # noqa: BLE001 — defensive; never want to leak
            logger.exception("Disconnect watchdog failed")
            return
        if disconnected:
            slot.client_alive.clear()
            return
        await asyncio.sleep(_DISCONNECT_POLL_S)


def _build_charts_event(slot: _Slot, answer: dict[str, Any]) -> dict[str, Any] | None:
    """Compute and persist Message 2 (chart, or the raw result table) for ``answer``.

    Runs unconditionally from ``_pump`` — the same guarantee the SQL answer
    already gets — instead of depending on a browser tab staying around to
    ask for it. Returns the SSE event to buffer (so any live/reattached
    consumer renders it immediately), or ``None`` when there's nothing to
    show (no executed result).
    """

    # Stateless calls (no ``conversation_id``) have nowhere to persist a
    # chart bubble to, so they simply don't get one.
    if slot.conversation_id is None or slot.user_id is None:
        return None

    sql_response_from_db = answer.get("sql_response_from_db")
    if not sql_response_from_db:
        return None

    charts: list[dict[str, Any]] | None = None
    llm = _non_reasoning_llm or _reasoning_llm
    if llm is not None:
        try:
            if is_visualization_enabled():
                charts = analyze_and_visualize(
                    llm=llm,
                    question=slot.question,
                    sql=str(answer.get("sql_code") or ""),
                    sql_response_from_db=sql_response_from_db,
                )
        except Exception:  # noqa: BLE001 — charts are best-effort
            logger.exception("Chart generation failed")

    if slot.cancelled.is_set():
        return None

    built = build_result_message(sql_response_from_db, charts)
    if built is None:
        return None
    content, sql_response = built

    if slot.cancelled.is_set():
        return None

    try:
        persist_result_message(
            conversation_id=slot.conversation_id,
            user_id=slot.user_id,
            content=content,
            sql_response=sql_response,
        )
    except Exception:  # noqa: BLE001 — persistence must not break SSE
        logger.exception("Failed to persist chart/result message")

    return {"type": "charts", "content": content, "sql_response": sql_response}


def _pump(slot: _Slot) -> None:
    """Drain ``slot.worker``'s event queue into ``slot.buffer``.

    Runs in its own thread, independent of any HTTP request/response — this
    decoupling is what lets a reloaded/reopened browser tab "reconnect" via
    ``/chat/watch`` and see every step of a run from the start, not just
    whatever happens to arrive after it joins. Releases the slot back to
    the pool once the run ends, naturally or via error, regardless of
    whether anyone is currently watching.
    """

    persisted = False

    def persist_event(event: dict) -> None:
        nonlocal persisted
        if persisted or slot.user_id is None or slot.analytics_id is None:
            return

        event_type = event.get("type")
        if event_type == "result":
            answer = event.get("answer") or {}
            response = str(answer.get("response") or "")
            # An answer the agent gave up on (``unconstructable_sql_response``)
            # carries no SQL, even when a query did run earlier in the turn.
            # Fall back to the last one streamed so the failed query is still
            # in history, matching what the user watched live.
            sql_code = answer.get("sql_code") or slot.last_sql
        elif event_type == "error":
            response = str(event.get("message") or "")
            sql_code = slot.last_sql
        else:
            return
        if not response and not sql_code:
            return

        try:
            if slot.conversation_id is not None:
                persist_assistant_result(
                    conversation_id=slot.conversation_id,
                    user_id=slot.user_id,
                    analytics_id=slot.analytics_id,
                    response=response,
                    sql_code=sql_code,
                )
            else:
                persist_analytics_result(
                    analytics_id=slot.analytics_id,
                    user_id=slot.user_id,
                    response=response,
                    sql_code=sql_code,
                )
            persisted = True
        except Exception:  # noqa: BLE001 — persistence must not break SSE
            logger.exception("Failed to persist assistant conversation turn")

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
            elif event.get("type") == "sql":
                # Streamed straight through for the live view; remembered so
                # the failure paths above have something to persist.
                slot.last_sql = str(event.get("sql") or "") or None
            persist_event(event)
            with slot.buffer_lock:
                slot.buffer.append(event)
            if event.get("type") == "result" and not slot.cancelled.is_set():
                try:
                    charts_event = _build_charts_event(slot, event.get("answer") or {})
                except Exception:  # noqa: BLE001 — the chart step must not break the run
                    logger.exception("Chart step failed")
                    charts_event = None
                if charts_event is not None:
                    with slot.buffer_lock:
                        slot.buffer.append(charts_event)
    except RuntimeError as exc:
        logger.exception("Agent stream failed")
        error_event = {"type": "error", "message": f"Agent stream failed: {exc}"}
        persist_event(error_event)
        with slot.buffer_lock:
            slot.buffer.append(error_event)
    finally:
        slot.finished.set()
        _release(slot)


def _subject_token(http_request: Request) -> str | None:
    """Return the caller's SSO JWT when a connection authenticates as the user.

    The frontend forwards the browser user's SSO token (or a service caller's
    own bearer token) as ``Authorization: Bearer``. Fail closed: when any
    connection has "Authenticate as signed-in user" enabled, a request without a
    token is rejected rather than allowed to run under that connection's PAT.
    """
    from gsf.connectors.databricks_oauth import any_connection_uses_sso_federation

    if not any_connection_uses_sso_federation():
        return None

    header = http_request.headers.get("authorization") or ""
    scheme, _, token = header.partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=401,
            detail=(
                "A connection is configured to authenticate as the signed-in "
                "user, but the request carried no SSO bearer token."
            ),
        )
    return token


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


@router.post(
    "/chat/completions",
    response_class=EventStreamResponse,
    responses=SSE_RESPONSES,
)
async def chat_completions(
    request: ChatRequest, http_request: Request
) -> StreamingResponse:
    """Run the text-to-SQL agent and stream OpenAI-style SSE frames back.

    Emits a ``sql`` event once the agent has a validated query, just before it
    executes, so callers can show the SQL while it runs. Emits the final SQL
    and formatted answer as a ``result`` event, then — if that
    answer has an executed result — generates and persists the chart/table
    bubble itself and streams it back as a ``charts`` event before the stream
    closes with ``[DONE]``. Callers that only care about the text/SQL answer
    can still stop reading after ``result``; the chart step and its
    persistence happen server-side regardless of whether anyone keeps
    listening (see ``_build_charts_event``).

    Requires a compiled semantic layer; without one the request is rejected with
    409 rather than run against a bare schema. ``conversation_id`` persists the
    turn and claims that conversation's single run slot — a second concurrent
    request for the same conversation gets 409 too. Omitting ``conversation_id``
    runs the question statelessly: no history, no persisted turn, and no chart
    step (``_build_charts_event`` needs a conversation to save into).
    """
    logger.info("Chat completions request: %s", request.model_dump())

    # Block chat when the semantic layer hasn't been built — no connectors/graph
    # run happens. The chat page gates on the status API; this is the backstop.
    if not semantic_layer_calculated():
        raise HTTPException(status_code=409, detail=_SEMANTIC_MISSING_MSG)

    target_db = await asyncio.to_thread(_resolve_chat_target_db, request.target_db)

    # Conversation persistence trusts identity forwarded by the private Next.js
    # gateway; only calls that persist (i.e. carry a conversation_id) need it.
    user_id = resolve_internal_user(
        http_request, required=request.conversation_id is not None
    )
    key = (
        f"{user_id}:{request.conversation_id}"
        if request.conversation_id is not None
        else str(uuid.uuid4())
    )

    subject_token = _subject_token(http_request)

    pool = get_pool()
    worker = pool.acquire()
    slot = _Slot(
        key=key,
        worker=worker,
        user_id=user_id,
        conversation_id=request.conversation_id,
        analytics_id=None,
        question=request.question,
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

    conversation_history: list[dict[str, str | None]] = []
    if request.conversation_id is not None and user_id is not None:
        try:
            prepared = await asyncio.to_thread(
                prepare_conversation,
                conversation_id=request.conversation_id,
                user_id=user_id,
                question=request.question,
                source=http_request.headers.get("x-gsf-source") or "api",
            )
        except ConversationAccessError as exc:
            _release(slot)
            raise HTTPException(
                status_code=404, detail="Conversation not found"
            ) from exc
        except Exception as exc:
            logger.exception("Failed to prepare conversation history")
            _release(slot)
            raise HTTPException(
                status_code=503, detail="Conversation storage unavailable"
            ) from exc

        slot.analytics_id = prepared.analytics_id
        conversation_history = [
            {
                "question": turn.question,
                "response": turn.response,
                "sql_code": turn.sql_code,
            }
            for turn in prepared.history
        ]
    elif user_id is not None:
        try:
            slot.analytics_id = await asyncio.to_thread(
                create_stateless_analytics,
                user_id=user_id,
                question=request.question,
                source=http_request.headers.get("x-gsf-source") or "api",
            )
        except Exception:  # noqa: BLE001 — analytics is best-effort
            logger.exception("Failed to create stateless conversation analytics")

    worker.submit(
        request.question,
        prediction=request.prediction,
        target_db=target_db,
        subject_token=subject_token,
        conversation_history=conversation_history,
    )
    threading.Thread(target=_pump, args=(slot,), daemon=True).start()
    asyncio.create_task(_watch_disconnect(http_request, slot))

    return StreamingResponse(
        _stream_slot(slot),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.get(
    "/chat/watch",
    response_class=EventStreamResponse,
    responses=SSE_RESPONSES,
)
async def chat_watch(conversation_id: UUID, request: Request) -> StreamingResponse:
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

    user_id = resolve_internal_user(request, required=True)
    key = f"{user_id}:{conversation_id}"
    with _slot_lock:
        slot = _active_slots.get(key)

    if slot is None:
        return StreamingResponse(
            iter(["data: [DONE]\n\n"]),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    return StreamingResponse(
        _stream_slot(slot),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.post("/chat/cancel", response_model=ChatCancelResponse)
async def chat_cancel(conversation_id: UUID, request: Request) -> dict[str, bool]:
    """Abort the in-flight run for ``conversation_id``, if any.

    Backs the chat UI's Stop button. Without this, stopping only detached
    the browser while the agent kept running and holding the slot, so the
    very next question in the same conversation was rejected with 409.

    Killing the worker makes ``PrewarmedWorker.events()`` return rather
    than raise, so ``_pump`` appends no error event: the run's buffer ends
    with whatever steps completed and the stream closes with ``[DONE]``.
    Consumers therefore see a run that stopped without an answer, and the
    conversation keeps its user turn with no assistant reply.
    """

    user_id = resolve_internal_user(request, required=True)
    key = f"{user_id}:{conversation_id}"
    with _slot_lock:
        slot = _active_slots.get(key)

    if slot is None:
        return {"cancelled": False}

    slot.cancelled.set()
    _release(slot)
    return {"cancelled": True}
