# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Long-lived prewarmed agent subprocess + a single-slot warm pool.

The agent pipeline is sync Python that spends most of its time inside
C-level calls (psycopg, httpx, embedding clients). We can't interrupt
those from another Python thread, so cancellation has to go through the
OS via process kill — same constraint as before.

What's new vs. spawn-per-request: we keep **one** subprocess alive
between requests. It imports ``nemo_retriever`` and builds the
retriever/connector singletons once, then loops on an input queue:
``("ask", question)`` → stream agent events on the output queue → idle
again. The cold-start cost is paid at server boot and at each cancel
(when we kill+respawn), not per request.

Because the product spec is "one conversation at a time", the pool size
is hard-coded to 1: a single warm worker ready to be acquired, plus
async replenishment after acquire/cancel so the next question almost
always finds the slot prewarmed.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
from queue import Empty
from typing import Any, Generator

logger = logging.getLogger(__name__)

# Wire-format tags. ``ready`` is sent once after imports + retriever/connector
# init complete, so the pool can tell a warming worker from a fully usable
# one (the parent doesn't strictly *need* to wait — submit() will just block
# on the worker's in_queue.get() until the loop runs — but knowing readiness
# helps with logging/health checks).
_TAG_READY = "ready"
_TAG_EVENT = "event"
_TAG_DONE = "done"
_TAG_ERROR = "error"

# Inbound message tags (parent → worker).
_MSG_ASK = "ask"

# How often events() polls the outbound queue. Doubles as the heartbeat
# cadence the router uses to write SSE comments.
_QUEUE_POLL_S = 0.25

# Grace window between SIGTERM and SIGKILL.
_TERMINATE_GRACE_S = 1.0


def _worker_loop(
    in_q: "mp.Queue[tuple[str, Any]]",
    out_q: "mp.Queue[tuple[str, Any]]",
) -> None:
    """Subprocess entry: import once, then process ``ask`` messages forever."""

    # `multiprocessing` with the `spawn` start method does NOT inherit the
    # parent's logging configuration — the child runs straight into this
    # function, never executing the parent's `logging.basicConfig` call in
    # `gsf.server.__main__`. Without re-applying it here, INFO-level logs
    # from the agent ("Final answer (Xs):", per-step traces, etc.) are
    # silently dropped because Python's default root level is WARNING.
    # Subprocess stdout/stderr are inherited from the parent, so once
    # configured, log lines flow into the same terminal as everything else.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        from nemo_retriever.tabular_data.retrieval.text_to_sql.main import (
            stream_agent_response,
        )

        from gsf.connectors import get_connectors
        from gsf.server.chat.helpers import get_retriever
        from gsf.server.chat.settings_dal import (
            fetch_acronyms,
            fetch_custom_prompts,
        )

        # Eagerly build the heavyweights so they're hot before the first ask.
        retriever = get_retriever()
        connectors = get_connectors()
    except BaseException as exc:  # noqa: BLE001 — surface init failure to parent
        logger.exception("Worker init failed")
        try:
            out_q.put((_TAG_ERROR, f"Worker init failed: {exc}"))
            out_q.put((_TAG_DONE, None))
        except Exception:  # noqa: BLE001
            pass
        return

    try:
        out_q.put((_TAG_READY, None))
    except Exception:  # noqa: BLE001
        return

    while True:
        try:
            message = in_q.get()
        except (EOFError, OSError):
            return
        if not isinstance(message, tuple) or len(message) != 2:
            continue
        tag, payload = message
        if tag != _MSG_ASK:
            continue

        try:
            agent_payload = {
                "question": payload,
                "retriever": retriever,
                "connectors": connectors,
                "acronyms": fetch_acronyms(),
                "custom_prompts": fetch_custom_prompts(),
            }
            for event in stream_agent_response(agent_payload):
                out_q.put((_TAG_EVENT, event))
        except BaseException as exc:  # noqa: BLE001 — surface to parent
            logger.exception("Agent run failed")
            try:
                out_q.put((_TAG_ERROR, str(exc)))
            except Exception:  # noqa: BLE001
                pass
        finally:
            try:
                out_q.put((_TAG_DONE, None))
            except Exception:  # noqa: BLE001
                pass


class PrewarmedWorker:
    """A subprocess running ``_worker_loop`` — imports done, ready to ask.

    Lifecycle:

        worker = PrewarmedWorker()  # spawns subprocess; returns immediately
        worker.submit(question)     # enqueue question
        for event in worker.events():  # iterate until DONE / proc death
            ...
        # Optionally reuse: worker is now idle, can take another submit().
        worker.kill()               # SIGTERM → SIGKILL; idempotent
    """

    def __init__(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._in_q: "mp.Queue[tuple[str, Any]]" = self._ctx.Queue()
        self._out_q: "mp.Queue[tuple[str, Any]]" = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_worker_loop,
            args=(self._in_q, self._out_q),
            daemon=True,
        )
        self._proc.start()

    def is_alive(self) -> bool:
        return self._proc.is_alive()

    def submit(self, question: str) -> None:
        self._in_q.put((_MSG_ASK, question))

    def events(self) -> Generator[dict[str, Any] | None, None, None]:
        """Yield agent events for the current question.

        ``None`` is yielded when the queue is empty so the router can write
        an SSE heartbeat comment to keep proxies/load balancers happy.
        Returns when the worker emits ``done`` or the subprocess dies.
        Raises ``RuntimeError`` on worker-reported errors.
        """

        while True:
            try:
                tag, value = self._out_q.get(timeout=_QUEUE_POLL_S)
            except Empty:
                if not self._proc.is_alive():
                    return
                yield None
                continue
            if tag == _TAG_DONE:
                return
            if tag == _TAG_ERROR:
                raise RuntimeError(value)
            if tag == _TAG_READY:
                # Drained from a previous-readiness or warm-replay; ignore.
                continue
            yield value  # _TAG_EVENT

    def drain_until_ready(self, timeout: float | None = None) -> bool:
        """Block until the subprocess has finished its imports.

        Returns False if the worker errored or didn't become ready within
        the timeout. The pool uses this on warm spawns to log health; the
        request path doesn't need to wait — submit/events handle a still-
        warming worker transparently (they just block longer).
        """

        try:
            tag, _ = self._out_q.get(timeout=timeout)
        except Empty:
            return False
        return tag == _TAG_READY

    def kill(self) -> None:
        proc = self._proc
        if not proc.is_alive():
            return
        proc.terminate()
        proc.join(timeout=_TERMINATE_GRACE_S)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=_TERMINATE_GRACE_S)


class WarmPool:
    """Maintains exactly one prewarmed standby worker.

    * ``acquire()`` — take the standby (or cold-start synchronously if it
      isn't there yet); does NOT immediately spawn a replacement, since the
      caller decides whether the acquired worker will be returned (natural
      end) or killed (cancel).
    * ``return_alive(w)`` — request finished naturally; if ``w`` is still
      alive, put it back as the standby (no fresh spawn needed). Otherwise
      spawn a new standby.
    * ``replace(w)`` — request was cancelled; kill ``w`` and spawn a new
      standby in the background.

    All spawn work happens off the request thread so we never block a POST
    on subprocess startup unless the standby genuinely isn't there yet.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._standby: PrewarmedWorker | None = None
        self._spawn_async()

    def _spawn_async(self) -> None:
        threading.Thread(target=self._spawn_sync, daemon=True).start()

    def _spawn_sync(self) -> None:
        worker = PrewarmedWorker()
        # Block here (off the request path) until imports complete, so
        # ``acquire()`` is more likely to hand back a fully warm worker.
        # We don't enforce a timeout — initial imports can be slow on
        # cold caches and we'd rather wait than serve a half-warm one.
        worker.drain_until_ready()
        with self._lock:
            if self._standby is None:
                self._standby = worker
                return
        # A concurrent return_alive filled the slot first; drop the spare.
        worker.kill()

    def acquire(self) -> PrewarmedWorker:
        with self._lock:
            w = self._standby
            self._standby = None
        if w is not None and w.is_alive():
            return w
        # Standby missing/dead — pay cold start synchronously this once.
        w = PrewarmedWorker()
        w.drain_until_ready()
        return w

    def return_alive(self, worker: PrewarmedWorker) -> None:
        if not worker.is_alive():
            self._spawn_async()
            return
        with self._lock:
            if self._standby is None:
                self._standby = worker
                return
        worker.kill()

    def replace(self, worker: PrewarmedWorker) -> None:
        worker.kill()
        with self._lock:
            if self._standby is not None and self._standby.is_alive():
                return
        self._spawn_async()

    def shutdown(self) -> None:
        """Tear down on server shutdown so we don't leak orphaned subprocs."""

        with self._lock:
            standby = self._standby
            self._standby = None
        if standby is not None:
            standby.kill()


_pool: WarmPool | None = None
_pool_lock = threading.Lock()


def get_pool() -> WarmPool:
    """Lazily-initialised module-level pool. Safe under thread races."""

    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = WarmPool()
        return _pool


def shutdown_pool() -> None:
    global _pool
    with _pool_lock:
        pool = _pool
        _pool = None
    if pool is not None:
        pool.shutdown()
