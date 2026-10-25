# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Answering a question from retrieved chunks.

The prompt, the model, the pacing and the retries for the grounded answer the
unstructured pipeline produces. It lives here because it is what the product
does, not what the harness measures: the evaluation repo owned all of this
until now, which meant nothing outside that repo could answer a question, and
the harness could not be replaced without taking the prompt with it.

Two entry points. :func:`answer` is the one-shot path — context in, answer out.
:func:`complete` is the same budget and retries around an arbitrary
conversation, for callers that build their own turns; they must go through it
rather than the endpoint, since two independent pacers each keep to the budget
and together double it.

Not routed through ``auto_ontology.utils.llm_invoke`` yet, deliberately: that
gateway is configured for a different model on a different endpoint, and every
number measured for this pipeline is paced against *this* budget. Moving it is
a change of behaviour and belongs in its own step, not inside a move.
"""

import os
import random
import threading
import time

#: The generator and where it is served. A reasoning model, capped — see
#: ``REASONING_EFFORT``.
API_BASE = "https://integrate.api.nvidia.com/v1"
GEN_MODEL = "nvidia/nemotron-3-super-120b-a12b"
GEN_MAX_TOKENS = 4096

#: How the endpoint is told how much to think, per mode.
REASONING_MODES = {
    "default": {},
    "off": {"chat_template_kwargs": {"thinking": False}},
    "low": {"reasoning": {"effort": "low"}},
    "medium": {"reasoning": {"effort": "medium"}},
    "high": {"reasoning": {"effort": "high"}},
}

#: How much the generator may reason before answering. Unbounded, it spends the
#: whole budget thinking on hard questions and returns nothing; capping it keeps
#: the reasoning and leaves room for the answer. Measured on a question that
#: truncates with reasoning unbounded:
#:
#:   default  unbounded           4096 tokens, 47s, empty answer
#:   low      kept but capped      502 tokens,  5s, answers
#:   off      no reasoning block    19 tokens,  1s, answers
#:
#: The endpoint accepts several other spellings of this and then ignores them,
#: so they look like they work and change nothing: a flat `reasoning_effort`, a
#: `chat_template_kwargs.thinking_budget`, a "/no_think" suffix, a "detailed
#: thinking off" system line, and asking for brief reasoning in the prompt.
#: `max_thinking_tokens` is rejected with a 400.
REASONING_EFFORT = "low"

#: Requests a minute. The hosted endpoint caps per model; this one tops out near
#: 8/min and gets *worse* when pushed harder. Paced client-side so every request
#: lands inside the budget instead of spending the budget on rejections.
GEN_RPM = 7

#: Characters of prompt per reserved slot. The endpoint budgets tokens as well
#: as requests, and a request limiter cannot see tokens: going from five chunks
#: of context to ten doubled prompts (~15k characters to ~29k, measured) and so
#: doubled tokens a minute at an unchanged 7/min, which is when the 429s
#: started. A prompt therefore reserves slots in proportion to its size, so a
#: long one costs what it actually costs and short ones still go at full speed.
#:
#: Set so a ten-chunk prompt takes one slot and only unusually long ones take
#: two: ~51k tokens/min, the rate 318-question runs completed at with ~6% of
#: them needing a mop-up re-run. Halving it to 15_000 restores the rate that
#: produced no 429s at all, at twice the wall clock.
GEN_CHARS_PER_SLOT = 30_000

MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 5

#: Failures worth another attempt. Anything else is the model's answer, however
#: unwelcome, and retrying it only spends budget.
RETRYABLE_ERRORS = (
    "RateLimitError",
    "APITimeoutError",
    "APIConnectionError",
    "InternalServerError",
    "APIStatusError",
)

SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "Answer the question using ONLY the information provided in the context below. "
    "If the context does not contain enough information to answer, say so clearly. "
    "Be concise and factual."
)
USER_TEMPLATE = "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

#: Separator between passages, so a model that quotes one cannot be read as
#: having joined two.
CHUNK_SEPARATOR = "\n\n---\n\n"


class RateLimiter:
    """Spaces requests to one model evenly across all worker threads."""

    def __init__(self, per_minute: float):
        self._interval = 60.0 / per_minute
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self, requests: int = 1) -> None:
        """Block until *requests* slots are available, then reserve them."""
        with self._lock:
            start = max(time.monotonic(), self._next)
            self._next = start + self._interval * requests
        delay = start - time.monotonic()
        if delay > 0:
            time.sleep(delay)


#: One limiter for the generator, shared by every caller in the process. Two
#: processes would each pace to the full budget and together exceed it, so
#: stages are meant to be run one at a time.
GEN_LIMITER = RateLimiter(GEN_RPM)

_client = None
_client_lock = threading.Lock()


def api_key() -> str:
    return os.environ.get("NVIDIA_API_KEY", "")


def get_client():
    """The generation client, built once per process.

    No client-side retries: they bypass ``GEN_LIMITER``, so each paced attempt
    could put four requests on the wire instead of one — quadrupling the real
    request rate exactly when the endpoint is already rejecting, which is how
    429s came in clusters. :func:`complete` retries with backoff through the
    limiter, covering the same failures while staying inside the budget.
    """
    global _client
    with _client_lock:
        if _client is None:
            from openai import OpenAI

            key = api_key()
            if not key:
                raise RuntimeError(
                    "NVIDIA_API_KEY is not set — get a free key at "
                    "https://build.nvidia.com"
                )
            _client = OpenAI(
                base_url=API_BASE, api_key=key, timeout=180.0, max_retries=0
            )
    return _client


def build_context(
    chunks: list[str],
    labels: list[str] | None = None,
    note: str | None = None,
) -> str:
    """The context block, with whatever is known about the passages.

    *labels* and *note* are kept out of *chunks* rather than spliced into them:
    a harness scoring whether the answer's content appears in the retrieved
    text would otherwise count an annotation as retrieved evidence.
    """
    if not chunks:
        return "(no context retrieved)"

    labels = labels or []
    # A label names what the passage is a reading of, so it belongs immediately
    # above that passage; after it, the figure has already been read without
    # knowing whose reading it is.
    blocks = [
        f"[{labels[i]}]\n{chunk}" if i < len(labels) and labels[i] else chunk
        for i, chunk in enumerate(chunks)
    ]
    context = CHUNK_SEPARATOR.join(blocks)
    return f"{note}\n\n{context}" if note else context


def answer(
    query: str,
    chunks: list[str],
    labels: list[str] | None = None,
    note: str | None = None,
    client=None,
) -> tuple[str, str | None]:
    """A grounded answer to *query*, as ``(answer, error)``."""
    context = build_context(chunks, labels, note)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(context=context, query=query)},
    ]
    return complete(messages, client=client)


def complete(messages: list[dict], client=None) -> tuple[str, str | None]:
    """One paced, retried completion over a conversation, as ``(text, error)``.

    Never raises: a caller answering several hundred questions must not lose
    the run to one of them, so a failure is returned as a named error for the
    caller to record and retry later.
    """
    client = client or get_client()
    body = REASONING_MODES[REASONING_EFFORT]
    extra = {"extra_body": body} if body else {}

    # Charged per attempt, not once per call: a retry costs the endpoint the
    # same tokens as the attempt it is replacing.
    slots = max(1, round(sum(len(m["content"]) for m in messages) / GEN_CHARS_PER_SLOT))

    for attempt in range(MAX_ATTEMPTS):
        GEN_LIMITER.acquire(slots)
        try:
            response = client.chat.completions.create(
                model=GEN_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=GEN_MAX_TOKENS,
                **extra,
            )
        except Exception as exc:  # noqa: BLE001 — one bad call must not end a run
            error = f"{type(exc).__name__}: {exc}"[:200]
            if (
                type(exc).__name__ not in RETRYABLE_ERRORS
                or attempt == MAX_ATTEMPTS - 1
            ):
                return "", error
            time.sleep(BACKOFF_BASE_SECONDS * 2**attempt + random.random())
            continue

        if not response.choices:
            return "", "empty_choices"
        text = (response.choices[0].message.content or "").strip()
        # Reasoning models can spend the whole budget thinking and return
        # nothing. Retryable now, because REASONING_EFFORT caps the thinking.
        return text, None if text else "thinking_truncated"

    return "", "retries_exhausted"
