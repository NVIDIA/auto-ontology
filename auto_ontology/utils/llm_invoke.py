# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM client construction and structured-output invocation wrappers."""

import contextlib
import logging
import os
import random
import threading
import time
from typing import Iterator, Type, TypeVar

import requests as _requests
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from auto_ontology.utils.model_config import resolve

logger = logging.getLogger(__name__)


class StrictLLMOutputModel(BaseModel):
    """Base for LLM structured-output schemas.

    Forbids genuine hallucinated fields (``extra="forbid"``) but first strips any
    key starting with ``$`` — some models leak tool-calling metadata (e.g. a stray
    ``$FUNCTION_NAME`` field echoing the tool name) into the arguments payload,
    which would otherwise fail validation and burn all retry attempts even though
    the real data is fine. Opt a schema into this by inheriting from it instead of
    ``BaseModel``.
    """

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _strip_provider_artifact_keys(cls, data):
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if not k.startswith("$")}
        return data


RETRY_MAX_ATTEMPTS = 3
LLM_INVOKE_TIMEOUT_S = int(os.environ.get("LLM_INVOKE_TIMEOUT_S", "120"))

# Concurrent LLM requests are unbounded here by default. A process-wide cap used
# to live at this call site, defaulting to 6, to stop semantic compilation's
# nested fan-out (tables x terms) from saturating the endpoint's per-worker
# request cap and drawing HTTP 503s.
#
# The cost was paid by every caller. Callers that already bound their own
# concurrency -- a server handling N requests, an eval running N questions --
# gained nothing and hit a second, lower ceiling they could not see: measured on
# a 60-question benchmark at 12 workers, lifting this cap cut wall clock 25%
# with no change in error rate, because threads sat ~3s per node waiting on the
# semaphore before reaching the network.
#
# Fan-out that needs a bound now asks for one with `limit_inflight`, which
# applies it for the duration of a block. See `run_semantic_compilation`.
LLM_MAX_INFLIGHT = int(os.environ.get("LLM_MAX_INFLIGHT", "0"))

# The process-wide default, from the environment. None means unbounded. This is
# what the guard falls back to whenever no `limit_inflight` block is active.
_ENV_INFLIGHT: threading.BoundedSemaphore | None = (
    threading.BoundedSemaphore(LLM_MAX_INFLIGHT) if LLM_MAX_INFLIGHT > 0 else None
)

# Every `limit_inflight` block currently open, as (max_inflight, semaphore).
# Tracking the whole set rather than a save/restore of one global is what makes
# overlapping blocks safe: two blocks that overlap without nesting exit in the
# opposite order to a stack, and a save/restore would then have each one put
# back the value *it* captured -- dropping the bound of a block still running,
# and leaving a stale bound installed forever once both had exited.
_ACTIVE: list[tuple[int, threading.BoundedSemaphore]] = []
_ACTIVE_LOCK = threading.Lock()

# The bound in force. Rebound only under _ACTIVE_LOCK by `_refresh_inflight`,
# but read on the hot path without it: the read is a single attribute lookup,
# and a caller that reads a semaphore just as its block ends simply waits on a
# still-valid bound for that one invoke. That is a bounded staleness, not a
# correctness problem, and it keeps the lock off every LLM call.
_INFLIGHT: threading.BoundedSemaphore | None = _ENV_INFLIGHT


def _refresh_inflight() -> None:
    """Recompute the bound in force. Caller must hold ``_ACTIVE_LOCK``."""
    global _INFLIGHT
    # The most restrictive active block wins. Only one semaphore is ever
    # acquired, so nested blocks cannot deadlock against each other.
    _INFLIGHT = (
        min(_ACTIVE, key=lambda entry: entry[0])[1] if _ACTIVE else _ENV_INFLIGHT
    )


@contextlib.contextmanager
def limit_inflight(max_inflight: int) -> Iterator[None]:
    """Bound concurrent LLM invokes to *max_inflight* for the duration of the block.

    For call paths whose own parallelism multiplies -- semantic compilation runs
    tables in parallel and fans out again within each -- where the resulting
    request rate can exceed what the endpoint will serve and come back as 503.

    The bound is process-wide, because the fan-out it exists to limit spans
    worker threads the caller does not own. While several blocks overlap the
    most restrictive one applies, in any nesting or interleaving; once the last
    one exits the process returns to ``LLM_MAX_INFLIGHT``.

    A value of 0 or less registers nothing and leaves any other block's bound
    alone -- "I need no bound" must not mean "remove someone else's".
    """
    if max_inflight <= 0:
        yield
        return
    entry = (max_inflight, threading.BoundedSemaphore(max_inflight))
    with _ACTIVE_LOCK:
        _ACTIVE.append(entry)
        _refresh_inflight()
    try:
        yield
    finally:
        with _ACTIVE_LOCK:
            # Removal is by identity of the semaphore, so concurrent blocks
            # asking for the same size still remove their own entry.
            for index, candidate in enumerate(_ACTIVE):
                if candidate[1] is entry[1]:
                    del _ACTIVE[index]
                    break
            _refresh_inflight()


def _inflight_guard() -> "contextlib.AbstractContextManager[object]":
    """The active concurrency bound, or a no-op when unbounded."""
    semaphore = _INFLIGHT
    return semaphore if semaphore is not None else contextlib.nullcontext()


# Substrings that indicate a transient, retryable server condition.
_RETRYABLE_TOKENS = (
    "429",
    "Too Many Requests",
    "503",
    "ResourceExhausted",
    "Service Unavailable",
    "timed out",
    "APITimeoutError",
    "ReadTimeout",
    # Azure / LiteLLM connection failures
    "Connection error",
    "APIConnectionError",
    "AzureException",
    "InternalServerError",
    "litellm",
)

# Exception *type* names that are always transient/retryable regardless of
# message text. Needed because str(exception) for openai's typed errors
# (e.g. openai.InternalServerError) returns only the message body — never
# the class name — so a message like "upstream connect error or
# disconnect/reset before headers... Connection refused" silently fails
# every _RETRYABLE_TOKENS substring check above even though
# "InternalServerError" is right there in the token list. Matching on
# type(e).__name__ catches these regardless of how the provider phrases
# the underlying message.
_RETRYABLE_EXCEPTION_TYPES = frozenset(
    {
        "InternalServerError",
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "ServiceUnavailableError",
    }
)


def _is_retryable(e: Exception) -> bool:
    """Whether *e* signals a transient, retryable server condition."""
    return any(tok in str(e) for tok in _RETRYABLE_TOKENS) or (
        type(e).__name__ in _RETRYABLE_EXCEPTION_TYPES
    )


class _TimeoutSession(_requests.Session):
    """requests.Session that enforces a default timeout on every request."""

    def __init__(self, timeout: float = LLM_INVOKE_TIMEOUT_S, **kwargs):
        super().__init__(**kwargs)
        self._default_timeout = timeout

    def request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", self._default_timeout)
        return super().request(method, url, **kwargs)


T = TypeVar("T", bound=BaseModel)

# Main (reasoning) model triplet. Each field falls back to DEFAULT_MODELS_<field>
# (and the API key additionally to the legacy NVIDIA_API_KEY) when unset.
_BASE_URL = resolve("REASONING", "ENDPOINT")
_MODEL_NAME = resolve("REASONING", "MODEL")
_API_KEY = resolve("REASONING", "API_KEY")

# Non-reasoning model. Kept fully separate (key/endpoint/model) so it can point at
# a different endpoint than the main model (e.g. inference vs integrate API). Each
# field falls back to DEFAULT_MODELS_<field> when unset.
_NON_REASONING_BASE_URL = resolve("NON_REASONING", "ENDPOINT")
_NON_REASONING_MODEL_NAME = resolve("NON_REASONING", "MODEL")
_NON_REASONING_API_KEY = resolve("NON_REASONING", "API_KEY")


def _build_client(
    *,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
) -> BaseChatModel:
    """Build a chat client for the given model/endpoint.

    OpenAI-family models (``openai/``, ``azure/``, ``aws/`` prefixes) go through
    ``ChatOpenAI``, which supports structured output. Everything else uses
    ``ChatNVIDIA``.
    """
    if model.startswith(("openai/", "azure/", "aws/")):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            # temperature omitted: gpt-5.x/o-series reject any explicit value and
            # only allow the server default (1). Unset => langchain sends no
            # temperature field, so the provider default applies.
            # temperature=temperature,
            max_tokens=max_tokens,
            timeout=LLM_INVOKE_TIMEOUT_S,
            max_retries=0,
        )

    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    client = ChatNVIDIA(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    client._client.get_session_fn = lambda: _TimeoutSession(LLM_INVOKE_TIMEOUT_S)
    return client


def get_llm_client(
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> BaseChatModel:
    """Create an LLM client for the main reasoning model.

    Parameters
    ----------
    model : str | None
        Override the default ``REASONING_MODEL`` env var for this client.
    """
    if not _API_KEY:
        raise EnvironmentError("REASONING_API_KEY is not set")

    return _build_client(
        model=model or _MODEL_NAME,
        api_key=_API_KEY,
        base_url=_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_non_reasoning_llm_client(
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> BaseChatModel:
    """Create an LLM client for the non-reasoning model.

    Uses ``NON_REASONING_API_KEY`` / ``NON_REASONING_ENDPOINT`` /
    ``NON_REASONING_MODEL`` so it can target a different endpoint than the
    main agent model.

    Parameters
    ----------
    model : str | None
        Override the default ``NON_REASONING_MODEL`` env var for this client.
    """
    if not _NON_REASONING_API_KEY:
        raise EnvironmentError("NON_REASONING_API_KEY is not set")

    return _build_client(
        model=model or _NON_REASONING_MODEL_NAME,
        api_key=_NON_REASONING_API_KEY,
        base_url=_NON_REASONING_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _ensure_non_system_message(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Guarantee at least one non-system message.

    Anthropic/Bedrock (via the OpenAI-compatible gateway) reject requests that
    contain only system messages with "bedrock requires at least one non-system
    message". Many agents build a single ``SystemMessage`` prompt, so when no
    user/assistant message is present we promote the last system message to a
    ``HumanMessage`` (its content is the actual instruction). No-op when a
    non-system message already exists.
    """
    if not messages or any(not isinstance(m, SystemMessage) for m in messages):
        return messages
    converted = list(messages)
    converted[-1] = HumanMessage(content=converted[-1].content)
    return converted


def _structured_output_kwargs(llm: BaseChatModel) -> dict:
    """Pick the ``with_structured_output`` method for *llm*.

    Anthropic/Claude models served through the OpenAI-compatible gateway (e.g.
    ``aws/anthropic/bedrock-claude-opus-4-8``) reject the default json_schema /
    ``response_format`` path ("output_config.format: Extra inputs are not
    permitted"), but they support tool calling — so force ``function_calling``
    for them. Everything else keeps langchain's default (json_schema for
    OpenAI), which is preferred where supported.

    ``tool_choice=None`` suppresses the tool-choice langchain would otherwise set.
    Bedrock behind LiteLLM rejects the request when one is present, because the gateway
    both maps it into ``toolConfig.toolChoice`` and forwards the original
    ``tool_choice.type``::

        The additional field tool_choice/type conflicts with the existing field
        toolConfig.toolChoice.tool. Remove tool_choice/type and try again.

    Measured against the live endpoint: every explicit value ("auto", "any",
    "required", and langchain's default of naming the tool) hits that conflict, and only
    omitting it succeeds. The model still calls the tool without being forced to, and
    the caller retries if it ever answers without one.
    """
    model = str(getattr(llm, "model_name", "") or getattr(llm, "model", "") or "")
    lowered = model.lower()
    if "anthropic" in lowered or "claude" in lowered:
        return {"method": "function_calling", "tool_choice": None}
    return {}


def invoke_text(llm: BaseChatModel, prompt: str) -> str:
    """Invoke the LLM with a single system-message *prompt* and return its text.

    Free-text counterpart to :func:`invoke_with_structured_output`, for callers
    that parse the raw response themselves (e.g. the text-to-PQL pipeline).
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    content = getattr(response, "content", response)
    return content if isinstance(content, str) else str(content)


def safe_invoke_text(llm: BaseChatModel, prompt: str) -> str:
    """invoke_text with retry/backoff for 503/429 and the inflight semaphore."""
    messages = [HumanMessage(content=prompt)]
    for attempt in range(RETRY_MAX_ATTEMPTS):
        try:
            with _inflight_guard():
                response = llm.invoke(messages)
            content = getattr(response, "content", response)
            return content if isinstance(content, str) else str(content)
        except _requests.exceptions.ReadTimeout:
            logger.error(
                "LLM invoke timed out after %ds on attempt %d/%d",
                LLM_INVOKE_TIMEOUT_S,
                attempt + 1,
                RETRY_MAX_ATTEMPTS,
            )
            if attempt < RETRY_MAX_ATTEMPTS - 1:
                time.sleep(2 ** (attempt + 1) + random.uniform(0, 1))
                continue
            raise
        except Exception as e:
            is_retryable = _is_retryable(e)
            if is_retryable and attempt < RETRY_MAX_ATTEMPTS - 1:
                wait = 2 ** (attempt + 1) + random.uniform(0, 1)
                logger.warning(
                    "Retryable LLM error on attempt %d/%d — retrying in %.1fs: %s",
                    attempt + 1,
                    RETRY_MAX_ATTEMPTS,
                    wait,
                    str(e)[:120],
                )
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("safe_invoke_text exhausted retries")


def safe_invoke_with_structured_output(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    schema: Type[T],
) -> T:
    """LLM structured call with retry."""
    current_messages = _ensure_non_system_message(messages.copy())
    schema_name = getattr(schema, "__name__", str(schema))
    structured_kwargs = _structured_output_kwargs(llm)

    for attempt in range(RETRY_MAX_ATTEMPTS):
        try:
            model_llm = llm.with_structured_output(schema, **structured_kwargs)
            with _inflight_guard():
                result = model_llm.invoke(current_messages)
        except _requests.exceptions.ReadTimeout:
            logger.error(
                "LLM invoke timed out after %ds on attempt %d/%d for %s",
                LLM_INVOKE_TIMEOUT_S,
                attempt + 1,
                RETRY_MAX_ATTEMPTS,
                schema_name,
            )
            if attempt < RETRY_MAX_ATTEMPTS - 1:
                wait = 2 ** (attempt + 1) + random.uniform(0, 1)
                time.sleep(wait)
                continue
            raise
        except ValidationError as e:
            if attempt < RETRY_MAX_ATTEMPTS - 1:
                if attempt == 0:
                    # First retry: append the error so the model can self-correct.
                    current_messages.append(
                        SystemMessage(
                            content=(
                                "Your previous output did not validate. "
                                f"Validation errors:\n{str(e)}\n"
                                "Please return a **fully valid** object that satisfies the schema. "
                                "Do not omit required fields. Do not include extra keys."
                            )
                        )
                    )
                else:
                    # Subsequent retries: the accumulated error context didn't help —
                    # reset to fresh messages to avoid compounding the confusion.
                    logger.warning(
                        "Structured output still failing after error-context retry "
                        "(attempt %d/%d) for %s — resetting to fresh messages",
                        attempt + 1,
                        RETRY_MAX_ATTEMPTS,
                        schema_name,
                    )
                    current_messages = _ensure_non_system_message(messages.copy())
                continue
            else:
                logger.error(
                    f"Validation failed after {RETRY_MAX_ATTEMPTS} attempts for {schema_name}"
                )
                raise
        except Exception as e:
            is_retryable = _is_retryable(e)
            if is_retryable and attempt < RETRY_MAX_ATTEMPTS - 1:
                wait = 2 ** (attempt + 1) + random.uniform(0, 1)
                logger.warning(
                    "Retryable LLM error (endpoint saturated/rate-limited) on attempt "
                    "%d/%d for %s — retrying in %.1fs",
                    attempt + 1,
                    RETRY_MAX_ATTEMPTS,
                    schema_name,
                    wait,
                )
                time.sleep(wait)
                continue
            logger.error(
                f"Unexpected error on attempt {attempt + 1}/{RETRY_MAX_ATTEMPTS} for {schema_name}: "
                f"{type(e).__name__}: {e}",
                exc_info=True,
            )
            raise

        if result is None:
            logger.warning(
                "LLM returned None for %s on attempt %d/%d — retrying.",
                schema_name,
                attempt + 1,
                RETRY_MAX_ATTEMPTS,
            )
            if attempt < RETRY_MAX_ATTEMPTS - 1:
                wait = 2 ** (attempt + 1) + random.uniform(0, 1)
                time.sleep(wait)
            continue
        if isinstance(result, schema):
            return result
        return schema.model_validate(result)


def invoke_with_structured_output(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    schema: Type[T],
) -> T | None:
    """Safe wrapper that returns None on failure."""
    try:
        schema_name = getattr(schema, "__name__", str(schema))
        return safe_invoke_with_structured_output(llm, messages, schema)
    except Exception as e:
        logger.error(
            f"invoke_with_structured_output failed for {schema_name} after {RETRY_MAX_ATTEMPTS} attempts: "
            f"{type(e).__name__}: {e}",
            exc_info=True,
        )
        return None


# ── Non-reasoning → reasoning fallback helpers ───────────────────────────────
# These try the non-reasoning (fast/cheap) LLM first. If it exhausts all
# retries (e.g. persistent Azure connection errors), they transparently fall
# back to the reasoning LLM for that single call rather than crashing.


def safe_invoke_text_nr(prompt: str) -> str:
    """invoke_text using non-reasoning LLM; falls back to reasoning on persistent failure."""
    try:
        llm = get_non_reasoning_llm_client()
        return safe_invoke_text(llm, prompt)
    except Exception as e:
        logger.warning(
            "Non-reasoning LLM exhausted %d retries (%s: %s) — falling back to reasoning LLM",
            RETRY_MAX_ATTEMPTS,
            type(e).__name__,
            str(e)[:120],
        )
        return safe_invoke_text(get_llm_client(), prompt)


def safe_invoke_structured_nr(
    messages: list[BaseMessage],
    schema: Type[T],
) -> T | None:
    """invoke_with_structured_output using non-reasoning LLM; falls back to reasoning on persistent failure."""
    try:
        llm = get_non_reasoning_llm_client()
        return safe_invoke_with_structured_output(llm, messages, schema)
    except Exception as e:
        schema_name = getattr(schema, "__name__", str(schema))
        logger.warning(
            "Non-reasoning LLM exhausted %d retries for %s (%s: %s) — falling back to reasoning LLM",
            RETRY_MAX_ATTEMPTS,
            schema_name,
            type(e).__name__,
            str(e)[:120],
        )
        try:
            return safe_invoke_with_structured_output(
                get_llm_client(), messages, schema
            )
        except Exception as e2:
            logger.error(
                "Reasoning LLM also failed for %s: %s: %s",
                schema_name,
                type(e2).__name__,
                str(e2)[:120],
            )
            return None
