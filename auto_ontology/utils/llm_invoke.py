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
from typing import Callable, Iterator, Type, TypeVar

import requests as _requests
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from auto_ontology.utils.model_config import resolve

logger = logging.getLogger(__name__)

R = TypeVar("R")


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

# Rate limits, server errors and dropped connections come in bursts: a gateway
# throttling a key answered 17 429s in ~30s during a benchmark, and a NeMo Gym
# model server passes each on as an empty-body 500. Three attempts 2s and 4s
# apart are over in ~7s, so the step failed mid-burst. These are retried on
# their own budget, backing off 2, 4, 8, 16, then 30s (~60s in all).
LLM_TRANSIENT_RETRIES = int(os.environ.get("LLM_TRANSIENT_RETRIES", "5"))
LLM_RETRY_MAX_BACKOFF_S = float(os.environ.get("LLM_RETRY_MAX_BACKOFF_S", "30"))

# A reasoning call that spends max_tokens thinking returns nothing, and sampled
# again it often does the same: one benchmark question made 18 such calls at
# ~100s each and got no SQL. The retries after one such call run with thinking
# off, as json_schema, capped low. Measured with nemotron-3.5-lightning on five
# SQL-generation prompts x2: thinking off, json_schema finished all 10 in
# 330-1,060 tokens (2.5-7s), while tool calls ran 9 of 10 to 32k writing the
# leading free-text field. "off" disables the fallback.
LLM_THINKING_FALLBACK = os.environ.get("LLM_THINKING_FALLBACK", "on").lower() != "off"
LLM_THINKING_FALLBACK_MAX_TOKENS = int(
    os.environ.get("LLM_THINKING_FALLBACK_MAX_TOKENS", "4096")
)

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


# The model answered, but nothing usable came back: ChatOpenAI raises
# LengthFinishReasonError when a json_schema answer hits max_tokens, and
# OutputParserException when tool-call arguments do not parse (a truncated call
# among them). Matched by name, like the types above, so ChatNVIDIA-only
# installs need not import openai.
_UNUSABLE_OUTPUT_TYPES = frozenset({"LengthFinishReasonError", "OutputParserException"})


def _is_unusable_output(e: Exception) -> bool:
    """Whether *e* is a completed call whose output was cut off or did not parse."""
    return type(e).__name__ in _UNUSABLE_OUTPUT_TYPES


_TIMEOUT_EXCEPTION_TYPES = frozenset({"APITimeoutError", "ReadTimeout", "Timeout"})


def _is_timeout(e: Exception) -> bool:
    """Whether *e* is the client giving up on a call that was still running."""
    return type(e).__name__ in _TIMEOUT_EXCEPTION_TYPES or "timed out" in str(e)


def _is_transient(e: Exception) -> bool:
    """A rate limit, server error or dropped connection: worth waiting out.

    Timeouts are excluded. A call cut off by ``LLM_INVOKE_TIMEOUT_S`` is usually
    a model still generating, and minutes of backoff would only repeat it; those
    keep the ``RETRY_MAX_ATTEMPTS`` budget. So are client errors other than 408
    and 429: a gateway's 401 for a bad key mentions "litellm", which
    ``_is_retryable`` accepts, and would otherwise take a minute to surface.
    """
    status = getattr(e, "status_code", None)
    if isinstance(status, int) and 400 <= status < 500 and status not in (408, 429):
        return False
    return _is_retryable(e) and not _is_timeout(e)


def _invoke_with_backoff(call: Callable[[], R], label: str) -> R:
    """Run *call* under the inflight bound, waiting out transient server errors.

    Retries up to ``LLM_TRANSIENT_RETRIES`` times, backing off exponentially to
    ``LLM_RETRY_MAX_BACKOFF_S``. The bound is released while waiting. Anything
    else, and a transient error once the budget is spent, is raised.
    """
    for retry in range(LLM_TRANSIENT_RETRIES + 1):
        try:
            with _inflight_guard():
                return call()
        except Exception as e:
            if retry == LLM_TRANSIENT_RETRIES or not _is_transient(e):
                raise
            wait = min(2 ** (retry + 1), LLM_RETRY_MAX_BACKOFF_S) + random.uniform(0, 1)
            logger.warning(
                "Rate-limited or server error for %s (retry %d/%d) — retrying in "
                "%.1fs: %s: %s",
                label,
                retry + 1,
                LLM_TRANSIENT_RETRIES,
                wait,
                type(e).__name__,
                str(e)[:120],
            )
            time.sleep(wait)
    raise AssertionError("unreachable")


def _ran_out_while_thinking(e: Exception) -> bool:
    """Whether *e* is a call cut off at max_tokens before writing any answer."""
    if type(e).__name__ != "LengthFinishReasonError":
        return False
    try:
        message = e.completion.choices[0].message  # type: ignore[attr-defined]
    except (AttributeError, IndexError, TypeError):
        return False
    return not (message.content or "").strip() and not getattr(
        message, "tool_calls", None
    )


def _without_thinking(llm: BaseChatModel) -> BaseChatModel | None:
    """A copy of *llm* with thinking off and a low cap, or None if not applicable.

    Only for nemotron on ChatOpenAI, where ``enable_thinking`` was measured to
    work; a gateway fronting another model may reject the field. The copy shares
    the underlying HTTP client, and *llm* itself is left as it was.
    """
    if not LLM_THINKING_FALLBACK:
        return None
    client = getattr(llm, "bound", llm)
    if type(client).__name__ != "ChatOpenAI" or _thinking_is_off(client):
        return None
    model = str(getattr(client, "model_name", "") or "")
    if "nemotron" not in model.lower():
        return None
    extra_body = dict(getattr(client, "extra_body", None) or {})
    extra_body["chat_template_kwargs"] = {
        **(extra_body.get("chat_template_kwargs") or {}),
        "enable_thinking": False,
    }
    return client.model_copy(
        update={
            "extra_body": extra_body,
            "max_tokens": LLM_THINKING_FALLBACK_MAX_TOKENS,
        }
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


_CLIENTS = ("openai", "nvidia")
_THINKING = {"on": True, "off": False}


def max_tokens_for(role: str, default: int) -> int:
    """The completion-token cap for a *role* call that would otherwise use *default*.

    ``<role>_MAX_TOKENS`` (or ``DEFAULT_MODELS_MAX_TOKENS``) replaces every
    per-call cap for that role. Reasoning models count their thinking against the
    cap, so caps sized for the answer alone (300 to 8192 across the call sites)
    cut them off before the answer is written, and the structured parse fails.
    """
    value = resolve(role, "MAX_TOKENS")
    if not value:
        return default
    try:
        cap = int(value)
    except ValueError:
        raise ValueError(f"{role}_MAX_TOKENS must be an integer, got {value!r}")
    if cap <= 0:
        raise ValueError(f"{role}_MAX_TOKENS must be positive, got {cap}")
    return cap


def _client_kind(role: str, model: str) -> str:
    """``openai`` or ``nvidia``: which langchain client *role* uses for *model*."""
    # OpenAI-family names always need ChatOpenAI, whatever the key, so configs that
    # predate <role>_CLIENT keep working.
    if model.startswith(("openai/", "azure/", "aws/")):
        return "openai"
    kind = resolve(role, "CLIENT").lower()
    if kind not in _CLIENTS:
        raise ValueError(f"{role}_CLIENT must be one of {_CLIENTS}, got {kind!r}")
    return kind


def _template_kwargs(role: str) -> dict:
    """``chat_template_kwargs`` for *role*, from ``<role>_THINKING``; empty if unset."""
    value = resolve(role, "THINKING").lower()
    if not value:
        return {}
    if value not in _THINKING:
        raise ValueError(
            f"{role}_THINKING must be one of {tuple(_THINKING)}, got {value!r}"
        )
    return {"chat_template_kwargs": {"enable_thinking": _THINKING[value]}}


def _build_client(
    *,
    role: str,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
) -> BaseChatModel:
    """Build a chat client for *role*'s model/endpoint.

    ``<role>_CLIENT`` picks ``ChatOpenAI`` or ``ChatNVIDIA`` (see
    :mod:`auto_ontology.utils.model_config`). ``ChatNVIDIA`` requests
    ``{base_url}/models`` before structured output, so it fails behind any
    OpenAI-compatible proxy that serves only chat completions, such as a NeMo Gym
    model server.
    """
    max_tokens = max_tokens_for(role, max_tokens)
    template_kwargs = _template_kwargs(role)

    if _client_kind(role, model) == "openai":
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
            extra_body=template_kwargs or None,
        )

    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    client = ChatNVIDIA(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        model_kwargs=template_kwargs,
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
        role="REASONING",
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
        role="NON_REASONING",
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

    Nemotron behind an OpenAI-compatible endpoint uses ``function_calling``, with
    langchain's forced tool choice, when its thinking is off -- and only then.
    Either way the failure is the model not stopping once its answer is
    complete, and which path shows it depends on thinking. Measured with
    nemotron-3.5-lightning on inference-api, replaying Auto Ontology's own
    prompts:

    - Thinking off (97 non-reasoning prompts x5): json_schema ran to max_tokens
      on 12.0% of calls, 56 of 58 emitting only whitespace (which the JSON grammar
      allows between tokens) after a nearly complete answer. Tool calls ran to
      the cap on 1.6%, none in whitespace, at half the output tokens.
    - Thinking on (5 SQL-generation prompts x3): tool calls ran to the cap on
      15 of 15, generating unseen for ~90s after a complete call, while
      json_schema did on 1 of 15.
    """
    model = str(getattr(llm, "model_name", "") or getattr(llm, "model", "") or "")
    lowered = model.lower()
    if "anthropic" in lowered or "claude" in lowered:
        return {"method": "function_calling", "tool_choice": None}
    if "nemotron" in lowered and _thinking_is_off(llm):
        return {"method": "function_calling"}
    return {}


def _thinking_is_off(llm: BaseChatModel) -> bool:
    """Whether *llm* is a ChatOpenAI client sending ``enable_thinking: false``.

    Sees through ``llm.bind(...)``. Matched by class name, so ChatNVIDIA (its own
    structured-output path, not measured) and stand-ins return False without this
    module importing langchain_openai.
    """
    client = getattr(llm, "bound", llm)
    if type(client).__name__ != "ChatOpenAI":
        return False
    template = (getattr(client, "extra_body", None) or {}).get("chat_template_kwargs")
    return (template or {}).get("enable_thinking") is False


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
            response = _invoke_with_backoff(lambda: llm.invoke(messages), "text")
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
            # Transient errors were already waited out in _invoke_with_backoff.
            if _is_timeout(e) and attempt < RETRY_MAX_ATTEMPTS - 1:
                wait = 2 ** (attempt + 1) + random.uniform(0, 1)
                logger.warning(
                    "LLM call timed out on attempt %d/%d — retrying in %.1fs: %s",
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
            result = _invoke_with_backoff(
                lambda: model_llm.invoke(current_messages), schema_name
            )
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
            if _is_unusable_output(e):
                # Sampled again, the model usually finishes: a cut-off answer is a
                # bad draw, not a bad request, so this retries at once and fresh.
                if attempt < RETRY_MAX_ATTEMPTS - 1:
                    fallback = (
                        _without_thinking(llm) if _ran_out_while_thinking(e) else None
                    )
                    if fallback is not None:
                        logger.warning(
                            "%s spent max_tokens thinking on attempt %d/%d — "
                            "retrying with thinking off",
                            schema_name,
                            attempt + 1,
                            RETRY_MAX_ATTEMPTS,
                        )
                        llm = fallback
                        # json_schema, whatever the thinking-off rule picks: see
                        # LLM_THINKING_FALLBACK.
                        structured_kwargs = {}
                    else:
                        logger.warning(
                            "Unusable output for %s on attempt %d/%d (%s) — retrying",
                            schema_name,
                            attempt + 1,
                            RETRY_MAX_ATTEMPTS,
                            type(e).__name__,
                        )
                    current_messages = _ensure_non_system_message(messages.copy())
                    continue
                logger.error(
                    "Unusable output for %s on all %d attempts (%s)",
                    schema_name,
                    RETRY_MAX_ATTEMPTS,
                    type(e).__name__,
                )
                raise
            # Transient errors were already waited out in _invoke_with_backoff.
            if _is_timeout(e) and attempt < RETRY_MAX_ATTEMPTS - 1:
                wait = 2 ** (attempt + 1) + random.uniform(0, 1)
                logger.warning(
                    "LLM call timed out on attempt %d/%d for %s — retrying in %.1fs",
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
        # No attempt count: a non-retryable error stops after the first one.
        logger.error(
            f"invoke_with_structured_output failed for {schema_name}: "
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
            "Non-reasoning LLM failed (%s: %s) — falling back to reasoning LLM",
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
            "Non-reasoning LLM failed for %s (%s: %s) — falling back to reasoning LLM",
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
