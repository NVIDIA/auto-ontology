# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM client construction and structured-output invocation wrappers."""

import logging
import os
import random
import threading
import time
from typing import Type, TypeVar

import requests as _requests
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

RETRY_MAX_ATTEMPTS = 3
LLM_INVOKE_TIMEOUT_S = 50

# Bound total concurrent LLM requests across all worker threads so the pipeline's
# nested parallelism (tables × terms) doesn't saturate the hosted endpoint's
# per-worker request cap (which surfaces as HTTP 503 ResourceExhausted).
LLM_MAX_INFLIGHT = int(os.environ.get("LLM_MAX_INFLIGHT", "6"))
_INFLIGHT = threading.BoundedSemaphore(LLM_MAX_INFLIGHT)

# Substrings that indicate a transient, retryable server condition.
_RETRYABLE_TOKENS = (
    "429",
    "Too Many Requests",
    "503",
    "ResourceExhausted",
    "Service Unavailable",
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

_BASE_URL = os.environ.get("BASE_URL", "https://integrate.api.nvidia.com/v1")
_MODEL_NAME = os.environ.get("MODEL_NAME", "nvidia/nemotron-3-nano-30b-a3b")
_API_KEY = os.environ.get("NVIDIA_API_KEY", "")


def get_llm_client(
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> BaseChatModel:
    """Create an LLM client.

    Parameters
    ----------
    model : str | None
        Override the default ``MODEL_NAME`` env var for this client.
    """
    if not _API_KEY:
        raise EnvironmentError("NVIDIA_API_KEY is not set")

    resolved_model = model or _MODEL_NAME

    if resolved_model.startswith("openai/"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=resolved_model,
            api_key=_API_KEY,
            base_url=_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    client = ChatNVIDIA(
        model=resolved_model,
        api_key=_API_KEY,
        base_url=_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    client._client.get_session_fn = lambda: _TimeoutSession(LLM_INVOKE_TIMEOUT_S)
    return client


def invoke_text(llm: BaseChatModel, prompt: str) -> str:
    """Invoke the LLM with a single system-message *prompt* and return its text.

    Free-text counterpart to :func:`invoke_with_structured_output`, for callers
    that parse the raw response themselves (e.g. the text-to-PQL pipeline).
    """
    response = llm.invoke([SystemMessage(content=prompt)])
    content = getattr(response, "content", response)
    return content if isinstance(content, str) else str(content)


def safe_invoke_with_structured_output(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    schema: Type[T],
) -> T:
    """LLM structured call with retry."""
    current_messages = messages.copy()
    schema_name = getattr(schema, "__name__", str(schema))

    for attempt in range(RETRY_MAX_ATTEMPTS):
        try:
            model_llm = llm.with_structured_output(schema)
            with _INFLIGHT:
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
                continue
            else:
                logger.error(
                    f"Validation failed after {RETRY_MAX_ATTEMPTS} attempts for {schema_name}"
                )
                raise
        except Exception as e:
            is_retryable = any(tok in str(e) for tok in _RETRYABLE_TOKENS)
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
