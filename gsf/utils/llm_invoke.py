# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM client construction and structured-output invocation wrappers."""

import logging
import os
import time
from typing import Type, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

RETRY_MAX_ATTEMPTS = 3
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

    return ChatNVIDIA(
        model=resolved_model,
        api_key=_API_KEY,
        base_url=_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


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
            result = model_llm.invoke(current_messages)
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
            is_rate_limit = "429" in str(e) or "Too Many Requests" in str(e)
            if is_rate_limit and attempt < RETRY_MAX_ATTEMPTS - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(
                    "Rate-limited on attempt %d/%d for %s — retrying in %ds",
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
