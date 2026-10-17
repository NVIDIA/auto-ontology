# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for LLM invocation helpers."""

import contextlib
import threading
import time
from typing import Any

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from auto_ontology.utils import llm_invoke
from auto_ontology.utils.llm_invoke import (
    RETRY_MAX_ATTEMPTS,
    _build_client,
    _client_kind,
    _structured_output_kwargs,
    _template_kwargs,
    max_tokens_for,
    safe_invoke_with_structured_output,
)


class _Model:
    """Stand-in exposing the model name the way langchain clients do."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name


def test_claude_uses_function_calling_without_a_tool_choice() -> None:
    """Bedrock behind LiteLLM rejects any explicit tool choice.

    The gateway maps tool_choice into ``toolConfig.toolChoice`` *and* forwards the
    original ``tool_choice.type``, and Bedrock refuses the pair:

        The additional field tool_choice/type conflicts with the existing field
        toolConfig.toolChoice.tool.

    Measured against the live endpoint: "auto", "any", "required" and langchain's
    default (naming the tool) all hit it; only omitting the choice succeeds.
    """
    kwargs = _structured_output_kwargs(_Model("aws/anthropic/bedrock-claude-opus-4-8"))

    assert kwargs == {"method": "function_calling", "tool_choice": None}


def test_claude_is_matched_on_either_vendor_or_family_name() -> None:
    for name in (
        "aws/anthropic/bedrock-claude-opus-4-8",
        "claude-3-5-sonnet",
        "ANTHROPIC/Claude",
    ):
        assert _structured_output_kwargs(_Model(name))["method"] == "function_calling"


def test_other_models_keep_the_langchain_default() -> None:
    """json_schema is preferred where supported, so nothing is forced for them."""
    for name in ("nvidia/nemotron-3-nano-30b-a3b", "gpt-4o", ""):
        assert _structured_output_kwargs(_Model(name)) == {}


def test_a_client_exposing_model_instead_of_model_name_still_matches() -> None:
    """Some langchain clients expose `model` rather than `model_name`."""

    class _AltModel:
        model: Any = "aws/anthropic/bedrock-claude-opus-4-8"

    assert _structured_output_kwargs(_AltModel())["tool_choice"] is None


def _bound() -> int | None:
    """Capacity of the bound currently in force, or None when unbounded."""
    guard = llm_invoke._INFLIGHT
    return None if guard is None else guard._initial_value  # type: ignore[attr-defined]


def test_limit_inflight_applies_and_releases_a_bound() -> None:
    assert _bound() is None
    with llm_invoke.limit_inflight(6):
        assert _bound() == 6
    assert _bound() is None


def test_overlapping_blocks_do_not_leak_a_bound() -> None:
    """Two blocks that overlap without nesting exit in non-stack order.

    A save/restore of one global has each block put back the value it captured,
    which drops the bound of a block still running and then leaves a stale bound
    installed with nothing active. Both are silent, and the second permanently
    re-imposes the ceiling this module exists to avoid.
    """
    a_inside = threading.Event()
    b_inside = threading.Event()
    seen_by_b: list[int | None] = []

    def first() -> None:
        with llm_invoke.limit_inflight(6):
            a_inside.set()
            b_inside.wait(5)  # leave only once B is also inside

    def second() -> None:
        a_inside.wait(5)
        with llm_invoke.limit_inflight(6):
            b_inside.set()
            time.sleep(0.2)  # outlive A
            seen_by_b.append(_bound())

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert seen_by_b == [6], "B lost its bound when A exited"
    assert _bound() is None, "a bound survived with no block active"


def test_the_most_restrictive_overlapping_bound_wins() -> None:
    with llm_invoke.limit_inflight(8):
        assert _bound() == 8
        with llm_invoke.limit_inflight(2):
            assert _bound() == 2
        # Leaving the tighter block must not leave the process unbounded.
        assert _bound() == 8
    assert _bound() is None


def test_a_non_positive_limit_leaves_another_blocks_bound_alone() -> None:
    """ "I need no bound" must not mean "remove someone else's"."""
    with llm_invoke.limit_inflight(4):
        with llm_invoke.limit_inflight(0):
            assert _bound() == 4
        assert _bound() == 4
    assert _bound() is None


def test_an_exception_inside_the_block_still_releases_the_bound() -> None:
    with contextlib.suppress(RuntimeError):
        with llm_invoke.limit_inflight(3):
            assert _bound() == 3
            raise RuntimeError("boom")
    assert _bound() is None


# ── Client selection, token caps and thinking ─────────────────────────────────

_ROLE_ENV = [
    f"{prefix}_{field}"
    for prefix in ("REASONING", "NON_REASONING", "DEFAULT_MODELS")
    for field in ("API_KEY", "CLIENT", "MAX_TOKENS", "THINKING")
] + ["NVIDIA_API_KEY"]


@pytest.fixture
def role_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """An environment with none of the role settings these tests exercise."""
    for name in _ROLE_ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_sk_keys_default_to_the_openai_client(role_env: pytest.MonkeyPatch) -> None:
    """A Gym model server does not serve /models, which ChatNVIDIA requests first.

    inference-api (``sk-`` keys) is OpenAI-compatible for every model it serves,
    so an ``nvidia/`` model there goes through ChatOpenAI by default.
    """
    role_env.setenv("REASONING_API_KEY", "sk-test")

    assert _client_kind("REASONING", "nvidia/nvidia/nemotron-3.5-lightning") == "openai"


def test_nvapi_keys_keep_chat_nvidia(role_env: pytest.MonkeyPatch) -> None:
    role_env.setenv("REASONING_API_KEY", "nvapi-test")

    assert _client_kind("REASONING", "nvidia/nemotron-3-nano-30b-a3b") == "nvidia"


def test_client_is_overridable_per_role(role_env: pytest.MonkeyPatch) -> None:
    role_env.setenv("DEFAULT_MODELS_API_KEY", "nvapi-test")
    role_env.setenv("NON_REASONING_CLIENT", "OpenAI")

    assert _client_kind("NON_REASONING", "nvidia/nemotron-3-nano-30b-a3b") == "openai"
    assert _client_kind("REASONING", "nvidia/nemotron-3-nano-30b-a3b") == "nvidia"


def test_openai_family_names_ignore_the_client_setting(
    role_env: pytest.MonkeyPatch,
) -> None:
    """Configs that predate <role>_CLIENT relied on the name prefix alone."""
    role_env.setenv("REASONING_CLIENT", "nvidia")

    for name in ("openai/gpt-5", "azure/openai/gpt-5", "aws/anthropic/claude"):
        assert _client_kind("REASONING", name) == "openai"


def test_an_unknown_client_is_rejected(role_env: pytest.MonkeyPatch) -> None:
    role_env.setenv("REASONING_CLIENT", "anthropic")

    with pytest.raises(ValueError, match="REASONING_CLIENT"):
        _client_kind("REASONING", "nvidia/x")


def test_per_call_caps_apply_when_no_override_is_set(
    role_env: pytest.MonkeyPatch,
) -> None:
    assert max_tokens_for("REASONING", 1024) == 1024


def test_max_tokens_replaces_every_cap_for_its_role(
    role_env: pytest.MonkeyPatch,
) -> None:
    """Thinking counts against the cap, so a 1024 cap sized for the answer truncates."""
    role_env.setenv("REASONING_MAX_TOKENS", "32768")

    assert max_tokens_for("REASONING", 1024) == 32768
    assert max_tokens_for("REASONING", 8192) == 32768
    assert max_tokens_for("NON_REASONING", 1024) == 1024


def test_max_tokens_falls_back_to_the_shared_default(
    role_env: pytest.MonkeyPatch,
) -> None:
    role_env.setenv("DEFAULT_MODELS_MAX_TOKENS", "16384")

    assert max_tokens_for("NON_REASONING", 2048) == 16384


def test_a_malformed_max_tokens_is_rejected(role_env: pytest.MonkeyPatch) -> None:
    for value in ("lots", "0", "-1"):
        role_env.setenv("REASONING_MAX_TOKENS", value)
        with pytest.raises(ValueError, match="REASONING_MAX_TOKENS"):
            max_tokens_for("REASONING", 1024)


def test_thinking_is_sent_only_when_set(role_env: pytest.MonkeyPatch) -> None:
    assert _template_kwargs("REASONING") == {}

    role_env.setenv("REASONING_THINKING", "off")
    assert _template_kwargs("REASONING") == {
        "chat_template_kwargs": {"enable_thinking": False}
    }

    role_env.setenv("REASONING_THINKING", "ON")
    assert _template_kwargs("REASONING") == {
        "chat_template_kwargs": {"enable_thinking": True}
    }


def test_an_unknown_thinking_value_is_rejected(role_env: pytest.MonkeyPatch) -> None:
    role_env.setenv("REASONING_THINKING", "low")

    with pytest.raises(ValueError, match="REASONING_THINKING"):
        _template_kwargs("REASONING")


def test_openai_client_carries_the_cap_and_thinking(
    role_env: pytest.MonkeyPatch,
) -> None:
    role_env.setenv("NON_REASONING_MAX_TOKENS", "20000")
    role_env.setenv("NON_REASONING_THINKING", "off")

    client = _build_client(
        role="NON_REASONING",
        model="nvidia/nvidia/nemotron-3.5-lightning",
        api_key="sk-test",
        base_url="http://127.0.0.1:1/v1",
        temperature=0.0,
        max_tokens=1024,
    )

    assert type(client).__name__ == "ChatOpenAI"
    assert client.max_tokens == 20000
    assert client.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_openai_client_sends_no_extra_body_by_default(
    role_env: pytest.MonkeyPatch,
) -> None:
    """A gateway fronting Claude may reject chat_template_kwargs, so none is sent."""
    client = _build_client(
        role="REASONING",
        model="aws/anthropic/bedrock-claude-opus-4-8",
        api_key="sk-test",
        base_url="http://127.0.0.1:1/v1",
        temperature=0.0,
        max_tokens=8192,
    )

    assert client.max_tokens == 8192
    assert not client.extra_body


# ── Retrying cut-off and unparseable answers ──────────────────────────────────


class LengthFinishReasonError(Exception):
    """Stand-in for openai's: it is matched by name, and the real one needs a completion."""


class _Answer(BaseModel):
    value: str


class _ScriptedLLM:
    """A client whose structured calls play back *outcomes*: an exception to raise, or a result."""

    model_name = "nvidia/nvidia/nemotron-3.5-lightning"

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def with_structured_output(self, schema: Any, **kwargs: Any) -> "_ScriptedLLM":
        return self

    def invoke(self, messages: Any) -> Any:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


_QUESTION = [HumanMessage(content="q")]


def test_a_cut_off_answer_is_retried() -> None:
    """A capped answer is a bad draw: the whitespace runaway rarely repeats."""
    llm = _ScriptedLLM([LengthFinishReasonError(), _Answer(value="ok")])

    result = safe_invoke_with_structured_output(llm, _QUESTION, _Answer)  # type: ignore[arg-type]

    assert result == _Answer(value="ok")
    assert llm.calls == 2


def test_an_unparseable_tool_call_is_retried() -> None:
    llm = _ScriptedLLM([OutputParserException("bad args"), _Answer(value="ok")])

    assert safe_invoke_with_structured_output(llm, _QUESTION, _Answer).value == "ok"  # type: ignore[arg-type]
    assert llm.calls == 2


def test_a_cut_off_answer_every_time_gives_up_after_all_attempts() -> None:
    llm = _ScriptedLLM([LengthFinishReasonError() for _ in range(RETRY_MAX_ATTEMPTS)])

    with pytest.raises(LengthFinishReasonError):
        safe_invoke_with_structured_output(llm, _QUESTION, _Answer)  # type: ignore[arg-type]
    assert llm.calls == RETRY_MAX_ATTEMPTS


def test_a_non_retryable_error_still_stops_at_once() -> None:
    llm = _ScriptedLLM([KeyError("boom"), _Answer(value="unreached")])

    with pytest.raises(KeyError):
        safe_invoke_with_structured_output(llm, _QUESTION, _Answer)  # type: ignore[arg-type]
    assert llm.calls == 1
