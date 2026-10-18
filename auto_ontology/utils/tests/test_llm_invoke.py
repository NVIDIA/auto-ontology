# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for LLM invocation helpers."""

import contextlib
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from auto_ontology.utils import llm_invoke
from auto_ontology.utils.llm_invoke import _structured_output_kwargs


@pytest.fixture(autouse=True)
def _isolate_fallback_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent client identities and cached fallback clients leaking across tests."""
    monkeypatch.setattr(llm_invoke, "_NON_REASONING_IDS", set())
    monkeypatch.setattr(llm_invoke, "_NON_REASONING_STRONG_REFS", {})
    monkeypatch.setattr(llm_invoke, "_REASONING_FALLBACK_CLIENT", None)


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


class _Schema:
    """Stand-in schema; the structured-invoke mock never validates it."""


class _Binding:
    """Stand-in for ``llm.bind()``, which wraps the client as ``bound``."""

    def __init__(self, bound: object) -> None:
        self.bound = bound


def test_non_reasoning_client_is_marked(monkeypatch: Any) -> None:
    sentinel = object()
    monkeypatch.setattr(llm_invoke, "_NON_REASONING_API_KEY", "sk-test")
    monkeypatch.setattr(llm_invoke, "_build_client", lambda **_kwargs: sentinel)

    client = llm_invoke.get_non_reasoning_llm_client()

    assert client is sentinel
    assert llm_invoke._is_non_reasoning_client(client)
    assert llm_invoke._is_non_reasoning_client(_Binding(client))
    assert not llm_invoke._is_non_reasoning_client(object())
    assert not llm_invoke._is_non_reasoning_client(_Binding(object()))


def test_endless_bound_chain_is_not_followed_forever() -> None:
    """A ``MagicMock`` has a fresh ``bound`` on every access; the walk must stop."""
    assert not llm_invoke._is_non_reasoning_client(MagicMock())


def test_non_weakref_client_id_cannot_match_an_unrelated_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = object()
    registered = object()
    monkeypatch.setattr(llm_invoke, "_NON_REASONING_IDS", {id(candidate)})
    monkeypatch.setattr(
        llm_invoke,
        "_NON_REASONING_STRONG_REFS",
        {id(candidate): registered},
    )

    assert not llm_invoke._is_non_reasoning_client(candidate)


def test_structured_non_reasoning_call_falls_back_to_reasoning(
    monkeypatch: Any,
) -> None:
    primary = object()
    reasoning = object()
    llm_invoke._remember_non_reasoning(primary)
    calls: list[object] = []

    def invoke(llm: object, _messages: object, _schema: object) -> str:
        calls.append(llm)
        if llm is primary:
            raise RuntimeError("429 Too Many Requests")
        return "from-reasoning"

    monkeypatch.setattr(llm_invoke, "safe_invoke_with_structured_output", invoke)
    monkeypatch.setattr(llm_invoke, "get_llm_client", lambda **_kwargs: reasoning)

    result = llm_invoke.invoke_with_structured_output(primary, [], _Schema)

    assert result == "from-reasoning"
    assert calls == [primary, reasoning]


def test_bound_non_reasoning_client_falls_back(monkeypatch: Any) -> None:
    primary = object()
    reasoning = object()
    binding = _Binding(primary)
    llm_invoke._remember_non_reasoning(primary)
    calls: list[object] = []

    def invoke(llm: object, _messages: object, _schema: object) -> str:
        calls.append(llm)
        if llm is not reasoning:
            raise RuntimeError("down")
        return "ok"

    monkeypatch.setattr(llm_invoke, "safe_invoke_with_structured_output", invoke)
    monkeypatch.setattr(llm_invoke, "get_llm_client", lambda **_kwargs: reasoning)

    result = llm_invoke.invoke_with_structured_output(binding, [], _Schema)

    assert result == "ok"
    assert calls == [binding, reasoning]


def test_reasoning_fallback_client_is_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = object()
    second = object()
    reasoning = object()
    builds: list[object] = []
    llm_invoke._remember_non_reasoning(first)
    llm_invoke._remember_non_reasoning(second)
    monkeypatch.setattr(
        llm_invoke,
        "get_llm_client",
        lambda **_kwargs: builds.append(reasoning) or reasoning,
    )

    assert llm_invoke._reasoning_fallback_client(first) is reasoning
    assert llm_invoke._reasoning_fallback_client(second) is reasoning
    assert builds == [reasoning]


def test_structured_fallback_returns_none_when_reasoning_also_fails(
    monkeypatch: Any,
) -> None:
    primary = object()
    reasoning = object()
    llm_invoke._remember_non_reasoning(primary)
    calls: list[object] = []

    def invoke(llm: object, _messages: object, _schema: object) -> None:
        calls.append(llm)
        raise RuntimeError("429")

    monkeypatch.setattr(llm_invoke, "safe_invoke_with_structured_output", invoke)
    monkeypatch.setattr(llm_invoke, "get_llm_client", lambda **_kwargs: reasoning)
    monkeypatch.setattr(llm_invoke, "_NON_REASONING_API_KEY", "sk-test")
    monkeypatch.setattr(llm_invoke, "_build_client", lambda **_kwargs: primary)

    result = llm_invoke.safe_invoke_structured_nr([], _Schema)

    assert result is None
    assert calls == [primary, reasoning]


def test_structured_nr_uses_reasoning_when_non_reasoning_client_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reasoning = object()
    calls: list[object] = []

    monkeypatch.setattr(
        llm_invoke,
        "get_non_reasoning_llm_client",
        lambda **_kwargs: (_ for _ in ()).throw(EnvironmentError("missing key")),
    )
    monkeypatch.setattr(llm_invoke, "get_llm_client", lambda **_kwargs: reasoning)
    monkeypatch.setattr(
        llm_invoke,
        "safe_invoke_with_structured_output",
        lambda llm, _messages, _schema: calls.append(llm) or "ok",
    )

    assert llm_invoke.safe_invoke_structured_nr([], _Schema) == "ok"
    assert calls == [reasoning]


def test_structured_nr_returns_none_when_neither_client_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(**_kwargs: object) -> object:
        raise EnvironmentError("missing key")

    monkeypatch.setattr(llm_invoke, "get_non_reasoning_llm_client", unavailable)
    monkeypatch.setattr(llm_invoke, "get_llm_client", unavailable)

    assert llm_invoke.safe_invoke_structured_nr([], _Schema) is None


def test_reasoning_structured_call_does_not_switch_models(monkeypatch: Any) -> None:
    reasoning = object()

    def invoke(_llm: object, _messages: object, _schema: object) -> None:
        raise RuntimeError("429")

    def fail_client(**_kwargs: object) -> object:
        raise AssertionError("reasoning call must not build a fallback client")

    monkeypatch.setattr(llm_invoke, "safe_invoke_with_structured_output", invoke)
    monkeypatch.setattr(llm_invoke, "get_llm_client", fail_client)

    assert llm_invoke.invoke_with_structured_output(reasoning, [], _Schema) is None


def test_text_non_reasoning_call_falls_back_once(monkeypatch: Any) -> None:
    primary = object()
    reasoning = object()
    llm_invoke._remember_non_reasoning(primary)
    calls: list[object] = []

    def once(llm: object, _prompt: str) -> str:
        calls.append(llm)
        if llm is primary:
            raise RuntimeError("connection error")
        return "ok"

    monkeypatch.setattr(llm_invoke, "_safe_invoke_text", once)
    monkeypatch.setattr(llm_invoke, "get_llm_client", lambda **_kwargs: reasoning)

    assert llm_invoke.safe_invoke_text(primary, "question") == "ok"
    assert calls == [primary, reasoning]


def test_text_nr_uses_reasoning_when_non_reasoning_client_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reasoning = object()
    calls: list[object] = []

    monkeypatch.setattr(
        llm_invoke,
        "get_non_reasoning_llm_client",
        lambda **_kwargs: (_ for _ in ()).throw(EnvironmentError("missing key")),
    )
    monkeypatch.setattr(llm_invoke, "get_llm_client", lambda **_kwargs: reasoning)
    monkeypatch.setattr(
        llm_invoke,
        "_safe_invoke_text",
        lambda llm, _prompt: calls.append(llm) or "ok",
    )

    assert llm_invoke.safe_invoke_text_nr("question") == "ok"
    assert calls == [reasoning]


def test_text_fallback_is_not_attempted_twice(monkeypatch: Any) -> None:
    primary = object()
    reasoning = object()
    calls: list[object] = []

    def once(llm: object, _prompt: str) -> str:
        calls.append(llm)
        raise RuntimeError("429")

    monkeypatch.setattr(llm_invoke, "_NON_REASONING_API_KEY", "sk-test")
    monkeypatch.setattr(llm_invoke, "_build_client", lambda **_kwargs: primary)
    monkeypatch.setattr(llm_invoke, "_safe_invoke_text", once)
    monkeypatch.setattr(llm_invoke, "get_llm_client", lambda **_kwargs: reasoning)

    try:
        llm_invoke.safe_invoke_text_nr("question")
    except RuntimeError as exc:
        assert "429" in str(exc)
    else:
        raise AssertionError("both models failing must raise")

    assert calls == [primary, reasoning]


def test_invoke_text_falls_back_for_a_non_reasoning_client(monkeypatch: Any) -> None:
    primary = object()
    reasoning = object()
    llm_invoke._remember_non_reasoning(primary)
    calls: list[object] = []

    def once(llm: object, _prompt: str) -> str:
        calls.append(llm)
        if llm is primary:
            raise RuntimeError("down")
        return "pql"

    monkeypatch.setattr(llm_invoke, "_invoke_text_once", once)
    monkeypatch.setattr(llm_invoke, "get_llm_client", lambda **_kwargs: reasoning)

    assert llm_invoke.invoke_text(primary, "predict") == "pql"
    assert calls == [primary, reasoning]


def test_missing_reasoning_client_keeps_the_original_failure(monkeypatch: Any) -> None:
    primary = object()
    llm_invoke._remember_non_reasoning(primary)

    def invoke(_llm: object, _messages: object, _schema: object) -> None:
        raise RuntimeError("429")

    def unavailable(**_kwargs: object) -> object:
        raise EnvironmentError("REASONING_API_KEY is not set")

    monkeypatch.setattr(llm_invoke, "safe_invoke_with_structured_output", invoke)
    monkeypatch.setattr(llm_invoke, "get_llm_client", unavailable)

    assert llm_invoke.invoke_with_structured_output(primary, [], _Schema) is None
