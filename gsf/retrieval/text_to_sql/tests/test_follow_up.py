# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for bounded conversational follow-up resolution."""

from types import SimpleNamespace

from pytest import MonkeyPatch

from gsf.retrieval.text_to_sql import follow_up

_HISTORY = [
    {
        "question": "Show revenue for July",
        "response": "Revenue is aggregated by month.",
        "sql_code": "SELECT SUM(revenue) FROM sales WHERE month = 'July'",
    }
]


def test_no_history_skips_the_llm(monkeypatch: MonkeyPatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("resolver should not be invoked")

    monkeypatch.setattr(follow_up, "invoke_with_structured_output", fail_if_called)

    assert follow_up.resolve_follow_up(
        question="Show revenue", history=[], llm=object()
    ) == (False, "Show revenue")


def test_independent_question_is_never_rewritten(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        follow_up,
        "invoke_with_structured_output",
        lambda *args, **kwargs: SimpleNamespace(
            is_follow_up=False,
            standalone_question="A model-mutated question",
        ),
    )

    assert follow_up.resolve_follow_up(
        question="Count customers", history=_HISTORY, llm=object()
    ) == (False, "Count customers")


def test_follow_up_uses_standalone_question(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        follow_up,
        "invoke_with_structured_output",
        lambda *args, **kwargs: SimpleNamespace(
            is_follow_up=True,
            standalone_question="Show revenue for August",
        ),
    )

    assert follow_up.resolve_follow_up(
        question="What about August?", history=_HISTORY, llm=object()
    ) == (True, "Show revenue for August")


def test_resolver_failure_falls_back_to_original(monkeypatch: MonkeyPatch) -> None:
    def raise_error(*args, **kwargs):
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(follow_up, "invoke_with_structured_output", raise_error)

    assert follow_up.resolve_follow_up(
        question="What about August?", history=_HISTORY, llm=object()
    ) == (False, "What about August?")
