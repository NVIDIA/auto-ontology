# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the generator is reached and how it is asked to reason."""

from __future__ import annotations

import types
from typing import Any

import pytest

from unstructured_ao import answer

_PREFIX = "UNSTRUCTURED"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        f"{_PREFIX}_API_KEY",
        f"{_PREFIX}_ENDPOINT",
        f"{_PREFIX}_MODEL",
        "DEFAULT_MODELS_API_KEY",
        "DEFAULT_MODELS_ENDPOINT",
        "DEFAULT_MODELS_MODEL",
        "NVIDIA_API_KEY",
        "BASE_URL",
        "MODEL_NAME",
    ):
        monkeypatch.delenv(name, raising=False)


def test_explicit_triplet_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(f"{_PREFIX}_API_KEY", "sk-own")
    monkeypatch.setenv(f"{_PREFIX}_ENDPOINT", "https://custom.example/v1/")
    monkeypatch.setenv(f"{_PREFIX}_MODEL", "custom/model")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-shared")

    assert answer._resolved() == {
        "endpoint": "https://custom.example/v1",
        "api_key": "sk-own",
        "model": "custom/model",
    }


def test_legacy_triplet_is_used_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "sk-legacy")
    monkeypatch.setenv("BASE_URL", "https://inference-api.nvidia.com/v1")
    monkeypatch.setenv("MODEL_NAME", "aws/anthropic/some-model")

    assert answer._resolved() == {
        "endpoint": "https://inference-api.nvidia.com/v1",
        "api_key": "sk-legacy",
        "model": "aws/anthropic/some-model",
    }


def test_a_bare_key_picks_the_matching_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-legacy")
    assert answer._resolved() == {
        "endpoint": "https://integrate.api.nvidia.com/v1",
        "api_key": "nvapi-legacy",
        "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
    }


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("https://integrate.api.nvidia.com/v1/", {"reasoning": {"effort": "low"}}),
        ("https://inference-api.nvidia.com/v1/", {"reasoning_effort": "low"}),
    ],
)
def test_reasoning_is_spelled_for_the_endpoint(
    endpoint: str, expected: dict[str, Any]
) -> None:
    """The gateway 400s on ``reasoning``; integrate ignores ``reasoning_effort``."""
    assert answer.reasoning_body(endpoint) == expected


_MESSAGES = [{"role": "user", "content": "?"}]


def _fake_client(base_url: str, sent: dict[str, Any]) -> Any:
    def create(**kwargs: Any) -> Any:
        sent.update(kwargs)
        message = types.SimpleNamespace(content="42")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    return types.SimpleNamespace(
        base_url=base_url,
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)),
    )


@pytest.fixture
def unpaced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(answer.GEN_LIMITER, "acquire", lambda _slots=1: None)


@pytest.mark.usefixtures("unpaced")
def test_a_passed_client_uses_the_model_passed_with_it() -> None:
    sent: dict[str, Any] = {}
    client = _fake_client("https://inference-api.nvidia.com/v1/", sent)

    assert answer.complete(_MESSAGES, client=client, model="own/model") == ("42", None)
    assert sent["model"] == "own/model"
    assert sent["extra_body"] == {"reasoning_effort": "low"}


def test_a_passed_client_without_its_model_is_refused() -> None:
    client = _fake_client("https://inference-api.nvidia.com/v1/", {})
    with pytest.raises(ValueError, match="model"):
        answer.complete(_MESSAGES, client=client)


@pytest.mark.usefixtures("unpaced")
def test_the_shared_client_keeps_the_model_it_was_built_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing the environment later cannot pair the endpoint with another model."""
    monkeypatch.setattr(answer, "_client", None)
    monkeypatch.setattr(answer, "_model", "")
    monkeypatch.setenv("NVIDIA_API_KEY", "sk-gateway")
    answer.get_client()
    sent: dict[str, Any] = {}
    monkeypatch.setattr(
        answer, "_client", _fake_client("https://inference-api.nvidia.com/v1/", sent)
    )
    monkeypatch.setenv(f"{_PREFIX}_MODEL", "changed/later")

    assert answer.complete(_MESSAGES) == ("42", None)
    assert sent["model"] == "nvidia/nvidia/nemotron-3-super-120b-long-ctx"


def test_no_key_names_the_variables_that_would_supply_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(answer, "_client", None)
    monkeypatch.setattr(answer, "_model", "")
    with pytest.raises(RuntimeError, match=f"{_PREFIX}_API_KEY"):
        answer.get_client()
