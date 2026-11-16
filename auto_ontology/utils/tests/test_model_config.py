# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in model defaults, chosen by the API key prefix."""

from __future__ import annotations

import pytest

from auto_ontology.utils import model_config

_PREFIXES = ("DEFAULT_MODELS", "REASONING", "NON_REASONING", "EMBED", "RERANK")
_FIELDS = ("ENDPOINT", "API_KEY", "MODEL")
_LEGACY = ("NVIDIA_API_KEY", "BASE_URL", "MODEL_NAME")


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for prefix in _PREFIXES:
        for field in _FIELDS:
            monkeypatch.delenv(f"{prefix}_{field}", raising=False)
    for name in _LEGACY:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


NVAPI_DEFAULTS = {
    ("REASONING", "MODEL"): "nvidia/nemotron-3.5-lightning-30b-a3b",
    ("REASONING", "ENDPOINT"): "https://integrate.api.nvidia.com/v1",
    ("NON_REASONING", "MODEL"): "nvidia/nemotron-3.5-lightning-30b-a3b",
    ("NON_REASONING", "ENDPOINT"): "https://integrate.api.nvidia.com/v1",
    ("EMBED", "MODEL"): "nvidia/nemotron-3.5-embed-1b",
    ("EMBED", "ENDPOINT"): "https://integrate.api.nvidia.com/v1",
    ("RERANK", "MODEL"): "nvidia/nemotron-3.5-rerank-8b",
    ("RERANK", "ENDPOINT"): (
        "https://ai.api.nvidia.com/v1/retrieval/nvidia/nemotron-3.5-rerank-8b/reranking"
    ),
}


@pytest.mark.parametrize(("role", "field"), sorted(NVAPI_DEFAULTS))
def test_an_nvapi_key_selects_the_nemotron_3_5_defaults(
    clean_env: pytest.MonkeyPatch, role: str, field: str
) -> None:
    clean_env.setenv("DEFAULT_MODELS_API_KEY", "nvapi-test")

    assert model_config.resolve(role, field) == NVAPI_DEFAULTS[(role, field)]


def test_an_sk_key_keeps_the_inference_api_defaults(
    clean_env: pytest.MonkeyPatch,
) -> None:
    clean_env.setenv("DEFAULT_MODELS_API_KEY", "sk-test")

    assert model_config.resolve("REASONING", "MODEL") == (
        "azure/anthropic/claude-opus-5-5"
    )
    assert model_config.resolve("EMBED", "MODEL") == (
        "nvidia/nvidia/nemotron-3.5-embed-1b-nvfp4"
    )


def test_an_explicit_setting_wins_over_the_default(
    clean_env: pytest.MonkeyPatch,
) -> None:
    clean_env.setenv("DEFAULT_MODELS_API_KEY", "nvapi-test")
    clean_env.setenv("REASONING_MODEL", "nvidia/some-other-model")

    assert model_config.resolve("REASONING", "MODEL") == "nvidia/some-other-model"
