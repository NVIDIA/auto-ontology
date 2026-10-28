# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for model triplet default resolution."""

from __future__ import annotations

import pytest

from auto_ontology.utils.model_config import resolve


@pytest.mark.parametrize(
    ("api_key", "endpoint", "model"),
    [
        (
            "sk-test",
            "https://inference-api.nvidia.com/v1/ocr",
            "nvidia/nvidia/nemotron-page-elements-v3",
        ),
        (
            "nvapi-test",
            "https://ai.api.nvidia.com/v1/cv/nvidia",
            "nemotron-page-elements-v3",
        ),
    ],
)
def test_page_elements_uses_key_specific_defaults(
    monkeypatch: pytest.MonkeyPatch,
    api_key: str,
    endpoint: str,
    model: str,
) -> None:
    prefix = "UNSTRUCTURED_PAGE_ELEMENTS"
    monkeypatch.setenv(f"{prefix}_API_KEY", api_key)
    monkeypatch.delenv(f"{prefix}_ENDPOINT", raising=False)
    monkeypatch.delenv(f"{prefix}_MODEL", raising=False)

    assert resolve(prefix, "ENDPOINT", use_shared_default=False) == endpoint
    assert resolve(prefix, "MODEL", use_shared_default=False) == model


@pytest.mark.parametrize(
    ("api_key", "endpoint", "model"),
    [
        (
            "sk-test",
            "https://inference-api.nvidia.com/v1",
            "nvidia/nvidia/nemotron-3-super-120b-long-ctx",
        ),
        (
            "nvapi-test",
            "https://integrate.api.nvidia.com/v1",
            "nvidia/nemotron-3-super-120b-a12b",
        ),
    ],
)
def test_unstructured_answer_uses_key_specific_defaults(
    monkeypatch: pytest.MonkeyPatch,
    api_key: str,
    endpoint: str,
    model: str,
) -> None:
    prefix = "UNSTRUCTURED"
    for name in (
        f"{prefix}_API_KEY",
        f"{prefix}_ENDPOINT",
        f"{prefix}_MODEL",
        "DEFAULT_MODELS_API_KEY",
        "DEFAULT_MODELS_ENDPOINT",
        "DEFAULT_MODELS_MODEL",
        "BASE_URL",
        "MODEL_NAME",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", api_key)

    assert resolve(prefix, "API_KEY") == api_key
    assert resolve(prefix, "ENDPOINT") == endpoint
    assert resolve(prefix, "MODEL") == model


def test_shared_default_remains_the_normal_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBED_API_KEY", "sk-test")
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    monkeypatch.setenv("DEFAULT_MODELS_MODEL", "custom/shared-model")

    assert resolve("EMBED", "MODEL") == "custom/shared-model"
