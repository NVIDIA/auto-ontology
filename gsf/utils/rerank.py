# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIM reranking config shared by the rerank flow.

Mirrors :mod:`gsf.utils.embedding`: a single place that reads the endpoint,
model, and API key from the environment and hands back the keyword args
``nemo_retriever.operators.rerank.rerank_hits`` expects.
"""

from __future__ import annotations

import os

# Hosted NeMo reranking endpoint — no local GPU required. Override RERANK_ENDPOINT
# to point at a self-hosted vLLM/NIM ranking server.
_RERANK_ENDPOINT = os.environ.get(
    "RERANK_ENDPOINT",
    "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-1b-v2/reranking",
)
_RERANK_MODEL = os.environ.get("RERANK_MODEL", "nvidia/llama-nemotron-rerank-1b-v2")
_RERANK_API_KEY = os.environ.get("RERANK_API_KEY", "") or os.environ.get(
    "NVIDIA_API_KEY", ""
)

_KEY_ERROR = (
    "NVIDIA_API_KEY is not set. "
    "Export it (or RERANK_API_KEY) before running:\n\n"
    "    export NVIDIA_API_KEY='nvapi-...'\n\n"
    "Get your key at https://build.nvidia.com"
)


def get_rerank_kwargs() -> dict[str, str]:
    """Keyword args for ``rerank_hits`` (remote NeMo reranking endpoint)."""
    if not _RERANK_API_KEY:
        raise EnvironmentError(_KEY_ERROR)
    return {
        "rerank_invoke_url": _RERANK_ENDPOINT,
        "model_name": _RERANK_MODEL,
        "api_key": _RERANK_API_KEY,
    }
