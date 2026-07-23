# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIM reranking config shared by the rerank flow.

Mirrors :mod:`gsf.utils.embedding`: a single place that reads the endpoint,
model, and API key from the environment and hands back the keyword args
``nemo_retriever.operators.rerank.rerank_hits`` expects.
"""

from __future__ import annotations

from gsf.utils.model_config import resolve

# Hosted NeMo reranking endpoint — no local GPU required. Override RERANK_ENDPOINT
# to point at a self-hosted vLLM/NIM ranking server. Each field falls back to
# DEFAULT_AGENT_<field> when unset.
_RERANK_ENDPOINT = resolve("RERANK", "ENDPOINT")
_RERANK_MODEL = resolve("RERANK", "MODEL")
_RERANK_API_KEY = resolve("RERANK", "API_KEY")


def get_rerank_kwargs() -> dict[str, str]:
    """Keyword args for ``rerank_hits`` (remote NeMo reranking endpoint)."""
    return {
        "rerank_invoke_url": _RERANK_ENDPOINT,
        "model_name": _RERANK_MODEL,
        "api_key": _RERANK_API_KEY,
    }
