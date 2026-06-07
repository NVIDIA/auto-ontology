# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIM text-embedding config shared by ingest pipelines and the retriever."""

from __future__ import annotations

import os

from nemo_retriever.params import EmbedParams

# Remote NIM embedding endpoint — no local GPU required.
# MUST match the model used at ingest time; a mismatch produces garbage results
# or a dimension error from pgvector.
_EMBED_ENDPOINT = os.environ.get(
    "EMBED_ENDPOINT", "https://integrate.api.nvidia.com/v1"
)
_EMBED_MODEL = os.environ.get("EMBED_MODEL", "nvidia/llama-nemotron-embed-1b-v2")
_NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")

if not _NVIDIA_API_KEY:
    raise EnvironmentError(
        "NVIDIA_API_KEY is not set. "
        "Export it before running:\n\n"
        "    export NVIDIA_API_KEY='nvapi-...'\n\n"
        "Get your key at https://build.nvidia.com"
    )


def get_embed_kwargs() -> dict[str, str]:
    """Keyword args for ``Retriever`` embed configuration."""
    return {
        "model_name": _EMBED_MODEL,
        "embed_invoke_url": _EMBED_ENDPOINT,
        "api_key": _NVIDIA_API_KEY,
    }


def get_embed_params() -> EmbedParams:
    return EmbedParams(
        embed_invoke_url=_EMBED_ENDPOINT,
        model_name=_EMBED_MODEL,
        api_key=_NVIDIA_API_KEY,
        embed_modality="text",
    )
