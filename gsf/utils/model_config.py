# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared resolution of model-endpoint env var triplets.

Every triplet (``REASONING``, ``NON_REASONING``, ``EMBED``, ``RERANK``) exposes
three fields — ``ENDPOINT`` (URL), ``API_KEY``, and ``MODEL`` — as
``<PREFIX>_<FIELD>`` env vars. Any field left unset falls back to the
corresponding ``DEFAULT_AGENT_<FIELD>``, so a single ``DEFAULT_AGENT_*`` triplet
can supply the shared endpoint/key/model for all of them. API keys additionally
fall back to the legacy ``NVIDIA_API_KEY`` name for backward compatibility.
"""

from __future__ import annotations

import os


def _default(field: str) -> str:
    """``DEFAULT_AGENT_<field>``; API keys fall back to legacy ``NVIDIA_API_KEY``."""
    value = os.environ.get(f"DEFAULT_AGENT_{field}", "")
    if not value and field == "API_KEY":
        value = os.environ.get("NVIDIA_API_KEY", "")
    return value


def resolve(prefix: str, field: str, hardcoded_default: str = "") -> str:
    """Resolve one triplet field.

    Order of precedence: ``<prefix>_<field>`` env var, then the shared
    ``DEFAULT_AGENT_<field>``, then *hardcoded_default*.
    """
    return (
        os.environ.get(f"{prefix}_{field}", "") or _default(field) or hardcoded_default
    )
