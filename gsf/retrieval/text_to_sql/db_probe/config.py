# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for live DB probing.

All knobs are environment-driven so probing can be tuned without code changes.
The empty-result value-repair node is always wired in; the proactive
pre-execution literal check remains opt-in via ``DB_PROBE_PROACTIVE``.
"""

from __future__ import annotations

import os

_TRUTHY = {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def is_db_probe_proactive() -> bool:
    """Whether the *proactive* pre-execution literal check is wired in.

    Off by default: unlike the empty-result check this runs on every query with
    a categorical filter (a few cheap DISTINCT probes), so it trades some
    always-on cost for catching wrong literals that would return non-empty but
    wrong rows. Opt-in via ``DB_PROBE_PROACTIVE``.
    """
    return os.environ.get("DB_PROBE_PROACTIVE", "").strip().lower() in _TRUTHY


# Hard caps — keep Phase 1 cheap and always-on-safe.
# Total read-only queries allowed per question.
DB_PROBE_MAX_CALLS = _int_env("DB_PROBE_MAX_CALLS", 40)
# Max rows returned per probe (also the auto-appended LIMIT).
DB_PROBE_MAX_ROWS = _int_env("DB_PROBE_MAX_ROWS", 50)
# Per-probe wall-clock timeout (seconds).
DB_PROBE_TIMEOUT_S = _float_env("DB_PROBE_TIMEOUT_S", 5.0)
# Only probe the first N relevant tables.
DB_PROBE_MAX_TABLES = _int_env("DB_PROBE_MAX_TABLES", 8)
# Only probe the first N columns per table (per probe kind).
DB_PROBE_MAX_COLS_PER_TABLE = _int_env("DB_PROBE_MAX_COLS_PER_TABLE", 8)
# A DISTINCT probe is kept only if the column has <= this many distinct values.
DB_PROBE_LOW_CARD_THRESHOLD = _int_env("DB_PROBE_LOW_CARD_THRESHOLD", 20)


__all__ = [
    "is_db_probe_proactive",
    "DB_PROBE_MAX_CALLS",
    "DB_PROBE_MAX_ROWS",
    "DB_PROBE_TIMEOUT_S",
    "DB_PROBE_MAX_TABLES",
    "DB_PROBE_MAX_COLS_PER_TABLE",
    "DB_PROBE_LOW_CARD_THRESHOLD",
]
