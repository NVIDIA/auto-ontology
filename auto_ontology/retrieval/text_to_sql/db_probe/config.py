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


def _bounded_int_env(name: str, default: int, hard_max: int) -> int:
    """Read a positive integer limit that deployments may only tighten."""

    return min(max(_int_env(name, default), 1), hard_max)


def _bounded_float_env(name: str, default: float, hard_max: float) -> float:
    """Read a positive float limit that deployments may only tighten."""

    return min(max(_float_env(name, default), 0.1), hard_max)


def is_db_probe_proactive() -> bool:
    """Whether the *proactive* pre-execution literal check is wired in.

    Off by default: unlike the empty-result check this runs on every query with
    a categorical filter (a few cheap DISTINCT probes), so it trades some
    always-on cost for catching wrong literals that would return non-empty but
    wrong rows. Opt-in via ``DB_PROBE_PROACTIVE``.
    """
    return os.environ.get("DB_PROBE_PROACTIVE", "").strip().lower() in _TRUTHY


def is_db_probe_jsonb_path_check() -> bool:
    """Whether the pre-execution JSONB key-path check is wired in.

    Off by default, same trade-off as ``is_db_probe_proactive``: it runs a few
    cheap ``jsonb_object_keys`` probes on every query that navigates JSONB via
    ``->``/``->>``, to catch a hallucinated key/container that would otherwise
    silently return NULL instead of erroring. Opt-in via
    ``DB_PROBE_JSONB_PATH_CHECK``. Postgres-only regardless of this flag.
    """
    return os.environ.get("DB_PROBE_JSONB_PATH_CHECK", "").strip().lower() in _TRUTHY


def is_db_probe_join_path_check() -> bool:
    """Whether the pre-execution join-path check is wired in.

    Off by default, same trade-off as the other proactive checks: on the
    common case (join already correct) it's a handful of cheap
    existence checks against the ``SEMANTIC_FK`` graph per query; it only
    falls back to a live value-overlap probe (a couple of extra DB
    round-trips) when the graph has no edge for a predicate at all. Catches
    a hallucinated join between two plausible-looking identifier columns,
    which is syntactically/semantically valid SQL and so silently returns
    the wrong rows instead of erroring. Opt-in via
    ``DB_PROBE_JOIN_PATH_CHECK``.
    """
    return os.environ.get("DB_PROBE_JOIN_PATH_CHECK", "").strip().lower() in _TRUTHY


# Hard caps — keep Phase 1 cheap and always-on-safe.
# Total read-only queries allowed per question.
DB_PROBE_MAX_CALLS = _bounded_int_env("DB_PROBE_MAX_CALLS", 40, 100)
# Max rows returned per probe (also the auto-appended LIMIT).
DB_PROBE_MAX_ROWS = _bounded_int_env("DB_PROBE_MAX_ROWS", 50, 100)
# Max result columns retained from a probe.
DB_PROBE_MAX_RESULT_COLUMNS = _bounded_int_env("DB_PROBE_MAX_RESULT_COLUMNS", 16, 32)
# Per-probe wall-clock timeout (seconds).
DB_PROBE_TIMEOUT_S = _bounded_float_env("DB_PROBE_TIMEOUT_S", 5.0, 30.0)
# Only probe the first N relevant tables.
DB_PROBE_MAX_TABLES = _bounded_int_env("DB_PROBE_MAX_TABLES", 8, 32)
# Only probe the first N columns per table (per probe kind).
DB_PROBE_MAX_COLS_PER_TABLE = _bounded_int_env("DB_PROBE_MAX_COLS_PER_TABLE", 8, 32)
# A DISTINCT probe is kept only if the column has <= this many distinct values.
DB_PROBE_LOW_CARD_THRESHOLD = _bounded_int_env("DB_PROBE_LOW_CARD_THRESHOLD", 20, 100)


__all__ = [
    "is_db_probe_proactive",
    "is_db_probe_jsonb_path_check",
    "is_db_probe_join_path_check",
    "DB_PROBE_MAX_CALLS",
    "DB_PROBE_MAX_ROWS",
    "DB_PROBE_MAX_RESULT_COLUMNS",
    "DB_PROBE_TIMEOUT_S",
    "DB_PROBE_MAX_TABLES",
    "DB_PROBE_MAX_COLS_PER_TABLE",
    "DB_PROBE_LOW_CARD_THRESHOLD",
]
