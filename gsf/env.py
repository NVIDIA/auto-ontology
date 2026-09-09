# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment bootstrap shared by all GSF process entrypoints."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_repo_root = Path(__file__).resolve().parent.parent

# BIRD_INTERACT is a master flag: set BIRD_INTERACT=true (with no per-flag
# overrides in .env) to turn on every flag this deployment has approved for
# BIRD-Interact runs, in one place, instead of setting each individually.
#
# ASK_OUTPUT_TYPE, INJECT_CONDITIONAL_OUTPUT_HINT, and CLARIFY_MAX_DISTANCE
# are deliberately NOT listed here — they already default to INTERACTIVE's
# value when unset (see entity_resolution.py, output_type.py,
# conditional_output.py), so setting INTERACTIVE=true covers them.
#
# Each entry is applied with os.environ.setdefault(), so an explicit value
# already set in .env (or the process environment) always wins over this
# master flag's default.
_BIRD_INTERACT_DEFAULTS: dict[str, str] = {
    "INTERACTIVE": "true",
    "DB_PROBE_JSONB_PATH_CHECK": "true",
    "DB_PROBE_JOIN_PATH_CHECK": "true",
    "DB_PROBE_PROACTIVE": "true",
    "DETECT_VACUOUS_GROUP_BY": "true",
    "RELEVANCE_FILTER_INCLUDE_COLUMNS": "true",
    "HUB_SIBLING_EXPANSION_ENABLED": "true",
    "HUB_SIBLING_CAP": "6",
    "TABLE_BRIDGE_RECONCILIATION_ENABLED": "true",
}


def load_env() -> None:
    """Load the repo-root ``.env`` file into the process environment.

    If BIRD_INTERACT is truthy after loading, also applies the approved
    BIRD-Interact flag defaults (see _BIRD_INTERACT_DEFAULTS) to any of them
    left unset by .env.
    """
    load_dotenv(_repo_root / ".env")
    if os.environ.get("BIRD_INTERACT", "").strip().lower() in ("true", "1"):
        for key, value in _BIRD_INTERACT_DEFAULTS.items():
            os.environ.setdefault(key, value)
