# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for Column.sample_values stored on catalog nodes."""

from __future__ import annotations

import json
from typing import Any


def parse_sample_values(raw: Any) -> list[str] | None:
    """Normalize Column.sample_values (JSON string or list) to a string list.

    Profiling persists ``col.sample_values`` as a JSON string (see
    ``store_column_sample_values``); catalog PATCH may store a list. Callers
    expect ``list[str] | None``.
    """
    if raw is None:
        return None
    if isinstance(raw, list):
        values = raw
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return None
        if not isinstance(parsed, list):
            return None
        values = parsed
    else:
        return None
    return [str(value) for value in values]
