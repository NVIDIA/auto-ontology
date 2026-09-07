# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for Column.sample_values stored on catalog nodes."""

from __future__ import annotations

import json
from typing import Any


def parse_sample_values(raw: Any) -> list[Any] | None:
    """Normalize Column.sample_values (JSON string or list) to a Python list.

    Values keep the type they were profiled as (see
    ``gsf.semantic.visit_enter._json_ready_sample``): the column stores JSON,
    which carries the type with the value, so a numeric column reads back as
    ``[10, 20]`` rather than ``["10", "20"]``. A list is accepted as well as
    the stored string, since a catalog PATCH hands one over directly.
    ``None`` / JSON ``null`` entries are dropped so no consumer has to guard
    against them.

    Callers that need display text want ``stringify_sample_values`` instead —
    that is what the API returns and what prompts render.
    """
    if raw is None:
        return None
    if isinstance(raw, list):
        values: list[Any] = raw
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
    return [value for value in values if value is not None]


def stringify_sample_values(
    raw: Any,
    *,
    max_len: int | None = None,
) -> list[str] | None:
    """Render sample values as the string list prompts and the API expect.

    ``None`` passes through, so a column with no ``sample_values`` property
    stays distinguishable from one profiled as having no samples. *max_len*
    drops values whose rendered form exceeds the cap, which is what keeps
    prose and blobs out of prompts.
    """
    values = parse_sample_values(raw)
    if values is None:
        return None
    rendered = [render_sample_value(value) for value in values]
    if max_len is None:
        return rendered
    return [value for value in rendered if len(value) <= max_len]


def dump_sample_values(values: list[Any]) -> str | None:
    """Encode sample values for storage, or ``None`` when there is nothing to store.

    The inverse of :func:`parse_sample_values`, and the one place a write of
    ``Column.sample_values`` is encoded — whether the values were profiled
    from a live warehouse or supplied by a model import.

    Types are preserved rather than coerced to text: JSON distinguishes ``10``
    from ``"10"``, so a numeric column stays numeric and readers can tell it
    from a text column that happens to hold digits. Containers (the lists and
    dicts Postgres returns for array and JSON columns) nest as themselves.

    ``None`` entries are dropped, and a list left empty by that yields ``None``
    so the caller can skip the write: an empty list reads as "this column has
    no values", a different claim from "we did not profile it".
    """
    kept = [value for value in values if value is not None]
    if not kept:
        return None
    try:
        return json.dumps(kept, default=render_sample_value)
    except (TypeError, ValueError):
        return None


def render_sample_value(value: Any) -> str:
    """Render one sample value as display text.

    Non-scalars go through ``json.dumps`` rather than ``str()`` so a JSONB
    sample reads as ``{"a": 1}`` instead of Python's ``{'a': 1}``.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, int, float)):
        return str(value)
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)
