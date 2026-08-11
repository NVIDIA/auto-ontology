# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Zone constants, graph label helpers, and API payload formatting."""

from __future__ import annotations

from gsf.catalog.constants import Edges, Labels

from gsf.server.zones.constants import LABEL_ZONE

REL_CONTAINS = Edges.CONTAINS

ZONE_DATA_LABELS = (
    Labels.DB,
    Labels.SCHEMA,
    Labels.TABLE,
)

_GSF_LABEL_TO_API_NAME: dict[str, str] = {
    Labels.DB: "db",
    Labels.SCHEMA: "schema",
    Labels.TABLE: "table",
    LABEL_ZONE: "zone",
}


def _resolve_api_name(gsf_label: str) -> str:
    """Map a GSF graph label to its API-facing name."""
    return _GSF_LABEL_TO_API_NAME.get(gsf_label, gsf_label.lower())


def format_zone(
    row: dict,
    *,
    items: list[dict] | list[str] | None = None,
    enabled: bool | None = None,
) -> dict:
    """Shape a zone row for API responses."""
    result = {
        "id": row["id"],
        "name": row["name"],
        "description": row.get("description"),
        "color": row.get("color"),
        "label": LABEL_ZONE,
        "enabled": enabled if enabled is not None else row.get("enabled", True),
    }
    if items is not None:
        result["items"] = items
    return result
