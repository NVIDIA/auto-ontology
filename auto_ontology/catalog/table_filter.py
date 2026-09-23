# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regex allow/deny filtering of relations at catalog-extraction time.

Why this exists: a warehouse routinely exposes relations that are listed in its
metadata but cannot actually be read — a Distributed table whose shard-local
table has drifted, a Kafka-engine queue, a staging table nobody should query.
Metadata succeeds on all of them, so they land in the catalog, retrieval offers
them, and every generated query against them fails. Naming them is the only
thing the operator can do; these two patterns are how.

Applied in :func:`auto_ontology.catalog.extract.create_dataframe`, which is the
one place every connector's relations pass through, so filtering is uniform
rather than per-connector the way the ``schemas`` allowlist has to be.

Semantics, kept deliberately small:

``table_deny_regex``
    Drop any relation whose name matches.
``table_allow_regex``
    When set, keep *only* relations whose name matches.
Deny wins
    A relation matching both is dropped. The narrower rule is the safer one,
    and it makes "allow this family, minus these two" expressible.

Matching is :func:`re.search` against the **unqualified** relation name, so
``_stage`` matches ``pg_perfbot_flat_all_stage`` without anchors — grep-like,
which is what an operator typing a pattern into a form expects. Anchor with
``^``/``$`` for exact matches. Matching is case-sensitive, because the engines
we target treat identifiers that way.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

logger = logging.getLogger(__name__)

ALLOW_KEY = "table_allow_regex"
DENY_KEY = "table_deny_regex"

#: Connection keys this module reads, for callers that persist or forward them.
FILTER_KEYS = (ALLOW_KEY, DENY_KEY)


class InvalidTableFilterError(ValueError):
    """Raised when an allow/deny pattern is not a valid regular expression."""


def compile_pattern(raw: Any, *, field: str) -> "re.Pattern[str] | None":
    """Compile one pattern, or ``None`` when it is absent/blank.

    Raises :class:`InvalidTableFilterError` rather than returning ``None`` on a
    bad pattern. Silently ignoring it would ingest everything the operator was
    trying to exclude, which is the opposite of what they asked for and is
    invisible until a query fails much later.
    """
    if raw is None:
        return None
    pattern = str(raw).strip()
    if not pattern:
        return None
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise InvalidTableFilterError(f"{field} is not a valid regex: {exc}") from exc


@dataclass(frozen=True)
class TableFilter:
    """An allow/deny pair applied to unqualified relation names."""

    allow: "re.Pattern[str] | None" = None
    deny: "re.Pattern[str] | None" = None

    @property
    def active(self) -> bool:
        """Whether this filter would exclude anything at all."""
        return self.allow is not None or self.deny is not None

    def keep(self, table_name: str) -> bool:
        """Whether *table_name* survives the filter."""
        if not table_name:
            return True
        if self.deny is not None and self.deny.search(table_name):
            return False
        if self.allow is not None and not self.allow.search(table_name):
            return False
        return True

    def describe(self) -> str:
        """Short human-readable form, for log lines."""
        parts = []
        if self.allow is not None:
            parts.append(f"allow={self.allow.pattern!r}")
        if self.deny is not None:
            parts.append(f"deny={self.deny.pattern!r}")
        return ", ".join(parts) or "inactive"


def from_connection(connection: dict[str, Any] | None) -> TableFilter:
    """Build a :class:`TableFilter` from a connection dict.

    Returns an inactive filter when neither key is set, so callers can apply it
    unconditionally without branching.
    """
    if not connection:
        return TableFilter()
    return TableFilter(
        allow=compile_pattern(connection.get(ALLOW_KEY), field="Table allowlist"),
        deny=compile_pattern(connection.get(DENY_KEY), field="Table denylist"),
    )


def filter_frame(
    frame: "pd.DataFrame | None", table_filter: TableFilter
) -> "pd.DataFrame | None":
    """Drop rows whose ``table_name`` the filter excludes.

    A frame without a ``table_name`` column is returned untouched — ``queries``
    and other non-relation frames legitimately lack one.
    """
    if frame is None or not table_filter.active:
        return frame
    if getattr(frame, "empty", True):
        return frame
    if "table_name" not in frame.columns:
        return frame
    keep = frame["table_name"].map(
        lambda name: table_filter.keep("" if name is None else str(name))
    )
    return frame[keep]


def apply(
    tables: "pd.DataFrame | None",
    columns: "pd.DataFrame | None",
    views: "pd.DataFrame | None",
    table_filter: TableFilter,
) -> tuple["pd.DataFrame | None", "pd.DataFrame | None", "pd.DataFrame | None"]:
    """Filter the three relation frames together.

    All three must be filtered as one: ``create_dataframe`` validates that every
    listed relation has described columns, so dropping a table while keeping its
    columns (or the reverse) would trip that check.
    """
    if not table_filter.active:
        return tables, columns, views

    before = 0 if tables is None or tables.empty else len(tables)
    tables = filter_frame(tables, table_filter)
    columns = filter_frame(columns, table_filter)
    views = filter_frame(views, table_filter)
    after = 0 if tables is None or tables.empty else len(tables)

    if before != after:
        logger.info(
            "Table filter (%s) kept %d of %d table(s)",
            table_filter.describe(),
            after,
            before,
        )
    return tables, columns, views
