# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic column mapping — exclude FK columns."""

from __future__ import annotations

import re
from typing import Any

from auto_ontology.semantic.date_format import is_date_type
from auto_ontology.semantic.models import ColumnAttributeSpec


def to_term_name(table_name: str) -> str:
    """Return a human-readable fallback Term name from a physical table name."""
    separated = re.sub(
        r"(?<=[A-Z])(?=[A-Z][a-z])",
        " ",
        table_name.strip(),
    )
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", separated)
    parts = re.split(r"[^A-Za-z0-9]+", separated)
    return " ".join(
        part if part.isupper() else part.capitalize() for part in parts if part
    )


_FORMAT_MARKER = "format:"


def fk_source_columns(fks: list[dict[str, Any]]) -> set[str]:
    return {fk["source_column"] for fk in fks if fk.get("source_column")}


def _get_column_samples(
    col: dict[str, Any],
    columns_profiling_samples: dict[str, dict[str, Any]],
) -> list[str]:
    """Profiled sample values for a column.

    Date/time columns are excluded — concrete dates add no business meaning.
    """
    if is_date_type(col.get("data_type")):
        return []
    name = col.get("name", "")
    samples = (columns_profiling_samples.get(name) or {}).get("sample_values") or []
    return [str(s) for s in samples]


def _add_samples_suffix(text: str, samples: list[str]) -> str:
    """Append " — samples: v1, v2" to *text* when samples are present."""
    if not samples:
        return text
    return f"{text} — samples: {', '.join(samples)}"


def _date_format_clause(
    col: dict[str, Any],
    columns_profiling_samples: dict[str, dict[str, Any]],
) -> str | None:
    """How stored values are written, when a single notation fits them all.

    ``format`` is the column's storage notation — currently filled only for
    dates, but the same property would hold an id or address pattern later.
    """
    profile = columns_profiling_samples.get(col.get("name", "")) or {}
    notation = profile.get("format") or col.get("format")
    if not notation:
        return None
    return f"{_FORMAT_MARKER} {notation}"


def _add_date_format_clause(text: str, clause: str | None) -> str:
    """Append the notation, idempotently on its own marker."""
    if not clause or _FORMAT_MARKER in text:
        return text
    return f"{text} — {clause}" if text else clause


def _enrich_description(
    col: dict[str, Any],
    columns_profiling_samples: dict[str, dict[str, Any]],
    text: str | None,
) -> str | None:
    """Attach date notation (and, for existing descriptions, samples)."""
    enriched = _add_date_format_clause(
        text or "", _date_format_clause(col, columns_profiling_samples)
    )
    return enriched or None


def column_attribute_specs(
    columns: list[dict[str, Any]],
    fks: list[dict[str, Any]],
    suggested_fk_columns: set[str] | None = None,
    columns_profiling_samples: dict[str, dict[str, Any]] | None = None,
) -> list[ColumnAttributeSpec]:
    """Non-FK columns mapped 1:1 to ColumnAttribute candidates.

    Only a description the column already carries is used; columns without one
    are described after compilation by
    :mod:`auto_ontology.semantic.description_enrichment`, which can see the
    tables they join to.
    """
    fk_cols = fk_source_columns(fks)
    if suggested_fk_columns:
        fk_cols |= suggested_fk_columns

    profiling = columns_profiling_samples or {}
    specs: list[ColumnAttributeSpec] = []
    for col in columns:
        name = col.get("name", "")
        if not name or name in fk_cols:
            continue
        description = col.get("description")
        if description:
            description = _add_samples_suffix(
                description, _get_column_samples(col, profiling)
            )
        specs.append(
            ColumnAttributeSpec(
                source_column=name,
                name=_column_to_attr_name(name),
                datatype=str(col.get("data_type") or ""),
                description=_enrich_description(col, profiling, description),
            )
        )
    return specs


def fk_target_table_names(fks: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for fk in fks:
        tgt = fk.get("target_table")
        if tgt and tgt not in seen:
            seen.add(tgt)
            out.append(tgt)
    return out


def _column_to_attr_name(column_name: str) -> str:
    parts = re.split(r"[_\s]+", column_name.strip())
    if not parts:
        return column_name
    return parts[0].lower() + "".join(p.capitalize() for p in parts[1:] if p)
