"""Deterministic column mapping — exclude FK columns."""

from __future__ import annotations

import re
from typing import Any

from gsf.semantic.models import ColumnAttributeSpec


_GENERIC_SCHEMAS = {"public", "dbo", "main", "default"}


def to_term_name(table_name: str, schema_name: str = "") -> str:
    """User-friendly Term name from a table (and optional schema) name.

    The schema is prepended when it is non-generic and carries meaningful
    disambiguation (e.g. ``website.customers`` → ``"Website Customers"``
    vs ``public.orders`` → ``"Orders"``).
    """
    parts = re.split(r"[_\s]+", table_name.strip())
    name = " ".join(p.capitalize() for p in parts if p)
    if schema_name and schema_name.lower() not in _GENERIC_SCHEMAS:
        schema_parts = re.split(r"[_\s]+", schema_name.strip())
        schema_prefix = " ".join(p.capitalize() for p in schema_parts if p)
        return f"{schema_prefix} {name}"
    return name


def fk_source_columns(fks: list[dict[str, Any]]) -> set[str]:
    return {fk["source_column"] for fk in fks if fk.get("source_column")}


def column_attribute_specs(
    columns: list[dict[str, Any]],
    fks: list[dict[str, Any]],
    suggested_fk_columns: set[str] | None = None,
) -> list[ColumnAttributeSpec]:
    """Non-FK columns mapped 1:1 to ColumnAttribute candidates."""
    fk_cols = fk_source_columns(fks)
    if suggested_fk_columns:
        fk_cols |= suggested_fk_columns
    specs: list[ColumnAttributeSpec] = []
    for col in columns:
        name = col.get("name", "")
        if not name or name in fk_cols:
            continue
        attr_name = _column_to_attr_name(name)
        specs.append(
            ColumnAttributeSpec(
                source_column=name,
                name=attr_name,
                datatype=str(col.get("data_type") or ""),
                description=col.get("description") or None,
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
