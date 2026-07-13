"""Deterministic column mapping — exclude FK columns."""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from gsf.semantic.models import ColumnAttributeSpec, ColumnDescriptionResult
from gsf.utils.llm_invoke import get_llm_client, invoke_with_structured_output

logger = logging.getLogger(__name__)

_DESCRIPTION_SYSTEM = """\
You are a data analyst documenting the columns of a relational table for a \
semantic layer. For every column you are given, write ONE concise sentence \
describing what the column represents in business terms.

Rules:
- Use the column name, data type, and sample values as evidence.
- Keep each description to a single, factual sentence — no speculation.
- Return exactly one entry per column provided, using the physical column name."""


def to_term_name(table_name: str) -> str:
    """CamelCase provisional Term from snake_case table name."""
    parts = re.split(r"[_\s]+", table_name.strip())
    return "".join(p.capitalize() for p in parts if p)


_DATE_TYPE_TOKENS = ("date", "time", "timestamp", "datetime")


def fk_source_columns(fks: list[dict[str, Any]]) -> set[str]:
    return {fk["source_column"] for fk in fks if fk.get("source_column")}


def _is_date_type(data_type: str | None) -> bool:
    """Whether a declared column type is a date/time type."""
    if not data_type:
        return False
    lowered = data_type.lower()
    return any(token in lowered for token in _DATE_TYPE_TOKENS)


def _generate_column_descriptions(
    columns: list[dict[str, Any]],
    columns_profiling_samples: dict[str, dict[str, Any]] | None,
) -> dict[str, str]:
    """Ask the LLM (single call) for a business description of each column.

    Returns a ``{column_name: description}`` map; empty when the call fails or
    no columns are provided.
    """
    if not columns:
        return {}

    profiling = columns_profiling_samples or {}
    lines = []
    for col in columns:
        name = col.get("name", "")
        dtype = col.get("data_type") or "unknown"
        # Date/time columns: don't feed sample values into the description —
        # concrete dates add no business meaning.
        samples = (profiling.get(name) or {}).get("sample_values") or []
        if _is_date_type(col.get("data_type")):
            samples = []
        sample_str = (
            f" — samples: {', '.join(str(s) for s in samples)}" if samples else ""
        )
        lines.append(f"  - {name} ({dtype}){sample_str}")

    prompt = "Columns:\n" + "\n".join(lines)

    try:
        result = invoke_with_structured_output(
            get_llm_client(temperature=0.0),
            [
                SystemMessage(content=_DESCRIPTION_SYSTEM),
                HumanMessage(content=prompt),
            ],
            ColumnDescriptionResult,
        )
    except Exception:
        logger.warning(
            "column description LLM call errored — proceeding without them",
            exc_info=True,
        )
        return {}
    if result is None:
        logger.warning("column description LLM call failed — proceeding without them")
        return {}

    return {
        d.column_name: d.description.strip()
        for d in result.descriptions
        if d.column_name and d.description.strip()
    }


def column_attribute_specs(
    columns: list[dict[str, Any]],
    fks: list[dict[str, Any]],
    suggested_fk_columns: set[str] | None = None,
    columns_profiling_samples: dict[str, dict[str, Any]] | None = None,
) -> list[ColumnAttributeSpec]:
    """Non-FK columns mapped 1:1 to ColumnAttribute candidates."""
    fk_cols = fk_source_columns(fks)
    if suggested_fk_columns:
        fk_cols |= suggested_fk_columns

    candidates = [
        col for col in columns if (name := col.get("name", "")) and name not in fk_cols
    ]

    llm_descriptions = _generate_column_descriptions(
        candidates, columns_profiling_samples
    )

    specs: list[ColumnAttributeSpec] = []
    for col in candidates:
        name = col["name"]
        attr_name = _column_to_attr_name(name)
        specs.append(
            ColumnAttributeSpec(
                source_column=name,
                name=attr_name,
                datatype=str(col.get("data_type") or ""),
                description=col.get("description")
                or llm_descriptions.get(name)
                or None,
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
