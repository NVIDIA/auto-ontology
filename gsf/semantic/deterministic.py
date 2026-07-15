"""Deterministic column mapping — exclude FK columns."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from gsf.semantic.models import ColumnAttributeSpec, ColumnDescriptionResult
from gsf.utils.llm_invoke import get_llm_client, invoke_with_structured_output

logger = logging.getLogger(__name__)

# Wide tables make the model emit one JSON object per column in a single
# structured-output response; large batches generate too many tokens and time
# out. Bucket columns into small batches and describe them concurrently so one
# slow/failed batch never wipes out the whole table's descriptions.
_DESCRIPTION_BATCH_SIZE = 15
_DESCRIPTION_MAX_WORKERS = 1

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


def _describe_column_batch(
    columns: list[dict[str, Any]],
    columns_profiling_samples: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Ask the LLM for a business description of a single batch of columns.

    Returns a ``{column_name: description}`` map; empty when the call fails.
    """
    lines = []
    for col in columns:
        name = col.get("name", "")
        dtype = col.get("data_type") or "unknown"
        # Date/time columns: don't feed sample values into the description —
        # concrete dates add no business meaning.
        samples = (columns_profiling_samples.get(name) or {}).get("sample_values") or []
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
            "column description batch errored — proceeding without them",
            exc_info=True,
        )
        return {}
    if result is None:
        logger.warning("column description batch failed — proceeding without them")
        return {}

    # The prompt annotates each column as "<name> (<dtype>)". When a column name
    # # contains spaces or parentheses (e.g. BIRD's "Academic Year",
    # "Charter School (Y/N)"), the model sometimes echoes the annotation back as
    # the column_name (e.g. "Academic Year (TEXT)"). Resolve each returned name
    # to the requested physical name: exact match first, then the longest
    # requested name the returned string starts with.
    requested_names = [c.get("name", "") for c in columns if c.get("name")]
    requested_set = set(requested_names)
    out: dict[str, str] = {}
    for d in result.descriptions:
        raw = d.column_name or ""
        desc = (d.description or "").strip()
        if not desc:
            continue
        if raw in requested_set:
            out.setdefault(raw, desc)
            continue
        prefixes = [n for n in requested_names if n and raw.startswith(n)]
        if prefixes:
            out.setdefault(max(prefixes, key=len), desc)
    return out


def _generate_column_descriptions(
    columns: list[dict[str, Any]],
    columns_profiling_samples: dict[str, dict[str, Any]] | None,
) -> dict[str, str]:
    """Ask the LLM for a business description of each column.

    Columns are bucketed into small batches described concurrently, so wide
    tables don't overflow a single structured-output response (which times out).

    Returns a ``{column_name: description}`` map; empty when no columns are
    provided. Batches that fail are simply skipped.
    """
    if not columns:
        return {}

    profiling = columns_profiling_samples or {}
    batches = [
        columns[i : i + _DESCRIPTION_BATCH_SIZE]
        for i in range(0, len(columns), _DESCRIPTION_BATCH_SIZE)
    ]

    descriptions: dict[str, str] = {}
    if len(batches) == 1:
        return _describe_column_batch(batches[0], profiling)

    max_workers = min(len(batches), _DESCRIPTION_MAX_WORKERS)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_describe_column_batch, batch, profiling) for batch in batches
        ]
        for future in as_completed(futures):
            descriptions.update(future.result())

    return descriptions


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
