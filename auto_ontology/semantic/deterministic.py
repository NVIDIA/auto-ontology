# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic column mapping — exclude FK columns."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from auto_ontology.semantic.date_format import is_date_type
from auto_ontology.semantic.models import ColumnAttributeSpec, ColumnDescriptionResult
from auto_ontology.utils.llm_invoke import (
    get_llm_client,
    invoke_with_structured_output,
)

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

    Date/time columns are excluded — concrete dates add no business meaning
    (to an LLM description prompt, or as a stand-in when no description
    needed to be generated).
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
        data_type = col.get("data_type") or "unknown"
        samples = _get_column_samples(col, columns_profiling_samples)
        line = _add_samples_suffix(f"  - {name} ({data_type})", samples)
        lines.append(line)

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
    # contains spaces or parentheses, the model sometimes echoes the annotation back as
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

    # Only ask the LLM for columns that don't already have a description
    cols_with_description = [col for col in candidates if col.get("description")]
    cols_without_description = [col for col in candidates if not col.get("description")]
    llm_descriptions = _generate_column_descriptions(
        cols_without_description, columns_profiling_samples
    )

    profiling = columns_profiling_samples or {}

    def _build_column_attribute_spec(
        col: dict[str, Any], description: str | None
    ) -> ColumnAttributeSpec:
        name = col["name"]
        return ColumnAttributeSpec(
            source_column=name,
            name=_column_to_attr_name(name),
            datatype=str(col.get("data_type") or ""),
            description=description,
        )

    # Columns with an existing description skip the LLM entirely, so unlike
    # LLM descriptions (which were already generated with the samples as
    # evidence) their description never saw the sample values — bundle them
    # in now via the same " — samples: ..." suffix.
    specs = [
        _build_column_attribute_spec(
            col,
            _enrich_description(
                col,
                profiling,
                _add_samples_suffix(
                    col["description"], _get_column_samples(col, profiling)
                ),
            ),
        )
        for col in cols_with_description
    ]
    # Columns without a description: keep the LLM description as generated
    specs += [
        _build_column_attribute_spec(
            col,
            _enrich_description(col, profiling, llm_descriptions.get(col["name"])),
        )
        for col in cols_without_description
    ]
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
