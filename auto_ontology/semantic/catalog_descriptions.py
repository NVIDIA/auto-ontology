# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Describe catalog columns the source left undescribed, during the taxonomy pass.

A column whose catalog description is blank, or only repeats the column name,
gets one generated from the table, its sibling columns, its type, storage
format and profiled samples. The text is written to ``catalog_column`` before
FK detection and Term extraction, so both prompts and the ColumnAttribute
embedding read it. Source descriptions are never touched.

The ColumnAttribute descriptions are a separate, later pass —
:mod:`auto_ontology.semantic.description_enrichment` — that also sees the
tables each column joins to.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from auto_ontology.semantic.date_format import is_date_type
from auto_ontology.semantic.description_enrichment import resolve_column_names
from auto_ontology.semantic.models import ColumnDescriptionsResult
from auto_ontology.utils.llm_invoke import (
    get_llm_client,
    invoke_with_structured_output,
)
from auto_ontology.utils.sample_values import stringify_sample_values

logger = logging.getLogger(__name__)

# Wide tables overflow one structured-output response.
_BATCH_SIZE = 15
_MAX_SAMPLES = 5
_MAX_SAMPLE_LEN = 40
_MAX_SIBLING_COLUMNS = 60

_SYSTEM_PROMPT = """\
You document a relational database for a semantic layer that a text-to-SQL \
agent reads. Write the description of each column listed under "Columns to \
describe".

Rules:
- Use the table name, the other columns of the table, the data type, the \
format and the samples as evidence.
- Say what the column represents, and how its values are written when it \
matters for filtering (for example "7-digit code stored as text", "Y/N flag").
- One plain sentence per column, a factual statement about the data. No SQL, \
no speculation beyond the evidence, and never refer to the samples or this \
prompt.
- Return exactly one entry per column to describe, using its physical name."""


def _normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def needs_description(column: dict[str, Any]) -> bool:
    """Whether *column* has no description beyond its own name."""
    name = str(column.get("name") or "")
    description = str(column.get("description") or "").strip()
    return bool(name) and (
        not description or _normalized(description) == _normalized(name)
    )


def _samples(column: dict[str, Any], profiling: dict[str, dict[str, Any]]) -> list[str]:
    if is_date_type(column.get("data_type")):
        return []
    profile = profiling.get(str(column.get("name") or "")) or {}
    raw = profile.get("sample_values") or column.get("sample_values")
    values = stringify_sample_values(raw, max_len=_MAX_SAMPLE_LEN) or []
    return values[:_MAX_SAMPLES]


def _column_line(column: dict[str, Any], profiling: dict[str, dict[str, Any]]) -> str:
    line = f"- {column['name']} ({column.get('data_type') or 'unknown'})"
    profile = profiling.get(str(column["name"])) or {}
    notation = profile.get("format") or column.get("format")
    if notation:
        line += f" — format: {notation}"
    samples = _samples(column, profiling)
    if samples:
        line += f" — samples: {', '.join(samples)}"
    return line


def _prompt(
    table: dict[str, Any],
    columns: list[dict[str, Any]],
    batch: list[dict[str, Any]],
    profiling: dict[str, dict[str, Any]],
) -> str:
    lines = [f"Table: {table.get('name')}"]
    if table.get("description"):
        lines.append(f"Table description: {table['description']}")
    siblings = []
    for column in columns[:_MAX_SIBLING_COLUMNS]:
        sibling = f"- {column.get('name')} ({column.get('data_type') or 'unknown'})"
        if not needs_description(column):
            sibling += f": {column['description']}"
        siblings.append(sibling)
    lines += ["", "Columns of the table:", *siblings]
    lines += ["", "Columns to describe:", *(_column_line(c, profiling) for c in batch)]
    return "\n".join(lines)


def describe_catalog_columns(
    table: dict[str, Any],
    columns: list[dict[str, Any]],
    profiling: dict[str, dict[str, Any]] | None = None,
) -> dict[str, str]:
    """``{column_name: description}`` for the columns of *table* that need one.

    A batch that fails is logged and skipped; its columns stay undescribed.
    """
    profiling = profiling or {}
    targets = [c for c in columns if needs_description(c)]
    out: dict[str, str] = {}
    for start in range(0, len(targets), _BATCH_SIZE):
        batch = targets[start : start + _BATCH_SIZE]
        try:
            result = invoke_with_structured_output(
                get_llm_client(temperature=0.0),
                [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=_prompt(table, columns, batch, profiling)),
                ],
                ColumnDescriptionsResult,
            )
        except Exception:
            logger.warning(
                "[%s] catalog description batch errored — skipping it",
                table.get("name"),
                exc_info=True,
            )
            continue
        if result is None:
            logger.warning(
                "[%s] catalog description batch returned no result — skipping it",
                table.get("name"),
            )
            continue
        out.update(resolve_column_names(result, [str(c["name"]) for c in batch]))
    return out
