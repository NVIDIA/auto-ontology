# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM inference for columns that look like foreign keys."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from auto_ontology.semantic.deterministic import fk_source_columns
from auto_ontology.utils.llm_invoke import (
    get_llm_client,
    invoke_with_structured_output,
)
from auto_ontology.utils.sample_values import stringify_sample_values
from auto_ontology.semantic.models import (
    FkAndPkResult,
    PotentialFkResult,
    PotentialFkSuggestion,
)

logger = logging.getLogger(__name__)

_SYSTEM = """\
You review relational table metadata and produce three outputs:

1. fk_suggestions — columns that are likely foreign keys but are not already declared as
   FOREIGN_KEY or primary-key columns.

   A likely foreign key typically:
   - ends with _id or Id and has a meaningful prefix naming a different entity (e.g. customer_id,
     orderId) — NOT a bare "id"/"_id"/"Id" column or one whose prefix matches the table name,
     unless it is marked is_unique: false
   - has a description that contains words like "references", "identifier of", or names another table
   - has an integer or string type consistent with identifiers
   - semantically points to a row in another table
   - is UUID-typed and is not the table's own primary key

   Omit columns that are measures, physical measurements/dimensions (e.g.
   weight, height, length, width, size, volume), name/label columns (e.g.
   name, title, label, description), timestamps, free text, flags, or
   otherwise unlikely to reference another table. Return an empty list when
   none qualify.

2. pk_column_names — columns that appear to be the table's own primary key even if not
   explicitly declared as such. These are typically a bare "id", "uuid", or "<table_name>_id"
   column of integer or UUID type whose description or name conveys it identifies the table's
   own records. Usually empty or one entry. Only list columns from the candidate list.

3. is_junction_table and junction_table_rationale — classify the table's row grain.
   Set is_junction_table true ONLY when each row represents an association between at least
   two entity roles. At least two columns must be foreign-key roles; two distinct roles may
   point to the same entity. Extra columns may be a surrogate key or relationship properties
   such as role, quantity, ordering, status, or effective dates, but they must describe the
   association rather than establish an independent entity grain.

   Set it false for:
   - an entity table that merely contains ordinary foreign keys
   - a satellite, extension, or detail table with one parent FK plus descriptive attributes
   - lookup/reference tables
   - self-hierarchies with only one relationship role (for example id + parent_id)
   - event or transaction tables whose rows have their own business identity or lifecycle

   Be conservative. Multiple foreign keys alone do not make a table a junction table.

The is_unique marker comes from sampled column profiling. A column marked is_unique: false
contains repeated values, so it cannot be the table's own primary key. Never include it in
pk_column_names. If it is identifier-like, including a bare "id", evaluate it as a possible
foreign key using its name, type, description, and samples. Non-uniqueness alone does not prove
that a non-identifier attribute is a foreign key."""


def _pk_column_names(table: dict[str, Any]) -> set[str]:
    pk = table.get("pk") or []
    if isinstance(pk, str):
        return {pk} if pk else set()
    return {str(name) for name in pk if name}


def _candidate_columns(
    columns: list[dict[str, Any]],
    *,
    excluded: set[str],
) -> list[dict[str, Any]]:
    return [
        col
        for col in columns
        if (name := col.get("name"))
        and name not in excluded
        and not col.get("is_foreign_key_target")
    ]


def _format_sample_values(raw: Any) -> str:
    """Return a 'samples: ...' string filtered to ≤30-char values, or empty."""
    values = stringify_sample_values(raw, max_len=30)
    if not values:
        return ""
    return "samples: " + ", ".join(values)


def _format_column_line(col: dict[str, Any], is_unique: bool | None = None) -> str:
    desc = col.get("description") or ""
    suffix = f" — {desc}" if desc else ""
    sample_str = _format_sample_values(col.get("sample_values"))
    if sample_str:
        suffix += f" [{sample_str}]"
    if is_unique is not None:
        suffix += f" [is_unique: {str(is_unique).lower()}]"
    return f"  - {col['name']} ({col.get('data_type', '')}){suffix}"


def suggest_potential_foreign_keys(
    table: dict[str, Any],
    ctx: dict[str, Any],
    columns_profiling_samples: dict[str, dict[str, Any]] | None = None,
) -> PotentialFkResult:
    """Ask the LLM which non-PK, non-declared-FK columns may be foreign keys.

    *columns_profiling_samples* maps each column to its profiling data
    (``{"sample_values": [...], "is_unique": bool}``) when a live sample was
    available. Columns referenced by declared foreign keys are not candidates:
    they identify rows in this table rather than referencing another table.
    """
    columns = ctx.get("columns", [])
    fks = ctx.get("fks", [])
    pk_names = _pk_column_names(table)
    known_fk_names = fk_source_columns(fks)
    excluded = pk_names | known_fk_names
    candidates = _candidate_columns(columns, excluded=excluded)

    # A payload-bearing junction can have no FK-suggestion candidates because
    # all of its key columns are already declared. Tables wider than two
    # columns still need the table-grain decision from this same LLM call.
    if not candidates and len(columns) <= 2:
        return PotentialFkResult()

    schema_name = table.get("schema_name") or ""
    table_header = f"{schema_name}.{table['name']}" if schema_name else table["name"]
    known_fk_block = ", ".join(sorted(known_fk_names)) if known_fk_names else "(none)"
    pk_block = ", ".join(sorted(pk_names)) if pk_names else "(none)"
    profiling = columns_profiling_samples or {}
    all_column_lines = "\n".join(
        _format_column_line(col, (profiling.get(col["name"]) or {}).get("is_unique"))
        for col in columns
        if col.get("name")
    )
    candidate_lines = (
        "\n".join(
            _format_column_line(
                col, (profiling.get(col["name"]) or {}).get("is_unique")
            )
            for col in candidates
        )
        or "  (none)"
    )
    known_fk_lines = (
        "\n".join(
            (
                f"  - {fk.get('source_column')} -> "
                f"{fk.get('target_table') or fk.get('target_table_id') or '(unknown target)'}"
            )
            for fk in fks
            if fk.get("source_column")
        )
        or "  (none)"
    )

    prompt = (
        f"Table: {table_header}\n"
        f"Description: {table.get('description') or ''}\n"
        f"Primary key columns (exclude from suggestions): {pk_block}\n"
        f"Known foreign key columns (exclude from suggestions): {known_fk_block}\n"
        f"Known foreign key roles:\n{known_fk_lines}\n"
        f"All table columns:\n{all_column_lines}\n"
        f"Candidate columns:\n{candidate_lines}\n"
    )

    result = invoke_with_structured_output(
        get_llm_client(temperature=0.0),
        [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)],
        FkAndPkResult,
    )
    if result is None:
        return PotentialFkResult()

    llm_pk_names = {
        name
        for raw_name in result.pk_column_names
        if (name := raw_name.strip())
        and (profiling.get(name) or {}).get("is_unique") is not False
    }
    allowed = {col["name"] for col in candidates}
    filtered: list[PotentialFkSuggestion] = []
    seen: set[str] = set()
    for item in result.fk_suggestions:
        name = item.column_name.strip()
        if not name or name in seen or name not in allowed:
            continue
        seen.add(name)
        filtered.append(
            PotentialFkSuggestion(column_name=name, rationale=item.rationale.strip())
        )
    for col in candidates:
        name = col.get("name", "")
        if name in seen:
            continue
        is_unique = (profiling.get(name) or {}).get("is_unique")
        if (
            (col.get("data_type") or "").lower() == "uuid"
            and name not in llm_pk_names
            and is_unique is False
        ):
            seen.add(name)
            filtered.append(
                PotentialFkSuggestion(
                    column_name=name,
                    rationale=(
                        "uuid type, not a declared or inferred primary key — "
                        "almost certainly references another entity"
                    ),
                )
            )

    return PotentialFkResult(
        suggestions=filtered,
        is_junction_table=result.is_junction_table,
        junction_table_rationale=result.junction_table_rationale.strip(),
    )
