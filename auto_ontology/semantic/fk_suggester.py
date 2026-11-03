# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM inference for columns that look like foreign keys."""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
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

# A table's own key is named for that table. The same name on any other table
# points at it. Terms do not exist yet when this runs.
_OWN_KEY_SUFFIXES = ("_key", "_id", "_code")

_SYSTEM = """\
You review relational table metadata and produce three outputs:

1. fk_suggestions — columns that are likely foreign keys but are not already declared as
   FOREIGN_KEY or primary-key columns.

   A likely foreign key typically:
   - ends with _id, _key, or _code and has a meaningful prefix naming a different entity
     (e.g. customer_id, FCLT_ORGANIZATION_KEY) — NOT a bare "id"/"key" column, and NOT a
     column named exactly <this_table>_id, <this_table>_key, or <this_table>_code, unless
     it is marked is_unique: false
   - is another table's own key: the same name as a column {that_table}_id, {that_table}_key,
     or {that_table}_code on a different table. This is a foreign key even when is_unique
     is true. Uniqueness means one row here per referenced key, which is normal on a
     bridge; it does not make the column this table's primary key
   - is the other column of a two-column table whose first column is already a foreign
     key. Both columns reference an entity. The exception is a column that is this
     table's own key
   - has a description that contains words like "references", "identifier of", or names another table
   - has an integer or string type consistent with identifiers
   - semantically points to a row in another table
   - is UUID-typed and is not the table's own primary key

   Omit columns that are measures, physical measurements/dimensions (e.g.
   weight, height, length, width, size, volume), name/label columns (e.g.
   name, title, label, description), timestamps, free text, flags, or
   otherwise unlikely to reference another table. Return an empty list when
   none qualify.

2. pk_column_names — columns that identify this table's own records even if not declared.
   A bare "id" or "uuid", or a column named exactly <this_table>_id, <this_table>_key, or
   <this_table>_code. A column named *_key, *_id, or *_code can also be this table's own
   key when it identifies this table's rows: the primary key itself uses that suffix, as
   DLC_KEY does on the department table. Never list another table's own key, even when
   is_unique is true, and never list a column marked is_unique: false. Usually empty or
   one entry. Only list columns from the candidate list.

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


def references_other_table_key(
    column_name: str,
    table_name: str,
    own_key_column_names: Collection[str],
) -> bool:
    """Whether *column_name* is another table's own key, not this table's.

    An own key is a column named ``{table}_key``, ``{table}_id``, or
    ``{table}_code``. ``own_key_column_names`` lists those columns from the
    catalog. Uniqueness is irrelevant: a bridge often has one row per
    referenced key, and that still references the other table.
    """
    column = column_name.casefold()
    if column not in {name.casefold() for name in own_key_column_names}:
        return False
    table = table_name.casefold()
    return not any(
        column.endswith(suffix) and column[: -len(suffix)] == table
        for suffix in _OWN_KEY_SUFFIXES
    )


def named_table_key_links(
    table_name: str,
    columns: list[dict[str, Any]],
    own_key_column_ids: Mapping[str, str],
    declared_fk_columns: Collection[str],
    declared_pk_columns: Collection[str] = (),
) -> list[tuple[str, str]]:
    """Column-id pairs for keys that name another table.

    The target id comes from the catalog column of the table the name belongs
    to. Declared keys are left as the warehouse recorded them. A column with
    no id, or one that is itself that key, is skipped.
    """
    folded = {
        name.casefold(): column_id for name, column_id in own_key_column_ids.items()
    }
    declared = {name.casefold() for name in declared_fk_columns}
    declared_pk = {name.casefold() for name in declared_pk_columns}
    links: list[tuple[str, str]] = []
    for column in columns:
        name = str(column.get("name") or "")
        source_id = column.get("id")
        if (
            not name
            or not source_id
            or name.casefold() in declared
            or name.casefold() in declared_pk
        ):
            continue
        if not references_other_table_key(name, table_name, folded):
            continue
        target_id = folded.get(name.casefold())
        if not target_id or target_id == source_id:
            continue
        links.append((str(source_id), target_id))
    return links


def _is_this_tables_own_key(column_name: str, table_name: str) -> bool:
    """Whether *column_name* is ``{table}_key``, ``_id``, or ``_code``."""
    column = column_name.casefold()
    table = table_name.casefold()
    return any(
        column.endswith(suffix) and column[: -len(suffix)] == table
        for suffix in _OWN_KEY_SUFFIXES
    )


def _proven_fk_names(
    candidates: list[dict[str, Any]],
    table_name: str,
    declared_fk_names: Collection[str],
    own_key_column_names: Collection[str],
) -> set[str]:
    """Foreign keys known before the model's opinion of the other columns.

    Declared foreign keys, plus columns named ``{other_table}_key``. A model
    guess is not enough to decide the remaining column of a two-column table.
    """
    proven = {name for name in declared_fk_names if name}
    for col in candidates:
        name = str(col.get("name") or "")
        if references_other_table_key(name, table_name, own_key_column_names):
            proven.add(name)
    return proven


def _paired_fk_suggestions(
    columns: list[dict[str, Any]],
    table_name: str,
    proven_fk_names: Collection[str],
    declared_pk_names: Collection[str],
    seen: set[str],
) -> list[PotentialFkSuggestion]:
    """The remaining column of a two-column table that already has one FK.

    A bridge stores one association per row, so the second column references
    an entity too. This uses foreign keys already known from the warehouse or
    from a ``{table}_key`` name, not a guess about the other column. The
    table's own key is left as its identifier.
    """
    if len(columns) != 2:
        return []
    proven = {name.casefold() for name in proven_fk_names if name}
    if not proven:
        return []
    pk_names = {name.casefold() for name in declared_pk_names}
    suggestions: list[PotentialFkSuggestion] = []
    for col in columns:
        name = str(col.get("name") or "")
        if (
            not name
            or name in seen
            or name.casefold() in proven
            or name.casefold() in pk_names
            or col.get("is_foreign_key_target")
            or _is_this_tables_own_key(name, table_name)
        ):
            continue
        seen.add(name)
        suggestions.append(
            PotentialFkSuggestion(
                column_name=name,
                rationale=(
                    "two-column table whose other column is already a foreign key"
                ),
            )
        )
    return suggestions


def _other_table_key_suggestions(
    candidates: list[dict[str, Any]],
    table_name: str,
    own_key_column_names: Collection[str],
    seen: set[str],
) -> list[PotentialFkSuggestion]:
    """FK suggestions the catalog name proves, whatever the model answered.

    ``seen`` holds names already suggested for this table. It is not a record
    of tables already visited: the other table's key id comes from the catalog
    and is known before either table is compiled.
    """
    suggestions: list[PotentialFkSuggestion] = []
    for col in candidates:
        name = str(col.get("name") or "")
        if not name or name in seen:
            continue
        if not references_other_table_key(name, table_name, own_key_column_names):
            continue
        seen.add(name)
        suggestions.append(
            PotentialFkSuggestion(
                column_name=name,
                rationale=(
                    "column name is another table's key; uniqueness does not "
                    "make it this table's primary key"
                ),
            )
        )
    return suggestions


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
    own_key_column_names: Collection[str] | None = None,
) -> PotentialFkResult:
    """Ask the LLM which non-PK, non-declared-FK columns may be foreign keys.

    *columns_profiling_samples* maps each column to its profiling data
    (``{"sample_values": [...], "is_unique": bool}``) when a live sample was
    available. Columns referenced by declared foreign keys are not candidates:
    they identify rows in this table rather than referencing another table.

    *own_key_column_names* are catalog columns named ``{table}_key``,
    ``{table}_id``, or ``{table}_code``. A candidate with one of those names,
    on a different table, is a foreign key even when the model calls it this
    table's primary key. The names come from the catalog; terms are not
    available yet.

    A table with exactly two columns and one foreign key already known keeps
    the other column as a foreign key too. The known key is a declared foreign
    key or another table's ``{table}_key`` name, not a guess. This table's own
    key stays its identifier.
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

    table_name = str(table.get("name") or "")
    own_keys = own_key_column_names or ()
    other_table_keys = [
        col["name"]
        for col in candidates
        if references_other_table_key(str(col.get("name") or ""), table_name, own_keys)
    ]
    other_table_key_block = (
        (
            "Columns that name another table's key, so they are foreign keys "
            "even when is_unique is true: " + ", ".join(other_table_keys) + "\n"
        )
        if other_table_keys
        else ""
    )
    already_foreign_keys = sorted(known_fk_names | set(other_table_keys))
    pair_block = (
        (
            "This table has two columns and "
            + ", ".join(already_foreign_keys)
            + " is already a foreign key, so the other column is a foreign "
            "key unless it is this table's own key.\n"
        )
        if len(columns) == 2 and already_foreign_keys
        else ""
    )
    prompt = (
        f"Table: {table_header}\n"
        f"Description: {table.get('description') or ''}\n"
        f"Primary key columns (exclude from suggestions): {pk_block}\n"
        f"Known foreign key columns (exclude from suggestions): {known_fk_block}\n"
        f"Known foreign key roles:\n{known_fk_lines}\n"
        f"All table columns:\n{all_column_lines}\n"
        f"Candidate columns:\n{candidate_lines}\n"
        f"{other_table_key_block}"
        f"{pair_block}"
    )

    result = invoke_with_structured_output(
        get_llm_client(temperature=0.0),
        [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)],
        FkAndPkResult,
    )
    if result is None:
        seen: set[str] = set()
        suggestions = _other_table_key_suggestions(
            candidates, table_name, own_keys, seen
        )
        suggestions.extend(
            _paired_fk_suggestions(
                columns,
                table_name,
                _proven_fk_names(candidates, table_name, known_fk_names, own_keys),
                pk_names,
                seen,
            )
        )
        return PotentialFkResult(suggestions=suggestions)

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
        # A column the model names as this table's key keeps its attribute.
        # Another table's own key stays a foreign key; that case is added below.
        if (
            not name
            or name in seen
            or name not in allowed
            or (
                name in llm_pk_names
                and not references_other_table_key(name, table_name, own_keys)
            )
        ):
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
    filtered.extend(
        _other_table_key_suggestions(candidates, table_name, own_keys, seen)
    )
    # Only a foreign key the warehouse declared, or one named for another
    # table, makes the second column a foreign key. A model guess about one
    # column must not pull the other column in with it.
    filtered.extend(
        _paired_fk_suggestions(
            columns,
            table_name,
            _proven_fk_names(candidates, table_name, known_fk_names, own_keys),
            pk_names,
            seen,
        )
    )

    return PotentialFkResult(
        suggestions=filtered,
        is_junction_table=result.is_junction_table,
        junction_table_rationale=result.junction_table_rationale.strip(),
    )
