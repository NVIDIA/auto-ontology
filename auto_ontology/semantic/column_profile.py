# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whole-table column profiles for text-to-SQL, measured at ingest.

The profile the AskData BIRD submission builds (Shkapenyuk et al., "Automatic
Metadata Extraction for Text-to-SQL", 2025): for every column, its NULL,
empty-string and distinct counts over the whole table, its range and length,
the shape of its text values, and its most common values with their counts. A
model then reads each profile beside the column's name, its table and the
description it already has, and writes what the column holds and how its
values are written.

``sample_values`` comes from the first 1000 rows, which on a table stored in
key order is one corner of it. Everything here except the shape is an
aggregate over every row; the shape needs whole values, so it reads the same
bounded sample.
"""

from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage

from auto_ontology.dal.datasources import (
    fetch_table_context,
    fetch_tables_for_database,
    store_column_profiles,
)
from auto_ontology.semantic import visit_enter
from auto_ontology.semantic.models import ColumnDescriptionResult
from auto_ontology.utils.column_profile import profile_facts
from auto_ontology.utils.llm_invoke import (
    get_llm_client,
    invoke_with_structured_output,
)
from auto_ontology.utils.sql_identifiers import quoted_identifier

if TYPE_CHECKING:
    import pandas as pd

    from auto_ontology.connectors.base import SQLDatabase

logger = logging.getLogger(__name__)

#: Declared-type substrings whose values cannot be compared or counted
#: distinctly on every engine (Postgres has no equality on ``json``, no
#: ``MIN`` on ``boolean``).
_UNAGGREGATED_TYPES = (
    "json",
    "array",
    "[]",
    "bool",
    "blob",
    "bytea",
    "binary",
    "map",
    "struct",
    "geometry",
    "geography",
    "variant",
    "object",
)
#: Columns aggregated per statement; a wide table is split rather than sent as
#: one SELECT with hundreds of expressions.
_STATS_COLUMNS_PER_QUERY = 40
#: Most common values kept, with their counts.
_TOP_VALUES = 10
#: Values longer than this are not kept as a most common value.
_MAX_TOP_VALUE_LEN = 60
_SUMMARY_BATCH_SIZE = 15
_SUMMARY_WORKERS = 4

_NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")

_SUMMARY_SYSTEM = """\
You document the columns of one database table for someone about to write SQL \
against it. Each column comes with its type, the description it already has, if \
any, and a profile measured over the whole table.

For every column write one sentence saying what it holds and how its values are \
written: the format or pattern, what codes stand for when the values or the \
description make it clear, units, and anything that decides how a filter or a \
comparison has to be written, such as numbers stored as text, empty strings \
standing in for missing values, or a placeholder value.

The reader also sees the profile itself, so do not repeat its counts, ranges or \
percentages. Use only what the name, the description and the profile support: \
do not guess why values are NULL or what a value means when nothing shows it.

Add facts about the values, never new meaning. When the column has a \
description, keep its meaning in its own words: do not narrow, widen or reword \
it, and do not tie it to this table or another one ("the account's branch", \
"the client's home") unless the description itself says so. Do not describe \
keys or joins ("joinable to", "references", "foreign key", "identifier used to \
link"): the joins are given to the reader separately. When there is nothing \
about the values worth adding, return the description unchanged, or the bare \
type when there is none.

Return exactly one entry per column, using its physical name."""


def _aggregable(data_type: str | None) -> bool:
    lowered = (data_type or "").lower()
    return not any(token in lowered for token in _UNAGGREGATED_TYPES)


def _is_text(data_type: str | None) -> bool:
    return visit_enter._is_text_sample_type(data_type)


def _int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def value_shape(values: list[Any]) -> list[str]:
    """How a text column's values are written, as short phrases.

    Only what holds for every value is reported, since a filter has to match
    the stored form exactly: "digits only" means a literal must be quoted, and
    "numbers stored as text" means a comparison sorts as text unless cast.
    """
    texts = [v for v in values if isinstance(v, str)]
    if not texts:
        return []
    shape: list[str] = []
    filled = [t for t in texts if t]
    lettered = [t for t in filled if any(c.isalpha() for c in t)]
    if filled and all(t.isdigit() for t in filled):
        shape.append("digits only")
    elif filled and all(_NUMBER.fullmatch(t) for t in filled):
        shape.append("numbers stored as text")
    elif lettered and len(lettered) == len(filled):
        if all(t.isupper() for t in lettered):
            shape.append("upper case")
        elif all(t.islower() for t in lettered):
            shape.append("lower case")
    if any(t != t.strip() for t in texts):
        shape.append("some values have leading or trailing spaces")
    distinct = set(filled)
    if len(distinct) > 1:
        prefix = os.path.commonprefix(sorted(distinct))
        if len(prefix) >= 2:
            shape.append(f"every value starts with {prefix!r}")
    return shape


def _stats_sql(
    qualified: str, columns: list[dict[str, Any]], dialect: str | None
) -> str:
    parts = ["COUNT(*) AS n"]
    for i, column in enumerate(columns):
        q = quoted_identifier(column["name"], dialect)
        parts += [
            f"COUNT({q}) AS nn{i}",
            f"COUNT(DISTINCT {q}) AS d{i}",
            f"MIN({q}) AS lo{i}",
            f"MAX({q}) AS hi{i}",
        ]
        if _is_text(column.get("data_type")):
            parts += [
                f"MIN(LENGTH({q})) AS ln{i}",
                f"MAX(LENGTH({q})) AS lx{i}",
                f"SUM(CASE WHEN {q} = '' THEN 1 ELSE 0 END) AS e{i}",
            ]
    return f"SELECT {', '.join(parts)} FROM {qualified}"


def _read_stats(row: Any, columns: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows = _int(row["n"])
    out: dict[str, dict[str, Any]] = {}
    for i, column in enumerate(columns):
        non_null = _int(row[f"nn{i}"])
        profile: dict[str, Any] = {
            "row_count": rows,
            "null_count": None if rows is None or non_null is None else rows - non_null,
            "distinct_count": _int(row[f"d{i}"]),
            "min": visit_enter._json_ready_sample(row[f"lo{i}"]),
            "max": visit_enter._json_ready_sample(row[f"hi{i}"]),
        }
        if f"ln{i}" in row:
            profile["min_length"] = _int(row[f"ln{i}"])
            profile["max_length"] = _int(row[f"lx{i}"])
            profile["empty_count"] = _int(row[f"e{i}"])
        out[column["name"]] = profile
    return out


def _run_stats(
    connector: "SQLDatabase", qualified: str, columns: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Aggregate *columns*, falling back to one statement per column on error.

    One column whose type the engine refuses to aggregate must not cost the
    rest of the table its profile.
    """
    dialect = getattr(connector, "dialect", None)
    try:
        df = visit_enter._execute_capped(
            connector, _stats_sql(qualified, columns, dialect)
        )
        return _read_stats(df.iloc[0], columns)
    except Exception as exc:
        visit_enter._sampling_breaker.record_failure(connector, exc)
        if len(columns) == 1:
            logger.info(
                "profile of %s.%s skipped (%s)",
                qualified,
                columns[0]["name"],
                type(exc).__name__,
            )
            return {}
    out: dict[str, dict[str, Any]] = {}
    for column in columns:
        out.update(_run_stats(connector, qualified, [column]))
    return out


def _top_values(connector: "SQLDatabase", qualified: str, name: str) -> list[list[Any]]:
    q = quoted_identifier(name, getattr(connector, "dialect", None))
    try:
        df = visit_enter._execute_capped(
            connector,
            f"SELECT {q} AS v, COUNT(*) AS c FROM {qualified} WHERE {q} IS NOT NULL "
            f"GROUP BY {q} ORDER BY COUNT(*) DESC, {q} LIMIT {_TOP_VALUES}",
        )
    except Exception as exc:
        visit_enter._sampling_breaker.record_failure(connector, exc)
        return []
    out = []
    for value, count in zip(df.iloc[:, 0].tolist(), df.iloc[:, 1].tolist()):
        value = visit_enter._json_ready_sample(value)
        if len(str(value)) <= _MAX_TOP_VALUE_LEN:
            out.append([value, int(count)])
    return out


def _shape_sample(connector: "SQLDatabase", qualified: str) -> "pd.DataFrame | None":
    try:
        return visit_enter._execute_capped(
            connector,
            f"SELECT * FROM {qualified} LIMIT {visit_enter._PROFILING_SAMPLE_LIMIT}",
        )
    except Exception as exc:
        visit_enter._sampling_breaker.record_failure(connector, exc)
        return None


def measure_column_profiles(
    connector: "SQLDatabase",
    table: dict[str, Any],
    columns: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """``{column_name: profile}`` for *table*, measured on the live database."""
    qualified = connector.qualify(table.get("schema_name"), table["name"])
    measurable = [
        c for c in columns if c.get("name") and _aggregable(c.get("data_type"))
    ]
    stats: dict[str, dict[str, Any]] = {}
    for start in range(0, len(measurable), _STATS_COLUMNS_PER_QUERY):
        chunk = measurable[start : start + _STATS_COLUMNS_PER_QUERY]
        stats.update(_run_stats(connector, qualified, chunk))

    sample = _shape_sample(connector, qualified)
    profiles: dict[str, dict[str, Any]] = {}
    for column in measurable:
        name = column["name"]
        profile = stats.get(name)
        if profile is None:
            continue
        rows, nulls = profile.get("row_count"), profile.get("null_count")
        non_null = rows - nulls if rows is not None and nulls is not None else None
        if non_null and profile.get("distinct_count") != non_null:
            profile["top_values"] = _top_values(connector, qualified, name)
        if (
            _is_text(column.get("data_type"))
            and sample is not None
            and name in sample.columns
        ):
            profile["shape"] = value_shape(sample[name].dropna().tolist())
        profiles[name] = profile
    return profiles


def _resolve_names(returned: str, requested: list[str]) -> str | None:
    """The requested column *returned* refers to.

    The model sometimes echoes the ``name (type)`` annotation back as the
    name, so a returned string that only starts with a requested name is taken
    as the longest such name.
    """
    if returned in requested:
        return returned
    prefixes = [n for n in requested if n and returned.startswith(n)]
    return max(prefixes, key=len) if prefixes else None


def _summarize_batch(
    table: dict[str, Any],
    all_names: list[str],
    batch: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
) -> dict[str, str]:
    lines = [f"Table: {table['name']}"]
    if table.get("description"):
        lines.append(f"Table description: {table['description']}")
    lines.append(f"Columns in this table: {', '.join(all_names)}")
    lines.append("")
    lines.append("Columns to document:")
    for column in batch:
        name = column["name"]
        lines.append(f"- {name} ({column.get('data_type') or 'unknown'})")
        lines.append(f"  Description: {column.get('description') or '(none)'}")
        lines.append(f"  Profile: {profile_facts(profiles[name], top_values=True)}")
    try:
        result = invoke_with_structured_output(
            get_llm_client(temperature=0.0),
            [
                SystemMessage(content=_SUMMARY_SYSTEM),
                HumanMessage(content="\n".join(lines)),
            ],
            ColumnDescriptionResult,
        )
    except Exception:
        logger.warning(
            "[%s] profile summary batch failed — continuing without it",
            table["name"],
            exc_info=True,
        )
        return {}
    if result is None:
        return {}
    requested = [c["name"] for c in batch]
    out: dict[str, str] = {}
    for entry in result.descriptions:
        name = _resolve_names(entry.column_name or "", requested)
        text = (entry.description or "").strip()
        if name and text:
            out.setdefault(name, text)
    return out


def summarize_column_profiles(
    table: dict[str, Any],
    columns: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """``{column_name: summary}``, written by a model from each profile."""
    profiled = [c for c in columns if c.get("name") in profiles]
    all_names = [c["name"] for c in columns if c.get("name")]
    batches = [
        profiled[i : i + _SUMMARY_BATCH_SIZE]
        for i in range(0, len(profiled), _SUMMARY_BATCH_SIZE)
    ]
    summaries: dict[str, str] = {}
    for batch in batches:
        summaries.update(_summarize_batch(table, all_names, batch, profiles))
    return summaries


def profile_table(
    connector: "SQLDatabase",
    table: dict[str, Any],
    columns: list[dict[str, Any]],
    *,
    summarize: bool = True,
) -> dict[str, dict[str, Any]]:
    """Measure, summarise and store the profiles of one table's columns."""
    profiles = measure_column_profiles(connector, table, columns)
    if summarize and profiles:
        for name, summary in summarize_column_profiles(
            table, columns, profiles
        ).items():
            profiles[name]["summary"] = summary
    store_column_profiles(table["id"], profiles)
    return profiles


def profile_database(
    database_name: str,
    *,
    summarize: bool = True,
    max_workers: int = _SUMMARY_WORKERS,
) -> int:
    """Profile every catalog table of an already-ingested database.

    For a database ingested before profiles existed: compilation skips tables
    that already have a Term, so it would never reach them. Returns the number
    of columns profiled.
    """
    connector = visit_enter._resolve_connector(database_name)
    if connector is None:
        raise ValueError(f"no live connector for database {database_name!r}")
    tables = fetch_tables_for_database(database_name)

    def one(table: dict[str, Any]) -> int:
        columns = fetch_table_context(table["id"])["columns"]
        try:
            profiles = profile_table(connector, table, columns, summarize=summarize)
        except Exception:
            logger.warning("[%s] profiling failed", table["name"], exc_info=True)
            return 0
        logger.info("[%s] profiled %d columns", table["name"], len(profiles))
        return len(profiles)

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        return sum(pool.map(one, tables))


def reembed_column_attributes(database_name: str) -> int:
    """Re-embed every ColumnAttribute of *database_name* from its stored rows.

    The embedding text carries the profile summary, so a database profiled
    after it was compiled keeps its old vectors until this runs. Each table's
    rows are embedded before its old ones are deleted, so a failed embed
    leaves the previous vectors in place. Returns the rows written.
    """
    from auto_ontology.dal.terms import fetch_terms_and_attributes_for_table
    from auto_ontology.semantic.embed import build_semantic_embedder, _build_rows
    from auto_ontology.utils.embedding import batch_embed
    from auto_ontology.vdb.records import write_embedded_rows

    embedder = build_semantic_embedder(database_name, reset=False)
    if embedder is None:
        raise ValueError("no embedding endpoint configured")
    written = 0
    for table in fetch_tables_for_database(database_name):
        _terms, attrs = fetch_terms_and_attributes_for_table(str(table["id"]))
        rows = _build_rows(database_name, {}, attrs)
        if not rows:
            continue
        embedded = batch_embed(rows, embedder.embed_params)
        records = [
            row
            for row in ([] if embedded is None else embedded.to_dict(orient="records"))
            if (row.get("metadata") or {}).get("embedding")
        ]
        if len(records) < len(rows):
            logger.warning(
                "[%s] embedded %d/%d attributes; keeping the old vectors",
                table["name"],
                len(records),
                len(rows),
            )
            continue
        for attr in attrs:
            if attr.get("id"):
                embedder.vdb.delete_by_id(str(attr["id"]))
        written += write_embedded_rows(embedder.vdb, records)
    return written
