# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find semantic FKs from shared column values.

Name- and description-based resolution misses a reference whose column is not
named for its target (a drop-point column holding depot keys), and a target
that is not strictly unique (a depot key repeated for a depot split between two
zones). Both show up in the data: the column's values are the target's values.

Algorithm
---------
1. **Profile.** One bounded query per table counts non-null and distinct
   values over the first ``_PROFILE_ROWS`` rows of every joinable column.
2. **Key-like targets.** A column that owns an attribute and is unique, or
   nearly so (``_NEAR_UNIQUE_SHARE`` of its sampled values distinct), is a
   possible target. Its distinct values are loaded from a bounded sample.
3. **Candidates.** Every other joinable column without a semantic FK, and not
   key-like itself, is compared value by value with the targets on other
   tables. Values shorter than ``_MIN_VALUE_LENGTH`` are ignored: flags and
   small numbers coincide with every key. A target is a candidate when it
   holds at least ``_MIN_SHARED_VALUES`` of the column's sampled values and
   ``_MIN_SHARED_SHARE`` of them.
4. **Decision.** The LLM sees the column, its best ``_MAX_CANDIDATES``
   candidates and their overlap, and picks the referenced key or none.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from sqlglot import exp

from auto_ontology.connectors.base import SQLDatabase
from auto_ontology.dal.attributes import merge_semantic_fk
from auto_ontology.dal.datasources import fetch_columns_for_value_overlap
from auto_ontology.semantic.models import FkHitSelection
from auto_ontology.utils.llm_invoke import (
    get_llm_client,
    invoke_with_structured_output,
)
from auto_ontology.utils.sql_identifiers import quoted_identifier

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("SEMANTIC_FK_VALUE_OVERLAP", "true").strip().lower() not in {
    "0",
    "false",
    "no",
}
_NEAR_UNIQUE_SHARE = float(os.environ.get("SEMANTIC_FK_NEAR_UNIQUE_SHARE", "0.98"))
_MIN_KEY_ROWS = 20
_PROFILE_ROWS = 10_000
_TARGET_VALUE_ROWS = 50_000
_SOURCE_VALUE_ROWS = 10_000
_MAX_SOURCE_VALUES = 2_000
_MIN_VALUE_LENGTH = 3
_MIN_SHARED_VALUES = 5
_MIN_SHARED_SHARE = 0.1
_MAX_CANDIDATES = 5
_MAX_PROMPT_SAMPLES = 5
_WORKERS = 2

_SYSTEM_PROMPT = """\
You are a database schema expert. A column shares values with one or more \
key columns on other tables. Decide which candidate key, if any, the column \
references: the column you would join it to so that each of its rows reaches \
the row it refers to.

Rules:
- Return the exact column_id of the referenced candidate, or null.
- Shared values alone are not enough. The column and the candidate must \
identify the same kind of thing, such as a drop point and a depot key, or a \
customer number and a customer id.
- Return null for counts, amounts, years, ranks, flags, free text and other \
values that only coincide with a key.
- Prefer the candidate whose table is the one the column's values identify, \
over a table that merely repeats those values.
- Do NOT guess. If unsure, return null.
"""


@dataclass(frozen=True)
class _Column:
    id: str
    name: str
    data_type: str
    description: str
    sample_values: tuple[str, ...]
    attribute_id: str | None
    table_id: str
    table_name: str
    schema_name: str | None


@dataclass(frozen=True)
class _Candidate:
    target: _Column
    shared: int
    source_values: int


def is_joinable_type(data_type: str) -> bool:
    """Whether a column of *data_type* can hold a key: text or integer."""
    folded = (data_type or "").casefold()
    if any(token in folded for token in ("char", "text", "string", "uuid")):
        return True
    base = folded.split("(")[0].strip()
    return base in {
        "int",
        "integer",
        "tinyint",
        "smallint",
        "mediumint",
        "bigint",
        "int2",
        "int4",
        "int8",
    }


def is_key_like(non_null: int, distinct: int, is_unique: bool | None) -> bool:
    """Whether sampled counts make a column a key a reference can point at."""
    if is_unique:
        return non_null > 0
    return non_null >= _MIN_KEY_ROWS and distinct >= _NEAR_UNIQUE_SHARE * non_null


def overlap_accepted(source_values: int, shared: int) -> bool:
    """Whether *shared* of *source_values* distinct values is enough to propose a link."""
    if not source_values or not shared:
        return False
    return (
        shared >= min(_MIN_SHARED_VALUES, source_values)
        and shared >= _MIN_SHARED_SHARE * source_values
    )


def _informative(value: str) -> bool:
    return len(value) >= _MIN_VALUE_LENGTH


def _text(column: str, dialect: str | None) -> str:
    cast = exp.cast(exp.to_identifier(column, quoted=True), "TEXT")
    try:
        return cast.sql(dialect=dialect or None)
    except ValueError:
        return cast.sql()


def _sampled_values_sql(
    connector: SQLDatabase,
    schema: str | None,
    table: str,
    column: str,
    rows: int,
) -> str:
    """Distinct text values of *column* over its first *rows* non-null rows."""
    dialect = getattr(connector, "dialect", None)
    col = quoted_identifier(column, dialect)
    return (
        f"SELECT DISTINCT {_text(column, dialect)} AS v FROM "
        f"(SELECT {col} FROM {connector.qualify(schema, table)} "
        f"WHERE {col} IS NOT NULL LIMIT {rows}) AS sampled"
    )


def profile_columns(
    connector: SQLDatabase,
    schema: str | None,
    table: str,
    columns: list[str],
) -> dict[str, tuple[int, int]]:
    """``{column: (non_null, distinct)}`` over the table's first ``_PROFILE_ROWS`` rows."""
    if not columns:
        return {}
    dialect = getattr(connector, "dialect", None)
    quoted = [quoted_identifier(name, dialect) for name in columns]
    counts = ", ".join(
        f"COUNT({col}) AS nn_{i}, COUNT(DISTINCT {col}) AS d_{i}"
        for i, col in enumerate(quoted)
    )
    sql = (
        f"SELECT {counts} FROM (SELECT {', '.join(quoted)} FROM "
        f"{connector.qualify(schema, table)} LIMIT {_PROFILE_ROWS}) AS sampled"
    )
    row = connector.execute(sql).iloc[0].tolist()
    return {
        name: (int(row[2 * i] or 0), int(row[2 * i + 1] or 0))
        for i, name in enumerate(columns)
    }


def sampled_values(
    connector: SQLDatabase,
    schema: str | None,
    table: str,
    column: str,
    *,
    rows: int,
    limit: int | None = None,
) -> set[str]:
    """Distinct non-empty text values of *column* from a bounded sample."""
    sql = _sampled_values_sql(connector, schema, table, column, rows)
    if limit:
        sql += f" LIMIT {limit}"
    frame = connector.execute(sql)
    return {
        text
        for value in frame.iloc[:, 0].tolist()
        if value is not None and (text := str(value).strip())
    }


def shared_value_counts(
    connector: SQLDatabase,
    source: tuple[str | None, str, str],
    target: tuple[str | None, str, str],
) -> tuple[int, int]:
    """``(source distinct values, of those also in target)`` over bounded samples."""
    source_sql = _sampled_values_sql(connector, *source, _SOURCE_VALUE_ROWS)
    target_sql = _sampled_values_sql(connector, *target, _TARGET_VALUE_ROWS)
    sql = (
        f"SELECT COUNT(*) AS source_values, COUNT(t.v) AS shared FROM ({source_sql}) AS s "
        f"LEFT JOIN ({target_sql}) AS t ON t.v = s.v"
    )
    row = connector.execute(sql).iloc[0].tolist()
    return int(row[0] or 0), int(row[1] or 0)


def is_near_unique_column(
    connector: SQLDatabase,
    schema: str | None,
    table: str,
    column: str,
) -> bool:
    """Whether *column* is unique or nearly so over a bounded sample."""
    non_null, distinct = profile_columns(connector, schema, table, [column])[column]
    return is_key_like(non_null, distinct, None)


def discover_value_overlap_fks(
    database_name: str, connector: SQLDatabase | None
) -> int:
    """Create semantic FKs for columns whose values are another table's keys.

    Returns the number of edges written.
    """
    if not _ENABLED:
        logger.info("value-overlap FK discovery disabled (SEMANTIC_FK_VALUE_OVERLAP)")
        return 0
    if connector is None:
        logger.info(
            "value-overlap FK discovery skipped: no connector for %s", database_name
        )
        return 0

    targets, sources = _profile_database(database_name, connector)
    if not targets or not sources:
        logger.info("value-overlap FK discovery: no key-like targets or sources")
        return 0

    index: dict[str, list[_Column]] = {}
    for target in targets:
        try:
            values = sampled_values(
                connector,
                target.schema_name,
                target.table_name,
                target.name,
                rows=_TARGET_VALUE_ROWS,
            )
        except Exception:
            logger.info(
                "value-overlap: could not read %s.%s",
                target.table_name,
                target.name,
                exc_info=True,
            )
            continue
        for value in values:
            if _informative(value):
                index.setdefault(value, []).append(target)

    work: list[tuple[_Column, list[_Candidate]]] = []
    for source in sources:
        try:
            values = sampled_values(
                connector,
                source.schema_name,
                source.table_name,
                source.name,
                rows=_SOURCE_VALUE_ROWS,
                limit=_MAX_SOURCE_VALUES,
            )
        except Exception:
            logger.info(
                "value-overlap: could not read %s.%s",
                source.table_name,
                source.name,
                exc_info=True,
            )
            continue
        candidates = candidates_for(source, values, index)
        if candidates:
            work.append((source, candidates))

    logger.info(
        "value-overlap FK discovery: %d key-like target(s), %d source column(s), "
        "%d with overlapping candidates",
        len(targets),
        len(sources),
        len(work),
    )
    if not work:
        return 0

    written = 0
    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        futures = {
            pool.submit(_decide_and_write, source, candidates): source
            for source, candidates in work
        }
        for future in as_completed(futures):
            if future.result():
                written += 1
    logger.info("value-overlap FK discovery: %d semantic FK edge(s) written", written)
    return written


def candidates_for(
    source: _Column,
    values: set[str],
    index: dict[str, list[_Column]],
) -> list[_Candidate]:
    """Targets on other tables holding enough of *source*'s values, best first."""
    informative = [value for value in values if _informative(value)]
    shared: Counter[str] = Counter()
    by_id: dict[str, _Column] = {}
    for value in informative:
        for target in index.get(value, ()):
            if target.table_id == source.table_id:
                continue
            shared[target.id] += 1
            by_id[target.id] = target
    total = len(informative)
    candidates = [
        _Candidate(by_id[target_id], count, total)
        for target_id, count in shared.items()
        if overlap_accepted(total, count)
    ]
    candidates.sort(key=lambda c: (-c.shared / c.source_values, -c.shared, c.target.id))
    return candidates[:_MAX_CANDIDATES]


def _profile_database(
    database_name: str, connector: SQLDatabase
) -> tuple[list[_Column], list[_Column]]:
    """Key-like target columns and candidate source columns of one database."""
    targets: list[_Column] = []
    sources: list[_Column] = []
    for table in fetch_columns_for_value_overlap(database_name):
        joinable = [
            col for col in table["columns"] if is_joinable_type(col["data_type"])
        ]
        if not joinable:
            continue
        try:
            stats = profile_columns(
                connector,
                table["schema_name"],
                table["name"],
                [col["name"] for col in joinable],
            )
        except Exception:
            logger.info(
                "value-overlap: profiling skipped for %s", table["name"], exc_info=True
            )
            continue
        pk = {name.casefold() for name in table["pk"]}
        for col in joinable:
            non_null, distinct = stats.get(col["name"], (0, 0))
            column = _Column(
                id=col["id"],
                name=col["name"],
                data_type=col["data_type"],
                description=col["description"],
                sample_values=tuple(col["sample_values"] or ()),
                attribute_id=col["attribute_id"],
                table_id=table["id"],
                table_name=table["name"],
                schema_name=table["schema_name"],
            )
            key_like = is_key_like(non_null, distinct, col["is_unique"])
            if key_like and col["attribute_id"]:
                targets.append(column)
            if (
                not key_like
                and not col["has_semantic_fk"]
                and col["name"].casefold() not in pk
                and distinct >= _MIN_SHARED_VALUES
            ):
                sources.append(column)
    return targets, sources


def _decide_and_write(source: _Column, candidates: list[_Candidate]) -> bool:
    try:
        selected = _llm_pick(source, candidates)
    except Exception:
        logger.exception(
            "value-overlap: LLM failed for %s.%s", source.table_name, source.name
        )
        return False
    if selected is None:
        return False
    if not selected.target.attribute_id or not merge_semantic_fk(
        source.id, selected.target.attribute_id
    ):
        return False
    logger.info(
        "resolve_semantic_fks [value overlap]: %s.%s -> %s.%s "
        "(%d of %d sampled values shared)",
        source.table_name,
        source.name,
        selected.target.table_name,
        selected.target.name,
        selected.shared,
        selected.source_values,
    )
    return True


def _describe(column: _Column) -> str:
    samples = ", ".join(column.sample_values[:_MAX_PROMPT_SAMPLES]) or "(none)"
    return (
        f"{column.table_name}.{column.name} ({column.data_type}) — "
        f"{column.description or '(no description)'} | samples: {samples}"
    )


def _llm_pick(source: _Column, candidates: list[_Candidate]) -> _Candidate | None:
    lines = [
        f"column_id={c.target.id} | {_describe(c.target)} | holds {c.shared} of the "
        f"column's {c.source_values} sampled values "
        f"({100 * c.shared / c.source_values:.0f}%)"
        for c in candidates
    ]
    human = (
        f"Column: {_describe(source)}\n\n"
        "Candidate key columns:\n" + "\n".join(lines) + "\n\n"
        "Which candidate does this column reference? Return its exact column_id, "
        "or null if none is a confident match."
    )
    result = invoke_with_structured_output(
        get_llm_client(max_tokens=4096),
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human)],
        FkHitSelection,
    )
    selected = result.selected_id if result is not None else None
    if not selected:
        return None
    for candidate in candidates:
        if candidate.target.id == selected:
            return candidate
    logger.warning(
        "value-overlap: LLM returned unknown id %r for %s.%s",
        selected,
        source.table_name,
        source.name,
    )
    return None
