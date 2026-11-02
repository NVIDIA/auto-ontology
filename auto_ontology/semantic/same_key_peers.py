# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Link two tables both ways when they store the same primary key.

``resolve_semantic_fks`` never reaches this case: it starts only from columns
without a ColumnAttribute, which a real key always has, and its vector search
offers only ``is_unique`` targets, so a copy of a key that repeats rows of the
same entity is no target either. Reversing every semantic FK is not the fix:
``join_path_edge`` is one-directional so that two foreign keys into one
attribute are never joined to each other.

This pass compares the identifier column of every table in a database with
every other one and, for each pair that holds the same key, points each column
at the other's ColumnAttribute. Two columns hold the same key when:

- the smaller distinct set, compared as text, is at least
  ``_MIN_CONTAINMENT`` inside the larger and has ``_MIN_DISTINCT`` values;
- each is the grain of its own table -- nearly one row per key, or repeats
  that are the same entity on the table's name and business-id columns;
- names and business ids agree for the shared keys, when both tables have
  them, and the column names name the same entity when they don't.

Only a table's own identifier column is compared. A column that overlaps
another table's key without being its own table's identifier is a parent or
foreign key, not a second copy of the primary key.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any

from sqlglot import exp

from auto_ontology.connectors.base import SQLDatabase
from auto_ontology.dal.attributes import (
    fetch_columns_with_attribute,
    merge_semantic_fk,
    set_semantic_fk_join_default,
)
from auto_ontology.semantic.semantic_fk import _resolve_connector
from auto_ontology.utils.sql_identifiers import quoted_identifier

logger = logging.getLogger(__name__)

#: Fewer distinct keys than this overlap by chance: a small lookup's codes
#: 1..12 sit inside nearly any surrogate key.
_MIN_DISTINCT = 20
_MIN_CONTAINMENT = 0.95
#: Distinct keys per non-null row at which a column is its table's grain.
_MIN_GRAIN_RATIO = 0.9
#: Share of the shared keys on which a name or business id must agree.
_MIN_AGREEMENT = 0.95
#: A column with more (key, description) groups than this is skipped rather
#: than held in memory.
_MAX_PROFILE_GROUPS = 50_000
#: Keys of the smaller set looked up in the larger before the full comparison.
_PROBE_SIZE = 32

_KEY_NAME = re.compile(r"(?:(?P<stem>.+)_)?(?:key|id)")
_DESCRIPTIVE_NAMES = {
    "name": re.compile(r"(?:(?P<stem>.+)_)?name"),
    "id": re.compile(r"(?:(?P<stem>.+)_)?id"),
}


@dataclass
class _KeyProfile:
    """One identifier column's keys, with its table's descriptive values per key."""

    table_id: str
    table_name: str
    column_id: str
    column_name: str
    attribute_id: str
    entity: str
    descriptive: dict[str, list[str]]
    rows: int
    #: Key -> the distinct tuples of descriptive values found with it, in
    #: the order of :attr:`columns`.
    groups: dict[str, set[tuple[str | None, ...]]]

    @property
    def columns(self) -> list[str]:
        return [
            column for kind in _DESCRIPTIVE_NAMES for column in self.descriptive[kind]
        ]

    @property
    def is_grain(self) -> bool:
        """Whether the column is one entity per key, not a key repeated per fact row."""
        if not self.rows:
            return False
        if len(self.groups) / self.rows >= _MIN_GRAIN_RATIO:
            return True
        return bool(self.columns) and all(
            len(found) == 1 for found in self.groups.values()
        )

    def values(self, column: str) -> dict[str, set[str]]:
        """The non-null values of descriptive *column* found with each key."""
        index = self.columns.index(column)
        return {
            key: {found[index] for found in tuples if found[index] is not None}
            for key, tuples in self.groups.items()
        }


def link_same_key_peers(database_name: str) -> int:
    """Link every pair of tables in *database_name* that store the same key.

    Each pair gets two semantic FKs, one from each column to the other's
    ColumnAttribute, and each is made the join-path default for its table and
    target. Existing semantic FKs from other tables onto either key are left
    as they are. Returns the number of pairs linked.
    """
    connector = _resolve_connector(database_name)
    if connector is None:
        return 0
    tables: dict[str, list[dict[str, Any]]] = {}
    for row in fetch_columns_with_attribute(database_name):
        tables.setdefault(row["table_id"], []).append(row)

    profiles: list[_KeyProfile] = []
    for columns in tables.values():
        if columns[0]["is_junction_table"]:
            continue
        names = [column["column_name"] for column in columns]
        identifiers = _identifiers(
            columns[0]["table_name"], columns[0]["table_pk"], names
        )
        for column in columns:
            if not column["attribute_id"] or column["column_name"] not in identifiers:
                continue
            profile = _profile(connector, column, names)
            if (
                profile is not None
                and len(profile.groups) >= _MIN_DISTINCT
                and profile.is_grain
            ):
                profiles.append(profile)

    linked = 0
    for index, a in enumerate(profiles):
        for b in profiles[index + 1 :]:
            if a.table_id != b.table_id and _same_key(a, b) and _link(a, b):
                linked += 1
    return linked


def _same_key(a: _KeyProfile, b: _KeyProfile) -> bool:
    """Whether *a* and *b*, each already its table's grain, hold one key."""
    small, large = sorted((a, b), key=lambda profile: len(profile.groups))
    if len(small.groups) < _MIN_DISTINCT:
        return False
    allowed_misses = math.floor((1 - _MIN_CONTAINMENT) * len(small.groups))
    probe = sorted(small.groups)[:_PROBE_SIZE]
    if sum(key not in large.groups for key in probe) > allowed_misses:
        return False
    if len(small.groups.keys() - large.groups.keys()) > allowed_misses:
        return False
    agreement = _descriptions_agree(a, b)
    if agreement is None:
        return _same_entity(a.entity, b.entity)
    return agreement


def _descriptions_agree(a: _KeyProfile, b: _KeyProfile) -> bool | None:
    """Whether names and business ids agree for the shared keys.

    ``None`` when the tables have nothing comparable. Two values agree when
    they are equal ignoring case and surrounding space; a key repeated with
    several values agrees when any of them match.
    """
    shared = a.groups.keys() & b.groups.keys()
    compared_any = False
    for kind in _DESCRIPTIVE_NAMES:
        for column_a, column_b in _column_pairs(
            a.descriptive[kind], b.descriptive[kind]
        ):
            values_a, values_b = a.values(column_a), b.values(column_b)
            compared = agreed = 0
            for key in shared:
                found_a, found_b = values_a.get(key), values_b.get(key)
                if found_a and found_b:
                    compared += 1
                    agreed += bool(found_a & found_b)
            if not compared:
                continue
            compared_any = True
            if agreed < _MIN_AGREEMENT * compared:
                return False
    return True if compared_any else None


def _column_pairs(a: list[str], b: list[str]) -> list[tuple[str, str]]:
    """Descriptive columns to compare: same-named ones, else the only one on each side."""
    by_name = {column.casefold(): column for column in b}
    pairs = [
        (column, by_name[column.casefold()])
        for column in a
        if column.casefold() in by_name
    ]
    if not pairs and len(a) == 1 and len(b) == 1:
        pairs = [(a[0], b[0])]
    return pairs


def _link(a: _KeyProfile, b: _KeyProfile) -> bool:
    if not (
        merge_semantic_fk(a.column_id, b.attribute_id)
        and merge_semantic_fk(b.column_id, a.attribute_id)
    ):
        return False
    set_semantic_fk_join_default(a.table_id, b.attribute_id, a.column_id)
    set_semantic_fk_join_default(b.table_id, a.attribute_id, b.column_id)
    logger.info(
        "same-key peers linked: %s.%s <-> %s.%s",
        a.table_name,
        a.column_name,
        b.table_name,
        b.column_name,
    )
    return True


# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------


def _stem(pattern: re.Pattern[str], column_name: str) -> str | None:
    """The part of *column_name* before the suffix *pattern* names, or ``None``."""
    match = pattern.fullmatch(column_name.casefold())
    if match is None:
        return None
    return match.group("stem") or ""


def _singular(name: str) -> str:
    return name[:-1] if name.endswith("s") and not name.endswith("ss") else name


def _same_entity(a: str, b: str) -> bool:
    """Whether names *a* and *b* name one entity: ``organization``, ``fac_organization``."""
    a, b = _singular(a.casefold()), _singular(b.casefold())
    return a == b or a.endswith(f"_{b}") or b.endswith(f"_{a}")


def _names_table(stem: str | None, table_name: str) -> bool:
    """Whether a column whose name starts with *stem* describes *table_name* itself."""
    if stem is None:
        return False
    return not stem or _same_entity(stem, table_name)


def _identifiers(
    table_name: str, pk: list[str] | None, column_names: list[str]
) -> set[str]:
    """The columns that are *table_name*'s own identifier.

    The declared primary key when it is one column; none when it is several,
    since no single column is then the row's identity. Otherwise every key or
    id column named for the table: ``ORGANIZATION_KEY`` on ``FAC_ORGANIZATION``,
    but not ``FCLT_ORG_PARENT_KEY`` on ``FCLT_ORGANIZATION``.
    """
    if pk:
        if len(pk) > 1:
            return set()
        declared = {
            name for name in column_names if name.casefold() == pk[0].casefold()
        }
        if declared:
            return declared
    return {
        name
        for name in column_names
        if _names_table(_stem(_KEY_NAME, name), table_name)
    }


def _descriptive_columns(
    table_name: str, column_names: list[str], key_column: str
) -> dict[str, list[str]]:
    """The name and business-id columns that describe *table_name*'s entity.

    Those named for the table when there are any, otherwise the table's only
    column of that kind; a table with several unrelated names has none.
    """
    result: dict[str, list[str]] = {}
    for kind, pattern in _DESCRIPTIVE_NAMES.items():
        found = [
            name
            for name in column_names
            if name != key_column and _stem(pattern, name) is not None
        ]
        own = [name for name in found if _names_table(_stem(pattern, name), table_name)]
        result[kind] = own or (found if len(found) == 1 else [])
    return result


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


def _profile(
    connector: SQLDatabase, column: dict[str, Any], column_names: list[str]
) -> _KeyProfile | None:
    """Read *column*'s keys with their descriptive values, or ``None`` on failure."""
    table_name = column["table_name"]
    key = column["column_name"]
    descriptive = _descriptive_columns(table_name, column_names, key)
    described = [name for kind in _DESCRIPTIVE_NAMES for name in descriptive[kind]]
    sql = _profile_sql(connector, column.get("schema_name"), table_name, key, described)
    try:
        frame = connector.execute(sql)
    except Exception:
        logger.info(
            "same-key profile skipped for %s.%s", table_name, key, exc_info=True
        )
        return None
    if len(frame) > _MAX_PROFILE_GROUPS:
        logger.info(
            "same-key profile skipped for %s.%s: more than %d keys",
            table_name,
            key,
            _MAX_PROFILE_GROUPS,
        )
        return None

    rows = 0
    groups: dict[str, set[tuple[str | None, ...]]] = {}
    for record in frame.itertuples(index=False, name=None):
        value = _text(record[0])
        if value is None:
            continue
        found = tuple(_description(item) for item in record[1:-1])
        groups.setdefault(value, set()).add(found)
        rows += int(record[-1])

    stem = _stem(_KEY_NAME, key)
    return _KeyProfile(
        table_id=column["table_id"],
        table_name=table_name,
        column_id=column["column_id"],
        column_name=key,
        attribute_id=column["attribute_id"],
        entity=stem or table_name,
        descriptive=descriptive,
        rows=rows,
        groups=groups,
    )


def _profile_sql(
    connector: SQLDatabase,
    schema_name: str | None,
    table_name: str,
    key: str,
    described: list[str],
) -> str:
    """Group *key*'s non-null rows by its text and the *described* columns' text."""
    dialect = getattr(connector, "dialect", None)
    aliases = [f"d{index}" for index in range(len(described))]
    casts = ", ".join(
        f"{_as_text(name, dialect)} AS {alias}"
        for name, alias in zip([key, *described], ["k", *aliases])
    )
    grouped = ", ".join(["k", *aliases])
    return (
        f"SELECT {grouped}, COUNT(*) AS n FROM (SELECT {casts} "
        f"FROM {connector.qualify(schema_name, table_name)} "
        f"WHERE {quoted_identifier(key, dialect)} IS NOT NULL) AS v "
        f"GROUP BY {grouped} LIMIT {_MAX_PROFILE_GROUPS + 1}"
    )


def _as_text(column: str, dialect: str | None) -> str:
    expression = exp.cast(exp.column(column, quoted=True), "TEXT")
    try:
        return expression.sql(dialect=dialect or None)
    except ValueError:
        return expression.sql()


def _text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def _description(value: Any) -> str | None:
    text = _text(value)
    return text.casefold() if text is not None else None
