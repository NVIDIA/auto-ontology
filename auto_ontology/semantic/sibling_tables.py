# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Notes that tell near-duplicate tables apart.

Warehouses keep several versions of one entity side by side: an all-time copy
and a recent one, a catalog view and a registrar view, a detail table and its
summary. Every other compile step looks at one table at a time, so nothing
records how such tables differ, and text-to-SQL picks between them by name.

This post-pass finds families of similar tables, measures what separates their
members (the period each covers, its size, the columns only it has), and asks
the LLM for one short note per table saying how it differs from its siblings
and when to use it. The note is stored on the catalog table and shown to table
selection and SQL generation beside the table's description.
"""

from __future__ import annotations

import difflib
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from auto_ontology.dal.datasources import (
    fetch_tables_for_sibling_notes,
    store_table_sibling_notes,
)
from auto_ontology.semantic.visit_enter import _execute_capped, _resolve_connector
from auto_ontology.utils.llm_invoke import (
    get_llm_client,
    invoke_with_structured_output,
)
from auto_ontology.utils.sql_identifiers import quoted_identifier

if TYPE_CHECKING:
    from auto_ontology.connectors.base import SQLDatabase

logger = logging.getLogger(__name__)

# "off" skips the pass, for a deployment that wants compile unchanged.
SIBLING_NOTES_ENABLED = os.environ.get("SEMANTIC_SIBLING_NOTES", "on").lower() != "off"

# A column found in more than this share of a database's tables is plumbing
# (load timestamps, audit ids) and says nothing about whether two tables hold
# the same entity, so it is ignored when comparing them.
_GENERIC_COLUMN_SHARE = 0.3
# Two tables are siblings when their names are related and they share at least
# this many non-generic columns...
_MIN_SHARED_COLUMNS = 2
# ...or, whatever their names, when their non-generic columns overlap this much.
_MIN_COLUMN_JACCARD = 0.6
_MIN_SHARED_COLUMNS_BY_OVERLAP = 4
# Names are related when one's words all appear in the other, when they share
# half their words, or when they are spelled nearly alike (FAC_ / FCLT_).
_MIN_NAME_TOKEN_JACCARD = 0.5
_MIN_NAME_RATIO = 0.8
# A family larger than this is usually several families chained together
# (BUILDING -> BUILDING_ADDRESS -> ...), so it is split again with the strict
# link: related names AND this much column overlap. Whatever is still too
# large after that is a naming convention, and a note comparing a dozen
# tables would help nobody.
_MAX_FAMILY_SIZE = 8
_STRICT_COLUMN_JACCARD = 0.5
# Row counts are capped so a huge table costs a bounded scan.
_ROW_COUNT_CAP = 1_000_000
# How many period columns are measured per table, and how many distinctive
# columns a note's facts list.
_MAX_PERIOD_COLUMNS = 3
_MAX_LISTED_COLUMNS = 20
_PERIOD_NAME = re.compile(r"(TERM|DATE|YEAR|MONTH|PERIOD|QUARTER|_DT$|_TS$|TIME)", re.I)
_PERIOD_TYPE = re.compile(r"(date|time)", re.I)
# Flags (IS_OFFERED_FALL_TERM) and labels (TERM_DESCRIPTION) match the period
# pattern, but a flag's range is N..Y and a label's is alphabetical, not
# chronological, so neither says what period a table covers.
_NOT_A_PERIOD = re.compile(r"(^IS_|^HAS_|FLAG|DESC|NAME|TITLE|TEXT|LABEL)", re.I)
_FLAG_VALUES = {"", "0", "1", "N", "Y", "T", "F", "TRUE", "FALSE", "NO", "YES"}
# MIN/MAX of a text column is alphabetical. That is chronological only when the
# value leads with the year ('2003JA', '2015-06-30'); '01-APR-02' is not.
_ORDERED_TYPE = re.compile(r"(date|time|int|num|dec|float|double|real|year)", re.I)
_LEADS_WITH_YEAR = re.compile(r"^(1[89]|20)\d\d")


@dataclass
class TableSpec:
    """A catalog table, as much of it as sibling detection needs."""

    id: str
    name: str
    schema_name: str
    columns: list[dict[str, Any]]
    description: str = ""

    @property
    def column_names(self) -> set[str]:
        return {str(c["name"]).upper() for c in self.columns}


@dataclass
class TableFacts:
    """What was measured about one family member."""

    table: TableSpec
    row_count: int | None = None
    periods: dict[str, tuple[str, str]] = field(default_factory=dict)
    only_here: list[str] = field(default_factory=list)


def _name_tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^A-Za-z0-9]+", name.upper()) if t]


def _names_related(a: str, b: str) -> bool:
    ta, tb = set(_name_tokens(a)), set(_name_tokens(b))
    if not ta or not tb:
        return False
    if ta <= tb or tb <= ta:
        return True
    if len(ta & tb) / len(ta | tb) >= _MIN_NAME_TOKEN_JACCARD:
        return True
    return (
        difflib.SequenceMatcher(None, a.upper(), b.upper()).ratio() >= _MIN_NAME_RATIO
    )


def generic_columns(tables: list[TableSpec]) -> set[str]:
    """Columns that appear in too many tables to say anything about kinship."""
    if len(tables) < 4:
        return set()
    counts = Counter(name for t in tables for name in t.column_names)
    limit = _GENERIC_COLUMN_SHARE * len(tables)
    return {name for name, n in counts.items() if n > limit}


def _groups(tables: list[TableSpec], linked) -> list[list[TableSpec]]:
    """The connected groups of *tables* under the pairwise test *linked*."""
    parent = {t.id: t.id for t in tables}

    def root(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(tables):
        for b in tables[i + 1 :]:
            if a.schema_name == b.schema_name and linked(a, b):
                parent[root(a.id)] = root(b.id)
    groups: dict[str, list[TableSpec]] = {}
    for t in tables:
        groups.setdefault(root(t.id), []).append(t)
    return list(groups.values())


def find_sibling_families(tables: list[TableSpec]) -> list[list[TableSpec]]:
    """Group a database's tables into families of near-duplicates.

    Two tables in the same schema are linked when their names are related and
    they share real columns, or when their columns overlap heavily whatever
    their names. Families are the connected groups of two or more tables. A
    group larger than ``_MAX_FAMILY_SIZE`` is split again with a stricter link,
    and what is still too large is left without notes.
    """
    generic = generic_columns(tables)
    cols = {t.id: t.column_names - generic for t in tables}

    def overlap(a: TableSpec, b: TableSpec) -> tuple[int, float]:
        shared = cols[a.id] & cols[b.id]
        union = cols[a.id] | cols[b.id]
        return len(shared), (len(shared) / len(union) if union else 0.0)

    def loose(a: TableSpec, b: TableSpec) -> bool:
        shared, jaccard = overlap(a, b)
        if _names_related(a.name, b.name) and shared >= _MIN_SHARED_COLUMNS:
            return True
        return (
            shared >= _MIN_SHARED_COLUMNS_BY_OVERLAP and jaccard >= _MIN_COLUMN_JACCARD
        )

    def strict(a: TableSpec, b: TableSpec) -> bool:
        shared, jaccard = overlap(a, b)
        return (
            _names_related(a.name, b.name)
            and shared >= _MIN_SHARED_COLUMNS
            and jaccard >= _STRICT_COLUMN_JACCARD
        )

    families = []
    for group in _groups(tables, loose):
        parts = _groups(group, strict) if len(group) > _MAX_FAMILY_SIZE else [group]
        for members in parts:
            if len(members) < 2:
                continue
            if len(members) > _MAX_FAMILY_SIZE:
                logger.info(
                    "Sibling notes: skipping a group of %d similar tables (%s, ...)",
                    len(members),
                    ", ".join(m.name for m in members[:4]),
                )
                continue
            families.append(sorted(members, key=lambda m: m.name))
    return sorted(families, key=lambda f: f[0].name)


def _period_columns(table: TableSpec, generic: set[str], shared: set[str]) -> list[str]:
    """The columns whose range tells what period a table covers.

    Columns the family shares come first, since a range is only comparable
    across siblings on the same column. Generic columns (load timestamps) are
    left out: when a row was loaded says nothing about what it describes.
    """
    candidates = [
        c["name"]
        for c in table.columns
        if str(c["name"]).upper() not in generic
        and not _NOT_A_PERIOD.search(str(c["name"]))
        and (
            _PERIOD_NAME.search(str(c["name"]))
            or _PERIOD_TYPE.search(str(c.get("data_type") or ""))
        )
    ]
    candidates.sort(key=lambda name: str(name).upper() not in shared)
    return candidates[:_MAX_PERIOD_COLUMNS]


def _scalar(connector: "SQLDatabase", sql: str) -> list[Any] | None:
    try:
        df = _execute_capped(connector, sql)
    except Exception:
        logger.warning("Sibling notes: query failed: %s", sql, exc_info=True)
        return None
    if df is None or df.empty:
        return None
    return list(df.iloc[0].tolist())


def measure_family(
    family: list[TableSpec],
    connector: "SQLDatabase | None",
    generic: set[str],
) -> list[TableFacts]:
    """Row counts, period coverage and distinctive columns for each member.

    A column is distinctive when at most half the family has it. Strictly
    unique would hide what sets a sub-group apart: three catalog tables that
    share forty catalog columns would each show none.
    """
    facts = []
    family_cols = [t.column_names - generic for t in family]
    shared = set.intersection(*family_cols) if family_cols else set()
    holders = Counter(name for cols in family_cols for name in cols)
    rare = {name for name, n in holders.items() if n <= len(family) / 2}
    dialect = getattr(connector, "dialect", None)
    for table, own in zip(family, family_cols):
        only_here = sorted(
            c["name"] for c in table.columns if str(c["name"]).upper() in own & rare
        )
        fact = TableFacts(table=table, only_here=only_here)
        if connector is not None:
            qualified = connector.qualify(table.schema_name, table.name)
            row = _scalar(
                connector,
                f"SELECT COUNT(*) FROM (SELECT 1 AS one FROM {qualified} LIMIT {_ROW_COUNT_CAP + 1}) capped",
            )
            if row and row[0] is not None:
                fact.row_count = int(row[0])
            types = {c["name"]: str(c.get("data_type") or "") for c in table.columns}
            for column in _period_columns(table, generic, shared):
                quoted = quoted_identifier(column, dialect)
                row = _scalar(
                    connector, f"SELECT MIN({quoted}), MAX({quoted}) FROM {qualified}"
                )
                if not row or row[0] is None or row[1] is None:
                    continue
                low, high = str(row[0]), str(row[1])
                # A flag, or a key that repeats a range already listed, adds nothing.
                if {low.upper(), high.upper()} <= _FLAG_VALUES or (
                    low,
                    high,
                ) in fact.periods.values():
                    continue
                ordered = _ORDERED_TYPE.search(types.get(column, "")) or (
                    _LEADS_WITH_YEAR.match(low) and _LEADS_WITH_YEAR.match(high)
                )
                if ordered:
                    fact.periods[column] = (low, high)
        facts.append(fact)
    return facts


def format_family_facts(facts: list[TableFacts]) -> str:
    """The measured differences, one block per table, for the note prompt."""
    blocks = []
    for f in facts:
        lines = [f"Table {f.table.name}"]
        if f.table.description:
            lines.append(f"  Business definition: {f.table.description}")
        if f.row_count is not None:
            size = (
                f"more than {_ROW_COUNT_CAP:,}"
                if f.row_count > _ROW_COUNT_CAP
                else f"{f.row_count:,}"
            )
            lines.append(f"  Rows: {size}")
        for column, (low, high) in f.periods.items():
            lines.append(f"  {column} ranges from {low} to {high}")
        lines.append(f"  Columns: {len(f.table.columns)}")
        if f.only_here:
            listed = ", ".join(f.only_here[:_MAX_LISTED_COLUMNS])
            more = len(f.only_here) - _MAX_LISTED_COLUMNS
            lines.append(
                "  Has columns most similar tables lack: "
                + listed
                + (f" and {more} more" if more > 0 else "")
            )
        else:
            lines.append("  Has columns most similar tables lack: none")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


_SYSTEM_PROMPT = """You write short notes that help a SQL assistant choose between similar tables
in one database. The tables below look alike: related names, overlapping columns.

For EACH table, write one or two sentences that say:
- what it holds that its siblings do not, and how it differs from them: the period
  it covers, its size, its level of detail, the columns most of them lack;
- when to use it rather than a sibling, naming the sibling.

Rules:
- Use only the facts given. Quote the measured ranges as they are.
- Do not guess a business meaning the names, definitions and columns do not support.
- Refer to tables by their exact names. Write no pipe characters.
- Return one note for every table listed, and no others."""


class _SiblingNote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table: str = Field(..., description="Exact table name.")
    note: str = Field(..., description="One or two sentences for this table.")


class _SiblingNotes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    notes: list[_SiblingNote]


def write_family_notes(facts: list[TableFacts]) -> dict[str, str]:
    """One note per family member, keyed by table id."""
    by_name = {f.table.name.upper(): f.table.id for f in facts}
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=format_family_facts(facts)),
    ]
    result = invoke_with_structured_output(
        get_llm_client(temperature=0.0), messages, _SiblingNotes
    )
    notes = {}
    for item in getattr(result, "notes", None) or []:
        table_id = by_name.get(item.table.strip().upper())
        text = " ".join(item.note.replace("|", "/").split())
        if table_id and text:
            notes[table_id] = text
    return notes


def write_sibling_table_notes(database_name: str) -> int:
    """Find each family of similar tables in a database and store a note per member.

    Every table of the database gets its note rewritten, and a table that no
    longer has siblings has its note cleared, so a recompile never leaves a
    stale comparison behind. Returns the number of notes written.
    """
    if not SIBLING_NOTES_ENABLED:
        logger.info("Sibling notes: off (SEMANTIC_SIBLING_NOTES=off)")
        return 0

    tables = [TableSpec(**row) for row in fetch_tables_for_sibling_notes(database_name)]
    generic = generic_columns(tables)
    families = find_sibling_families(tables)
    connector = _resolve_connector(database_name)
    if connector is None:
        logger.warning(
            "Sibling notes: no connector for %r; notes will carry no measurements",
            database_name,
        )

    notes: dict[str, str | None] = {t.id: None for t in tables}
    for family in families:
        logger.info(
            "Sibling notes: family of %d: %s",
            len(family),
            ", ".join(t.name for t in family),
        )
        try:
            notes.update(write_family_notes(measure_family(family, connector, generic)))
        except Exception:
            logger.warning(
                "Sibling notes: could not write notes for %s",
                ", ".join(t.name for t in family),
                exc_info=True,
            )
    store_table_sibling_notes(notes)
    written = sum(1 for note in notes.values() if note)
    logger.info(
        "Sibling notes: %d note(s) across %d famil(ies)", written, len(families)
    )
    return written
