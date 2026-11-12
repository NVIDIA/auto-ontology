# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Term families: copies of one schema told apart by a table-name prefix.

A warehouse migrated from one system to another often keeps both copies:
``OLD_ROOMS`` beside ``NEW_ROOMS``, ``OLD_BUILDING`` beside ``NEW_BUILDING``.
Each prefix is a *family*, and two tables with the same name after it are
*twins*. This pass finds the families and links each member table's Term to
its family, so later stages can tell the copies apart.

Algorithm, per schema and per pair of prefixes ``P`` / ``Q``
-------------------------------------------------------------
1. **Mirrored edges.** A data edge — declared FK or customer-SQL join —
   ``P_A.x — P_B.y`` whose counterpart ``Q_A.x — Q_B.y`` also exists, with
   column names compared after removing each table's own prefix.
2. **Twins.** Tables with the same base name whose prefix-stripped column sets
   overlap by Jaccard :data:`_MIN_TWIN_JACCARD`, or
   :data:`_MIN_EDGE_TWIN_JACCARD` when the table is an endpoint of a mirrored
   edge. History tables (``*_HIST``) are never twins.
3. **Confirmation.** Twins plus mirrored edges between twins must reach
   :data:`_MIN_PAIR_EVIDENCE`. One coincidental base name is not a family.
4. **Expansion.** A confirmed family also takes in every table with its prefix
   that data edges connect to a member, transitively but only through tables
   with that same prefix, so a shared ``DIM_DATE`` cannot carry it across the
   schema. History tables join this way.

A prefix in several confirmed pairs (``V1_``/``V2_``/``V3_``) gets one family
holding the twins of all of them. Pairs are not stored: twins are recovered as
members of sibling families with the same ``base_name``.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from auto_ontology.dal.term_families import (
    fetch_family_candidates,
    fetch_family_edges,
    replace_term_families,
)

logger = logging.getLogger(__name__)

_MIN_TWIN_JACCARD = 0.8
_MIN_EDGE_TWIN_JACCARD = 0.5
_MIN_PAIR_EVIDENCE = 3
_HISTORY_TOKENS = frozenset({"HIST", "HISTORY"})

#: An undirected edge between two (base name, stripped column) endpoints.
_Signature = frozenset[tuple[str, str]]


@dataclass
class TermFamily:
    """One prefix of one schema, and the Terms of its member tables."""

    schema_id: str
    prefix: str
    #: term id -> the member table's name without the prefix.
    members: dict[str, str] = field(default_factory=dict)
    twins: int = 0
    expanded: int = 0
    mirrored_edges: int = 0


@dataclass
class _Table:
    id: str
    schema_id: str
    name: str
    prefix: str
    base: str
    columns: frozenset[str]
    is_history: bool
    term_ids: list[str] = field(default_factory=list)


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^A-Za-z0-9]+", name.upper()) if t]


def _strip_prefix(name: str, prefix: str) -> str:
    """*name* upper-cased, without a leading ``<prefix>_``."""
    upper = name.upper()
    lead = f"{prefix}_"
    return upper[len(lead) :] if upper.startswith(lead) and upper != lead else upper


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def _index_tables(rows: list[dict[str, Any]]) -> dict[str, _Table]:
    """Prefixed tables by id; a name with fewer than two tokens has no prefix."""
    tables: dict[str, _Table] = {}
    for row in rows:
        table = tables.get(row["table_id"])
        if table is None:
            tokens = _tokens(row["table_name"])
            if len(tokens) < 2:
                continue
            prefix = tokens[0]
            table = tables[row["table_id"]] = _Table(
                id=row["table_id"],
                schema_id=row["schema_id"],
                name=row["table_name"],
                prefix=prefix,
                base="_".join(tokens[1:]),
                columns=frozenset(
                    _strip_prefix(c, prefix) for c in row.get("columns") or []
                ),
                is_history=tokens[-1] in _HISTORY_TOKENS,
            )
        table.term_ids.append(row["term_id"])
    return tables


def detect_term_families(
    rows: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> list[TermFamily]:
    """The families in *rows*, given the data *edges* between their tables.

    *rows* and *edges* have the shapes ``fetch_family_candidates`` and
    ``fetch_family_edges`` return. Pure: no I/O, so every rule is testable on
    hand-built inputs. Families come back ordered by schema, then prefix.
    """
    tables = _index_tables(rows)

    # (schema, prefix) -> base name -> table, and -> edge signatures.
    by_prefix: dict[tuple[str, str], dict[str, _Table]] = defaultdict(dict)
    signatures: dict[tuple[str, str], set[_Signature]] = defaultdict(set)
    neighbours: dict[str, set[str]] = defaultdict(set)
    for table in tables.values():
        by_prefix[(table.schema_id, table.prefix)].setdefault(table.base, table)
    for edge in edges:
        a = tables.get(edge["source_table_id"])
        b = tables.get(edge["target_table_id"])
        if a is None or b is None or a.id == b.id or a.schema_id != b.schema_id:
            continue
        if a.prefix != b.prefix:
            continue
        neighbours[a.id].add(b.id)
        neighbours[b.id].add(a.id)
        signatures[(a.schema_id, a.prefix)].add(
            frozenset(
                {
                    (a.base, _strip_prefix(edge["source_column"], a.prefix)),
                    (b.base, _strip_prefix(edge["target_column"], b.prefix)),
                }
            )
        )

    members: dict[tuple[str, str], dict[str, _Table]] = defaultdict(dict)
    mirrored_count: dict[tuple[str, str], int] = defaultdict(int)
    prefixes_by_schema: dict[str, list[str]] = defaultdict(list)
    for schema_id, prefix in sorted(by_prefix):
        prefixes_by_schema[schema_id].append(prefix)

    for schema_id, prefixes in prefixes_by_schema.items():
        for p, q in combinations(prefixes, 2):
            key_p, key_q = (schema_id, p), (schema_id, q)
            twins, mirrored = _confirm_pair(
                by_prefix[key_p],
                by_prefix[key_q],
                signatures[key_p] & signatures[key_q],
            )
            if twins is None:
                continue
            logger.info(
                "term family pair %s / %s: %d twin(s), %d mirrored edge(s)",
                p,
                q,
                len(twins),
                mirrored,
            )
            for base in twins:
                members[key_p][base] = by_prefix[key_p][base]
                members[key_q][base] = by_prefix[key_q][base]
            mirrored_count[key_p] += mirrored
            mirrored_count[key_q] += mirrored

    families: list[TermFamily] = []
    assigned: set[str] = set()
    for key in sorted(members):
        schema_id, prefix = key
        twin_ids = {table.id for table in members[key].values()}
        member_ids = _expand(twin_ids, neighbours)
        family = TermFamily(
            schema_id=schema_id,
            prefix=prefix,
            twins=len(twin_ids),
            expanded=len(member_ids - twin_ids),
            mirrored_edges=mirrored_count[key],
        )
        for table_id in sorted(member_ids, key=lambda i: tables[i].name):
            table = tables[table_id]
            for term_id in table.term_ids:
                if term_id in assigned:
                    logger.info(
                        "term family %s: Term %s of %s already belongs to another "
                        "family — skipped",
                        prefix,
                        term_id,
                        table.name,
                    )
                    continue
                assigned.add(term_id)
                family.members[term_id] = table.base
        families.append(family)
    return families


def _confirm_pair(
    p_tables: dict[str, _Table],
    q_tables: dict[str, _Table],
    shared_signatures: set[_Signature],
) -> tuple[list[str] | None, int]:
    """The twin base names of a confirmed pair, and its mirrored edge count.

    ``(None, 0)`` when the pair does not reach :data:`_MIN_PAIR_EVIDENCE`.
    """
    edge_bases = {base for signature in shared_signatures for base, _ in signature}
    twins: list[str] = []
    for base in sorted(p_tables.keys() & q_tables.keys()):
        a, b = p_tables[base], q_tables[base]
        if a.is_history or b.is_history:
            continue
        bar = _MIN_EDGE_TWIN_JACCARD if base in edge_bases else _MIN_TWIN_JACCARD
        if _jaccard(a.columns, b.columns) >= bar:
            twins.append(base)
    twin_set = set(twins)
    mirrored = sum(
        1
        for signature in shared_signatures
        if all(base in twin_set for base, _ in signature)
    )
    if len(twins) + mirrored < _MIN_PAIR_EVIDENCE:
        return None, 0
    return twins, mirrored


def _expand(seeds: set[str], neighbours: dict[str, set[str]]) -> set[str]:
    """*seeds* plus every table data edges connect them to.

    *neighbours* only ever links tables with the same prefix, which is what
    keeps the walk inside the seeds' family.
    """
    found = set(seeds)
    frontier = list(seeds)
    while frontier:
        for neighbour in neighbours.get(frontier.pop(), ()):
            if neighbour not in found:
                found.add(neighbour)
                frontier.append(neighbour)
    return found


def resolve_term_families(database_name: str) -> int:
    """Recompute the term families of *database_name*; return how many exist."""
    families = detect_term_families(
        fetch_family_candidates(database_name),
        fetch_family_edges(database_name),
    )
    linked = replace_term_families(
        database_name,
        [(f.schema_id, f.prefix, f.members) for f in families],
    )
    for family in families:
        logger.info(
            "term family %s: %d term(s) (%d twin table(s), %d by edges), "
            "%d mirrored edge(s)",
            family.prefix,
            len(family.members),
            family.twins,
            family.expanded,
            family.mirrored_edges,
        )
    logger.info(
        "resolve_term_families: %d family(ies), %d term(s) linked in %s",
        len(families),
        linked,
        database_name,
    )
    return len(families)
