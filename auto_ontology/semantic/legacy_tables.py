# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detect legacy tables before semantic compilation.

A database often keeps an outdated copy of a table next to the one that
replaced it, with nothing in the catalog saying which is which. Compiling both
gives retrieval two near-identical Terms and lets semantic FKs route through the
outdated copy. This pass infers the legacy side so compilation can skip it.

Algorithm
---------
1. **Pair** tables of one schema that look like two versions of one thing:

   * *prefix pairs* share everything after a different leading name token
     (``A_ORDERS`` / ``B_ORDERS``). Column names are compared with the table's
     own prefix token removed, so ``A_ORDERS_KEY`` and ``B_ORDERS_KEY`` match;
   * *marker pairs* differ only by a legacy marker or version token
     (``ORDERS`` / ``ORDERS_OLD``, ``ORDERS_V1`` / ``ORDERS_V2``).

   A pair needs a column-name Jaccard similarity of at least
   ``_MIN_COLUMN_SIMILARITY``.

2. **Gather evidence** per pair: name markers, extra columns, logged query
   usage, and — through bounded read-only probes — row counts, the latest value
   of a shared load/update date, and the overlap of shared key values. Prefix
   pairs also share family evidence: which prefix has ``_HIST`` tables and which
   one other tables reference by column name.

3. **Decide.** A weighted score decides clear cases; prefix pairs are scored as
   a family, so one family's pairs agree. The LLM judges whatever is left, and a
   pair it cannot decide stays unmarked. A pair whose key values overlap less
   than ``_MIN_KEY_OVERLAP`` holds different data and is never marked.

4. **Generalise.** A prefix or marker token confirmed on enough pairs is how
   this database spells "legacy", so every other table carrying it is marked
   too — including tables with no close counterpart.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from auto_ontology.connectors import get_connectors
from auto_ontology.connectors.base import SQLDatabase
from auto_ontology.dal.datasources import (
    fetch_tables_for_legacy_detection,
    set_legacy_tables,
)
from auto_ontology.retrieval.text_to_sql.db_probe.executor import ProbeExecutor
from auto_ontology.semantic.models import LegacyTableJudgement
from auto_ontology.utils.llm_invoke import (
    get_llm_client,
    invoke_with_structured_output,
)
from auto_ontology.utils.sql_identifiers import quoted_identifier

logger = logging.getLogger(__name__)

_LEGACY_MARKERS = frozenset(
    {
        "LEGACY",
        "OLD",
        "BAK",
        "BACKUP",
        "ARCHIVE",
        "ARCHIVED",
        "DEPRECATED",
        "OBSOLETE",
    }
)
_VERSION_TOKEN = re.compile(r"^V(\d+)$")
_FRESHNESS_TOKENS = frozenset({"LOAD", "LOADED", "UPDATE", "UPDATED", "MODIFIED"})
_DATE_TOKENS = frozenset({"DATE", "TIME", "TIMESTAMP", "TS", "AT", "DT"})
_HISTORY_TOKENS = frozenset({"HIST", "HISTORY"})
_KEY_SUFFIXES = ("_ID", "_NUMBER", "_CODE", "_NAME")

_MIN_COLUMN_SIMILARITY = 0.8
_MIN_KEY_OVERLAP = 0.5
_DECISIVE_SCORE = 2.5
_KEY_CANDIDATES = 2
_PROBES_PER_PAIR = 2 + 1 + 3 * _KEY_CANDIDATES
#: Decided prefix pairs needed before a prefix counts as this database's
#: legacy spelling. One pair can be a coincidence; a marker token is explicit.
_MIN_PREFIX_PATTERN_PAIRS = 2

_W_NAME = 3.0
_W_FRESHNESS = 2.0
_W_USAGE = 2.0
_W_FAMILY = 1.5
_W_COLUMNS = 1.0
_W_ROWS = 0.5

_SYSTEM_PROMPT = """\
You are a database curator. Two tables in one database hold near-identical
columns. Decide whether one of them is an outdated legacy copy that the other
replaced, using only the evidence given: names, columns each has alone, row
counts, the latest load or update dates, logged query usage, history tables,
and references from other tables.

Return the exact table_id of the outdated table, or null when the evidence does
not show that either one is outdated. Two tables can legitimately coexist, so
return null unless the evidence points one way.
"""


@dataclass
class _Pair:
    """Two tables that look like two versions of one thing.

    ``score`` is signed toward *second*: positive means *first* is the legacy
    copy, negative means *second* is.
    """

    first: dict[str, Any]
    second: dict[str, Any]
    kind: str
    #: Leading name tokens of a prefix pair, ``(first, second)``; empty otherwise.
    prefixes: tuple[str, str] | tuple[()]
    shared: dict[str, tuple[str, str]]
    first_only: list[str]
    second_only: list[str]
    similarity: float
    score: float = 0.0
    key_overlap: float | None = None
    evidence: list[str] = field(default_factory=list)

    def add(self, weight: float, message: str) -> None:
        self.score += weight
        self.evidence.append(f"{message} ({weight:+.1f})")


def detect_legacy_tables(database_name: str) -> int:
    """Mark the legacy tables of *database_name*; return how many were marked.

    Replaces the database's previous flags, so a table no longer judged legacy
    is compiled again on the next pass.
    """
    tables = fetch_tables_for_legacy_detection(database_name)
    pairs = _find_pairs(tables)
    if not pairs:
        set_legacy_tables(database_name, [])
        logger.info("detect_legacy_tables: no candidate pairs in %s", database_name)
        return 0
    logger.info(
        "detect_legacy_tables: %d candidate pair(s) in %s", len(pairs), database_name
    )

    for pair in pairs:
        _score_catalog(pair)
    _score_families(pairs, tables)
    connector = _resolve_connector(database_name)
    if connector is not None:
        with ProbeExecutor(
            connector, max_calls=_PROBES_PER_PAIR * len(pairs)
        ) as executor:
            for pair in pairs:
                _score_probes(pair, executor, connector)
    else:
        logger.info(
            "detect_legacy_tables: no connector for %s, deciding without probes",
            database_name,
        )

    decided = _decide(pairs)
    decided.update(_generalise(decided, pairs, tables))
    decisions = _resolve_chains(decided)
    set_legacy_tables(database_name, decisions)
    names = {t["id"]: t["name"] for t in tables}
    for decision in decisions:
        logger.info(
            "detect_legacy_tables: %s is legacy, replaced by %s — %s",
            names.get(decision["table_id"]),
            names.get(decision["replacement_table_id"] or "", "nothing"),
            decision["reason"],
        )
    return len(decisions)


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^A-Za-z0-9]+", name.upper()) if t]


def _split_prefix(name: str) -> tuple[str, str] | None:
    tokens = _tokens(name)
    if len(tokens) < 2:
        return None
    return tokens[0], "_".join(tokens[1:])


def _is_marker(token: str) -> bool:
    return token in _LEGACY_MARKERS or bool(_VERSION_TOKEN.match(token))


def _marker_base(name: str) -> str:
    return "_".join(t for t in _tokens(name) if not _is_marker(t))


def _normalized_columns(table: dict[str, Any], prefix: str | None) -> dict[str, str]:
    """Upper-cased column names, without *prefix* — mapped to the real name."""
    out: dict[str, str] = {}
    lead = f"{prefix}_" if prefix else None
    for column in table["columns"]:
        name = str(column["name"])
        key = name.upper()
        if lead and key.startswith(lead) and len(key) > len(lead):
            key = key[len(lead) :]
        out.setdefault(key, name)
    return out


def _make_pair(
    first: dict[str, Any],
    second: dict[str, Any],
    kind: str,
    prefixes: tuple[str, str] | tuple[()],
) -> _Pair | None:
    first_cols = _normalized_columns(first, prefixes[0] if prefixes else None)
    second_cols = _normalized_columns(second, prefixes[1] if prefixes else None)
    union = first_cols.keys() | second_cols.keys()
    if not union:
        return None
    common = first_cols.keys() & second_cols.keys()
    similarity = len(common) / len(union)
    if similarity < _MIN_COLUMN_SIMILARITY:
        return None
    return _Pair(
        first=first,
        second=second,
        kind=kind,
        prefixes=prefixes,
        shared={k: (first_cols[k], second_cols[k]) for k in sorted(common)},
        first_only=sorted(first_cols[k] for k in first_cols.keys() - common),
        second_only=sorted(second_cols[k] for k in second_cols.keys() - common),
        similarity=similarity,
    )


def _find_pairs(tables: list[dict[str, Any]]) -> list[_Pair]:
    pairs: list[_Pair] = []
    seen: set[frozenset[str]] = set()

    by_stem: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    by_base: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for table in tables:
        schema = table.get("schema_name") or ""
        split = _split_prefix(table["name"])
        if split is not None:
            by_stem[(schema, split[1])].append((split[0], table))
        by_base[(schema, _marker_base(table["name"]))].append(table)

    for group in by_stem.values():
        for (pa, ta), (pb, tb) in combinations(sorted(group, key=lambda g: g[0]), 2):
            if pa == pb:
                continue
            pair = _make_pair(ta, tb, "prefix", (pa, pb))
            if pair is not None:
                seen.add(frozenset((ta["id"], tb["id"])))
                pairs.append(pair)

    for (_, base), group in by_base.items():
        if not base or len(group) < 2:
            continue
        for ta, tb in combinations(sorted(group, key=lambda t: t["name"]), 2):
            key = frozenset((ta["id"], tb["id"]))
            if key in seen:
                continue
            if not any(
                _is_marker(t) for t in _tokens(ta["name"]) + _tokens(tb["name"])
            ):
                continue
            pair = _make_pair(ta, tb, "marker", ())
            if pair is not None:
                seen.add(key)
                pairs.append(pair)
    return pairs


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def _version(name: str) -> int | None:
    versions = [
        int(m.group(1)) for t in _tokens(name) if (m := _VERSION_TOKEN.match(t))
    ]
    return max(versions) if versions else None


def _score_catalog(pair: _Pair) -> None:
    """Evidence the catalog alone can give: names, columns and usage."""
    first_marked = any(t in _LEGACY_MARKERS for t in _tokens(pair.first["name"]))
    second_marked = any(t in _LEGACY_MARKERS for t in _tokens(pair.second["name"]))
    if first_marked != second_marked:
        sign = 1 if first_marked else -1
        marked = pair.first if first_marked else pair.second
        pair.add(sign * _W_NAME, f"{marked['name']} carries a legacy marker")
    else:
        va, vb = _version(pair.first["name"]), _version(pair.second["name"])
        if va is not None and vb is not None and va != vb:
            pair.add(
                _W_NAME if vb > va else -_W_NAME,
                f"version {max(va, vb)} is newer than {min(va, vb)}",
            )

    if pair.second_only and not pair.first_only:
        pair.add(
            _W_COLUMNS,
            f"{pair.second['name']} adds columns {', '.join(pair.second_only)}",
        )
    elif pair.first_only and not pair.second_only:
        pair.add(
            -_W_COLUMNS,
            f"{pair.first['name']} adds columns {', '.join(pair.first_only)}",
        )

    qa, qb = pair.first["query_count"], pair.second["query_count"]
    if qa == 0 and qb > 0:
        pair.add(_W_USAGE, f"only {pair.second['name']} appears in logged queries")
    elif qb == 0 and qa > 0:
        pair.add(-_W_USAGE, f"only {pair.first['name']} appears in logged queries")
    elif qa and qb and max(qa, qb) >= 2 * min(qa, qb):
        busier = pair.second if qb > qa else pair.first
        pair.add(
            (_W_USAGE / 2) * (1 if qb > qa else -1),
            f"{busier['name']} is queried at least twice as often",
        )


def _score_families(pairs: list[_Pair], tables: list[dict[str, Any]]) -> None:
    """Prefix-level evidence, shared by every pair of one prefix family."""
    families: dict[tuple[str, tuple[str, str]], list[_Pair]] = defaultdict(list)
    for pair in pairs:
        if pair.kind == "prefix":
            families[(pair.first.get("schema_name") or "", pair.prefixes)].append(pair)

    for (schema, (pa, pb)), members in families.items():
        in_schema = [t for t in tables if (t.get("schema_name") or "") == schema]
        names = {t["name"].upper() for t in in_schema}
        stems = {_split_prefix(p.first["name"])[1] for p in members}  # type: ignore[index]

        def history(prefix: str) -> int:
            return sum(
                any(f"{prefix}_{stem}_{h}" in names for h in _HISTORY_TOKENS)
                for stem in stems
            )

        def references(prefix: str) -> int:
            lead = f"{prefix}_"
            return sum(
                any(str(c["name"]).upper().startswith(lead) for c in t["columns"])
                for t in in_schema
                if _split_prefix(t["name"]) is None
                or _split_prefix(t["name"])[0] not in (pa, pb)  # type: ignore[index]
            )

        for label, (ca, cb) in (
            ("history tables", (history(pa), history(pb))),
            ("references from other tables", (references(pa), references(pb))),
        ):
            weight = _family_weight(ca, cb)
            if not weight:
                continue
            leader = pb if weight > 0 else pa
            message = f"prefix {leader}_ has more {label} ({pa}_: {ca}, {pb}_: {cb})"
            for pair in members:
                pair.add(weight, message)


def _family_weight(first: int, second: int) -> float:
    if first == second:
        return 0.0
    weight = _W_FAMILY if min(first, second) == 0 else _W_FAMILY / 2
    return weight if second > first else -weight


def _resolve_connector(database_name: str) -> SQLDatabase | None:
    target = database_name.casefold()
    for connector in get_connectors():
        name = getattr(connector, "database_name", None)
        if isinstance(name, str) and name.casefold() == target:
            return connector
    return None


def _scalar(executor: ProbeExecutor, sql: str, purpose: str) -> Any:
    result = executor.run(sql, purpose=purpose)
    rows = result.get("rows") or []
    if not result["ok"] or not rows or not rows[0]:
        return None
    return next(iter(rows[0].values()))


def _freshness_column(pair: _Pair) -> tuple[str, str] | None:
    for key, names in pair.shared.items():
        tokens = set(_tokens(key))
        if tokens & _FRESHNESS_TOKENS and tokens & _DATE_TOKENS:
            return names
    return None


def _key_candidates(pair: _Pair) -> list[tuple[str, str]]:
    pk = {str(c).upper() for c in pair.first.get("pk") or []} | {
        str(c).upper() for c in pair.second.get("pk") or []
    }
    freshness = _freshness_column(pair)

    def rank(key: str) -> int:
        names = pair.shared[key]
        if names[0].upper() in pk or names[1].upper() in pk:
            return 0
        if key.endswith(_KEY_SUFFIXES):
            return 1
        return 3 if key.endswith("_KEY") else 2

    keys = [k for k in pair.shared if pair.shared[k] != freshness]
    return [pair.shared[k] for k in sorted(keys, key=rank)[:_KEY_CANDIDATES]]


def _score_probes(pair: _Pair, executor: ProbeExecutor, connector: SQLDatabase) -> None:
    """Evidence from the live data: rows, freshness and shared key values."""
    dialect = getattr(connector, "dialect", None)
    first = connector.qualify(pair.first.get("schema_name"), pair.first["name"])
    second = connector.qualify(pair.second.get("schema_name"), pair.second["name"])

    na = _scalar(executor, f"SELECT COUNT(*) AS n FROM {first}", "legacy row count")
    nb = _scalar(executor, f"SELECT COUNT(*) AS n FROM {second}", "legacy row count")
    if na is not None and nb is not None:
        na, nb = int(na), int(nb)
        pair.evidence.append(
            f"rows: {pair.first['name']}={na}, {pair.second['name']}={nb}"
        )
        if max(na, nb) > 1.1 * min(na, nb):
            pair.add(
                _W_ROWS if nb > na else -_W_ROWS,
                f"{(pair.second if nb > na else pair.first)['name']} has more rows",
            )

    freshness = _freshness_column(pair)
    if freshness is not None:
        ca = quoted_identifier(freshness[0], dialect)
        cb = quoted_identifier(freshness[1], dialect)
        la = _scalar(
            executor, f"SELECT MAX({ca}) AS v FROM {first}", "legacy freshness"
        )
        lb = _scalar(
            executor, f"SELECT MAX({cb}) AS v FROM {second}", "legacy freshness"
        )
        if la is not None and lb is not None and str(la) != str(lb):
            newer = pair.second if str(lb) > str(la) else pair.first
            pair.add(
                _W_FRESHNESS if newer is pair.second else -_W_FRESHNESS,
                f"{newer['name']} was loaded more recently "
                f"({freshness[0]}: {la} vs {lb})",
            )

    overlaps: list[float] = []
    for col_a, col_b in _key_candidates(pair):
        ca = quoted_identifier(col_a, dialect)
        cb = quoted_identifier(col_b, dialect)
        da = _scalar(
            executor, f"SELECT COUNT(DISTINCT {ca}) AS n FROM {first}", "legacy keys"
        )
        db = _scalar(
            executor, f"SELECT COUNT(DISTINCT {cb}) AS n FROM {second}", "legacy keys"
        )
        if not da or not db:
            continue
        shared = _scalar(
            executor,
            f"SELECT COUNT(DISTINCT {ca}) AS n FROM {first} "
            f"WHERE {ca} IN (SELECT {cb} FROM {second})",
            "legacy key overlap",
        )
        if shared is not None:
            overlaps.append(int(shared) / min(int(da), int(db)))
    if overlaps:
        pair.key_overlap = max(overlaps)
        pair.evidence.append(f"shared key values: {pair.key_overlap:.0%}")


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


def _blocked(pair: _Pair) -> bool:
    return pair.key_overlap is not None and pair.key_overlap < _MIN_KEY_OVERLAP


def _decide(pairs: list[_Pair]) -> dict[str, tuple[str | None, str]]:
    """``legacy table id -> (replacement id, reason)`` for the decided pairs."""
    decided: dict[str, tuple[str | None, str]] = {}
    families: dict[tuple[str, tuple[str, str]], list[_Pair]] = defaultdict(list)
    for pair in pairs:
        if pair.kind == "prefix":
            families[(pair.first.get("schema_name") or "", pair.prefixes)].append(pair)

    family_score = {
        id(p): sum(m.score for m in members) / len(members)
        for members in families.values()
        for p in members
    }

    for pair in pairs:
        if _blocked(pair):
            logger.info(
                "detect_legacy_tables: %s / %s share too few key values to compare",
                pair.first["name"],
                pair.second["name"],
            )
            continue
        score = family_score.get(id(pair), pair.score)
        # A pair whose own evidence clearly disagrees with its family is judged
        # alone rather than overruled.
        if abs(pair.score) >= _DECISIVE_SCORE and pair.score * score < 0:
            score = pair.score
        if abs(score) >= _DECISIVE_SCORE:
            legacy, current = (
                (pair.first, pair.second) if score > 0 else (pair.second, pair.first)
            )
            decided[legacy["id"]] = (current["id"], "; ".join(pair.evidence))
            continue
        judged = _judge_with_llm(pair)
        if judged is None:
            logger.info(
                "detect_legacy_tables: %s / %s left unmarked (score %+.1f)",
                pair.first["name"],
                pair.second["name"],
                score,
            )
            continue
        decided[judged[0]] = (judged[1], judged[2])
    return decided


def _describe(table: dict[str, Any], only: list[str]) -> str:
    return (
        f"table_id={table['id']} name={table['name']} "
        f"columns={len(table['columns'])} logged_query_uses={table['query_count']}\n"
        f"  columns only here: {', '.join(only) or '(none)'}"
    )


def _judge_with_llm(pair: _Pair) -> tuple[str, str, str] | None:
    """``(legacy id, replacement id, reason)``, or ``None`` when undecided."""
    prompt = (
        f"Table A: {_describe(pair.first, pair.first_only)}\n"
        f"Table B: {_describe(pair.second, pair.second_only)}\n"
        f"Shared columns: {len(pair.shared)} "
        f"(name similarity {pair.similarity:.0%})\n"
        "Evidence (a positive weight points to A being outdated):\n"
        + ("\n".join(f"- {e}" for e in pair.evidence) or "- none")
    )
    result = invoke_with_structured_output(
        get_llm_client(temperature=0.0, max_tokens=1024),
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)],
        LegacyTableJudgement,
    )
    if result is None or result.legacy_table_id is None:
        return None
    ids = {pair.first["id"]: pair.second["id"], pair.second["id"]: pair.first["id"]}
    replacement = ids.get(result.legacy_table_id)
    if replacement is None:
        return None
    return result.legacy_table_id, replacement, f"LLM: {result.reason}"


def _generalise(
    decided: dict[str, tuple[str | None, str]],
    pairs: list[_Pair],
    tables: list[dict[str, Any]],
) -> dict[str, tuple[str | None, str]]:
    """Mark the other tables that carry a confirmed legacy prefix or marker."""
    by_id = {t["id"]: t for t in tables}
    current_ids = {replacement for replacement, _ in decided.values() if replacement}
    blocked_ids = {t["id"] for p in pairs if _blocked(p) for t in (p.first, p.second)}

    prefix_votes: dict[tuple[str, str, str], int] = defaultdict(int)
    markers: set[str] = set()
    for legacy_id, (replacement_id, _) in decided.items():
        legacy = by_id[legacy_id]
        current = by_id.get(replacement_id or "")
        if current is None:
            continue
        lp, cp = _split_prefix(legacy["name"]), _split_prefix(current["name"])
        if lp and cp and lp[0] != cp[0] and lp[1] == cp[1]:
            prefix_votes[(legacy.get("schema_name") or "", lp[0], cp[0])] += 1
        markers |= {t for t in _tokens(legacy["name"]) if t in _LEGACY_MARKERS} - set(
            _tokens(current["name"])
        )

    patterns = {
        key: votes
        for key, votes in prefix_votes.items()
        if votes >= _MIN_PREFIX_PATTERN_PAIRS
        # A prefix also judged current against the other way round is no pattern.
        and prefix_votes.get((key[0], key[2], key[1]), 0) == 0
    }

    extra: dict[str, tuple[str | None, str]] = {}
    for table in tables:
        if table["id"] in decided or table["id"] in current_ids:
            continue
        if table["id"] in blocked_ids:
            continue
        schema = table.get("schema_name") or ""
        split = _split_prefix(table["name"])
        if split is not None:
            for (p_schema, legacy_prefix, current_prefix), votes in patterns.items():
                if p_schema != schema or split[0] != legacy_prefix:
                    continue
                counterpart = next(
                    (
                        t
                        for t in tables
                        if (t.get("schema_name") or "") == schema
                        and t["name"].upper() == f"{current_prefix}_{split[1]}"
                    ),
                    None,
                )
                extra[table["id"]] = (
                    counterpart["id"] if counterpart else None,
                    f"carries prefix {legacy_prefix}_, judged legacy against "
                    f"{current_prefix}_ on {votes} table pairs",
                )
                break
        if table["id"] in extra:
            continue
        found = markers & set(_tokens(table["name"]))
        if found:
            extra[table["id"]] = (
                None,
                f"carries legacy marker {', '.join(sorted(found))} "
                "confirmed on another table",
            )
    return extra


def _resolve_chains(
    decided: dict[str, tuple[str | None, str]],
) -> list[dict[str, Any]]:
    """Point every legacy table at a replacement that is not itself legacy."""
    decisions: list[dict[str, Any]] = []
    for legacy_id, (replacement_id, reason) in decided.items():
        seen = {legacy_id}
        while replacement_id in decided and replacement_id not in seen:
            seen.add(replacement_id)
            replacement_id = decided[replacement_id][0]
        if replacement_id in seen:
            # Two tables each judged the other's legacy copy: no answer.
            continue
        decisions.append(
            {
                "table_id": legacy_id,
                "replacement_table_id": replacement_id,
                "reason": reason,
            }
        )
    return decisions
