# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Match generated SQL joins to ranked semantic path candidates.

This check deliberately fails open.  SQL parsing/catalog probing remains the
authority for structural validity; this module only rejects a resolved,
lower-compatible route when trusted metadata supplies a better exact route.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import sqlglot
from sqlglot import expressions as exp

from auto_ontology.retrieval.text_to_sql.models import (
    SemanticScopeValidationResult,
)

_ENFORCING_TRUST = {"certified", "curated", "reviewed", "high"}
_HIGH_COMPATIBILITY = {"high"}


@dataclass(frozen=True, order=True)
class JoinEndpoint:
    """A normalized qualified column reference."""

    table: tuple[str, ...]
    column: str


JoinHopSignature = tuple[JoinEndpoint, JoinEndpoint]


def _identifier_parts(parts: Iterable[exp.Expression]) -> tuple[str, ...]:
    return tuple(str(part.name).casefold() for part in parts if part.name)


def _signature(left: JoinEndpoint, right: JoinEndpoint) -> JoinHopSignature:
    return tuple(sorted((left, right)))  # type: ignore[return-value]


def _candidate_endpoint(hop: dict[str, Any], side: str) -> JoinEndpoint:
    table = tuple(
        str(hop.get(key) or "").casefold()
        for key in (
            f"{side}_database",
            f"{side}_schema",
            f"{side}_table",
        )
        if hop.get(key)
    )
    return JoinEndpoint(table, str(hop.get(f"{side}_column") or "").casefold())


def candidate_hop_signatures(candidate: dict[str, Any]) -> frozenset[JoinHopSignature]:
    """Return direction-independent exact-hop signatures for one candidate."""
    signatures = []
    for hop in candidate.get("hops") or []:
        left = _candidate_endpoint(hop, "source")
        right = _candidate_endpoint(hop, "target")
        if left.table and left.column and right.table and right.column:
            signatures.append(_signature(left, right))
    return frozenset(signatures)


def _table_compatible(sql_table: tuple[str, ...], expected: tuple[str, ...]) -> bool:
    """Treat differently qualified spellings as equal only by a full suffix."""
    if not sql_table or not expected:
        return False
    shorter = min(len(sql_table), len(expected))
    return sql_table[-shorter:] == expected[-shorter:]


def _hop_compatible(sql_hop: JoinHopSignature, expected: JoinHopSignature) -> bool:
    unmatched = list(expected)
    for actual_endpoint in sql_hop:
        match_index = next(
            (
                index
                for index, expected_endpoint in enumerate(unmatched)
                if actual_endpoint.column == expected_endpoint.column
                and _table_compatible(actual_endpoint.table, expected_endpoint.table)
            ),
            None,
        )
        if match_index is None:
            return False
        unmatched.pop(match_index)
    return not unmatched


def parse_sql_join_signatures(
    sql: str, dialect: str | None = None
) -> tuple[frozenset[JoinHopSignature], bool]:
    """Extract column equalities from JOIN predicates.

    The boolean is false when any join predicate cannot be reduced to qualified
    column equalities.  Callers must classify that case as unknown, never as a
    semantic rejection.
    """
    try:
        query = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return frozenset(), False

    aliases: dict[str, tuple[str, ...]] = {}
    for table in query.find_all(exp.Table):
        parts = _identifier_parts(table.parts)
        if not parts:
            continue
        aliases[table.alias_or_name.casefold()] = parts
        aliases[parts[-1]] = parts

    def endpoint(column: exp.Column) -> JoinEndpoint | None:
        parts = _identifier_parts(column.parts)
        if len(parts) < 2:
            return None
        column_name = parts[-1]
        qualifier = parts[:-1]
        if len(qualifier) == 1:
            resolved = aliases.get(qualifier[0])
            if resolved is None:
                return None
            qualifier = resolved
        return JoinEndpoint(qualifier, column_name)

    signatures: set[JoinHopSignature] = set()
    joins = list(query.find_all(exp.Join))
    for join in joins:
        on = join.args.get("on")
        if on is None:
            return frozenset(signatures), False
        equalities = list(on.find_all(exp.EQ))
        if not equalities:
            return frozenset(signatures), False
        resolved_in_join = 0
        for equality in equalities:
            if not isinstance(equality.left, exp.Column) or not isinstance(
                equality.right, exp.Column
            ):
                continue
            left = endpoint(equality.left)
            right = endpoint(equality.right)
            if left is None or right is None:
                return frozenset(signatures), False
            signatures.add(_signature(left, right))
            resolved_in_join += 1
        if resolved_in_join == 0:
            return frozenset(signatures), False
    return frozenset(signatures), True


def _candidate_is_enforcing(record: dict[str, Any]) -> bool:
    candidate = record.get("candidate") or {}
    hops = candidate.get("hops") or []
    if not hops:
        return False
    explicit = str(
        record.get("metadata_trust") or candidate.get("metadata_trust") or ""
    ).casefold()
    if explicit in _ENFORCING_TRUST:
        return True
    return all(
        str(hop.get("trust") or "").casefold() in _ENFORCING_TRUST for hop in hops
    )


def _confidence(record: dict[str, Any]) -> str:
    return str(
        (record.get("recommendation") or {}).get("confidence") or "unknown"
    ).casefold()


def _exact_joins(record: dict[str, Any]) -> str:
    joins = []
    for hop in (record.get("candidate") or {}).get("hops") or []:
        left = _candidate_endpoint(hop, "source")
        right = _candidate_endpoint(hop, "target")
        joins.append(
            f"{'.'.join((*left.table, left.column))} = "
            f"{'.'.join((*right.table, right.column))}"
        )
    return " AND ".join(joins)


def _scope(record: dict[str, Any]) -> str:
    meaning = (record.get("candidate") or {}).get("meaning") or {}
    return str(meaning.get("qualification_scope") or "unknown")


def validate_semantic_join_scope(
    sql: str,
    join_path_candidates: list[dict[str, Any]],
    *,
    dialect: str | None = None,
) -> SemanticScopeValidationResult:
    """Compare SQL-selected route(s) with phase-4 candidate rankings."""
    if not join_path_candidates:
        return SemanticScopeValidationResult(
            status="unknown_metadata",
            warnings=["No semantic join-path candidates were available."],
        )

    sql_hops, resolved = parse_sql_join_signatures(sql, dialect)
    if not resolved:
        return SemanticScopeValidationResult(
            status="unknown_metadata",
            warnings=[
                "Semantic join scope could not be resolved from SQL; structural validation remains authoritative."
            ],
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(join_path_candidates):
        destination = str(record.get("destination_id") or f"__group_{index}")
        groups[destination].append(record)

    selected: list[dict[str, Any]] = []
    ambiguous = False
    unresolved_groups = 0
    mismatch_alternatives: list[dict[str, Any]] = []
    warning_parts: list[str] = []

    for records in groups.values():
        matches = []
        for record in records:
            expected = candidate_hop_signatures(record.get("candidate") or {})
            if expected and all(
                any(_hop_compatible(actual, hop) for actual in sql_hops)
                for hop in expected
            ):
                matches.append(record)
        # A shorter candidate can be a strict subset of a selected longer
        # route.  Prefer the most complete match; equal-size alternatives
        # remain genuinely ambiguous.
        if matches:
            largest = max(
                len(candidate_hop_signatures(record.get("candidate") or {}))
                for record in matches
            )
            matches = [
                record
                for record in matches
                if len(candidate_hop_signatures(record.get("candidate") or {}))
                == largest
            ]
        if len(matches) > 1:
            ambiguous = True
            selected.extend(matches)
            continue
        if not matches:
            # Candidates for destinations not used by this SQL are normal.
            destination_tables = {
                str(
                    (record.get("destination") or {}).get("table_name") or ""
                ).casefold()
                for record in records
            }
            if destination_tables and not any(
                endpoint.table[-1] in destination_tables
                for hop in sql_hops
                for endpoint in hop
                if endpoint.table
            ):
                continue
            unresolved_groups += 1
            high = [
                record
                for record in records
                if _candidate_is_enforcing(record)
                and _confidence(record) in _HIGH_COMPATIBILITY
            ]
            mismatch_alternatives.extend(high)
            if not high:
                warning_parts.append(
                    "The selected join route did not match available semantic metadata, "
                    "but that metadata was not trusted and high-compatible."
                )
            continue

        chosen = matches[0]
        selected.append(chosen)
        better = [
            record
            for record in records
            if record is not chosen
            and _candidate_is_enforcing(record)
            and _confidence(record) in _HIGH_COMPATIBILITY
            and _confidence(chosen) not in _HIGH_COMPATIBILITY
        ]
        if better:
            mismatch_alternatives.extend(better)
        else:
            if _confidence(chosen) in {"low", "unknown"}:
                warning_parts.append(
                    f"Selected semantic scope '{_scope(chosen)}' has "
                    f"{_confidence(chosen)} question compatibility."
                )
            if not _candidate_is_enforcing(chosen):
                warning_parts.append(
                    f"Selected semantic scope '{_scope(chosen)}' is based on "
                    "suggestion or low-trust metadata."
                )

    selected_fingerprints = list(
        dict.fromkeys(str(record.get("fingerprint") or "") for record in selected)
    )
    alternatives = list(
        {
            str(record.get("fingerprint") or ""): record
            for record in mismatch_alternatives
            if record.get("fingerprint")
        }.values()
    )
    if alternatives:
        selected_scope = ", ".join(dict.fromkeys(_scope(record) for record in selected))
        selected_scope = selected_scope or "unrecognized route"
        rendered = "\n".join(
            f"- scope '{_scope(record)}' ({record.get('fingerprint')}): {_exact_joins(record)}"
            for record in alternatives
        )
        message = (
            f"Semantic scope mismatch: the SQL selected '{selected_scope}', which is "
            "incompatible with the question's trusted path ranking. Reconstruct the "
            "SQL using one of these exact alternatives:\n"
            f"{rendered}\nDo not silently choose an alternative; preserve every option "
            "unless exactly one is listed."
        )
        return SemanticScopeValidationResult(
            status="semantic_mismatch",
            selected_fingerprints=selected_fingerprints,
            alternative_fingerprints=[
                str(record.get("fingerprint")) for record in alternatives
            ],
            message=message,
            enforce_rejection=True,
            auto_repair_fingerprint=(
                str(alternatives[0].get("fingerprint"))
                if len(alternatives) == 1
                else None
            ),
        )

    if ambiguous:
        return SemanticScopeValidationResult(
            status="ambiguous",
            selected_fingerprints=selected_fingerprints,
            warnings=[
                "The SQL matches multiple semantic path fingerprints; all options were preserved."
            ],
        )
    if unresolved_groups or warning_parts:
        return SemanticScopeValidationResult(
            status="unknown_metadata",
            selected_fingerprints=selected_fingerprints,
            warnings=warning_parts
            or ["The SQL route was resolved but did not match a semantic candidate."],
        )
    return SemanticScopeValidationResult(
        status="matched",
        selected_fingerprints=selected_fingerprints,
    )


__all__ = [
    "JoinEndpoint",
    "JoinHopSignature",
    "candidate_hop_signatures",
    "parse_sql_join_signatures",
    "validate_semantic_join_scope",
]
