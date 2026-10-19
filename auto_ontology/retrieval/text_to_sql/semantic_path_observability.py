# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured, fail-open observability for semantic join-path decisions."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

TRACE_KEY = "semantic_join_observability"
TRACE_VERSION = 1


def _trace(path_state: dict[str, Any]) -> dict[str, Any]:
    trace = path_state.setdefault(
        TRACE_KEY,
        {
            "version": TRACE_VERSION,
            "paths": [],
            "sql_selections": [],
            "fallback_reasons": [],
            "counters": {
                "candidate_count": 0,
                "ambiguous_groups": 0,
                "selected_rank": None,
                "interventions": 0,
                "repair_result": "not_attempted",
            },
        },
    )
    return trace


def _metadata_receipt(hop: dict[str, Any]) -> dict[str, Any]:
    metadata = hop.get("relationship_metadata") or {}
    interpretation_source = str(hop.get("interpretation_source") or "unknown")
    interpretation = metadata.get(interpretation_source) or {}
    facts = metadata.get("facts") or {}
    return {
        "source": interpretation_source,
        "version": metadata.get("version"),
        "generator": interpretation.get("generator")
        or facts.get("resolution_generator"),
        "generator_version": interpretation.get("generator_version")
        or facts.get("resolution_version"),
    }


def _structural_source(hop: dict[str, Any]) -> str:
    metadata = hop.get("relationship_metadata") or {}
    facts = metadata.get("facts") or {}
    if facts.get("declared_foreign_key") or facts.get("physical_foreign_key"):
        return "declared_foreign_key"
    return str(facts.get("resolution_source") or "semantic_fk")


def record_ranked_paths(
    path_state: dict[str, Any],
    ranked: list[dict[str, Any]],
    *,
    fallback_reason: str | None = None,
) -> None:
    """Record every candidate and ranking without storing user text or SQL."""
    trace = _trace(path_state)
    paths = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    group_ranks: dict[str, int] = {}
    for record in ranked:
        candidate = record.get("candidate") or {}
        meaning = candidate.get("meaning") or {}
        recommendation = record.get("recommendation") or {}
        hops = candidate.get("hops") or []
        destination_id = str(record.get("destination_id") or "")
        grouped.setdefault(destination_id, []).append(record)
        group_ranks[destination_id] = group_ranks.get(destination_id, 0) + 1
        paths.append(
            {
                "fingerprint": str(record.get("fingerprint") or ""),
                "destination_id": destination_id,
                "hops": [
                    {
                        "fingerprint": str(hop.get("fingerprint") or ""),
                        "source": ".".join(
                            str(hop.get(key) or "")
                            for key in (
                                "source_schema",
                                "source_table",
                                "source_column",
                            )
                            if hop.get(key)
                        ),
                        "target": ".".join(
                            str(hop.get(key) or "")
                            for key in (
                                "target_schema",
                                "target_table",
                                "target_column",
                            )
                            if hop.get(key)
                        ),
                        "structural_source": _structural_source(hop),
                        "direction": hop.get("direction"),
                        "semantic_signature": {
                            "kind": hop.get("relationship_kind"),
                            "source_role": hop.get("source_role"),
                            "target_role": hop.get("target_role"),
                        },
                        "trust": hop.get("trust"),
                        "metadata": _metadata_receipt(hop),
                    }
                    for hop in hops
                ],
                "scope": meaning.get("qualification_scope"),
                "grain": meaning.get("resulting_grain"),
                "fan_out": meaning.get("fan_out"),
                "ranking": {
                    "rank": group_ranks[destination_id],
                    "category": recommendation.get("confidence", "unknown"),
                    "rationale": recommendation.get("rationale", ""),
                },
            }
        )
    trace["paths"] = paths
    counters = trace["counters"]
    counters["candidate_count"] = len(paths)
    counters["ambiguous_groups"] = sum(
        1 for records in grouped.values() if len(records) > 1
    )
    if fallback_reason:
        record_fallback(path_state, fallback_reason)
    _emit(path_state, "ranked")


def sql_fingerprint(sql: str) -> str:
    """Return a stable, non-reversible fingerprint of SQL structure text."""
    normalized = " ".join(sql.split()).casefold()
    return f"sql_{hashlib.sha256(normalized.encode()).hexdigest()[:24]}"


def record_generated_sql(path_state: dict[str, Any], sql: str) -> None:
    """Record a generated SQL fingerprint before optional validation."""
    trace = _trace(path_state)
    fingerprint = sql_fingerprint(sql)
    trace["sql_selections"].append(
        {
            "sql_fingerprint": fingerprint,
            "selected_path_fingerprints": [],
            "selected_rank": None,
            "validator_outcome": "not_run",
            "enforcement_enabled": True,
            "would_reject": False,
            "enforced": False,
        }
    )
    _emit(path_state, "generated")


def record_validator_outcome(
    path_state: dict[str, Any],
    sql: str,
    result: Any,
) -> None:
    """Record SQL/path matching and intervention counters."""
    trace = _trace(path_state)
    selected = list(getattr(result, "selected_fingerprints", []) or [])
    ranks = {
        path["fingerprint"]: path["ranking"]["rank"] for path in trace.get("paths", [])
    }
    selected_ranks = [ranks[value] for value in selected if value in ranks]
    selection = {
        "sql_fingerprint": sql_fingerprint(sql),
        "selected_path_fingerprints": selected,
        "selected_rank": min(selected_ranks) if selected_ranks else None,
        "validator_outcome": getattr(result, "status", "unknown_metadata"),
        "enforcement_enabled": True,
        "would_reject": bool(getattr(result, "enforce_rejection", False)),
        "enforced": bool(getattr(result, "enforce_rejection", False)),
    }
    prior = trace["sql_selections"][-1:] or [{}]
    if (
        prior[0].get("sql_fingerprint") == selection["sql_fingerprint"]
        and prior[0].get("validator_outcome") == "not_run"
    ):
        trace["sql_selections"][-1] = selection
    else:
        trace["sql_selections"].append(selection)
    counters = trace["counters"]
    counters["selected_rank"] = min(selected_ranks) if selected_ranks else None
    if getattr(result, "enforce_rejection", False):
        counters["interventions"] += 1
        counters["repair_result"] = "pending"
    elif counters.get("repair_result") == "pending":
        counters["repair_result"] = (
            "matched" if getattr(result, "status", "") == "matched" else "unresolved"
        )
    _emit(path_state, "validated")


def record_fallback(path_state: dict[str, Any], reason: str) -> None:
    """Record a bounded fallback reason."""
    trace = _trace(path_state)
    reasons = trace["fallback_reasons"]
    if reason and reason not in reasons:
        reasons.append(reason)
    _emit(path_state, "fallback")


def debug_summary(path_state: dict[str, Any]) -> str:
    """Return a concise operator/debug thought derived from structured state."""
    counters = _trace(path_state)["counters"]
    return (
        "Semantic join paths: "
        f"{counters['candidate_count']} candidates, "
        f"{counters['ambiguous_groups']} ambiguous groups, "
        f"selected rank {counters['selected_rank']}, "
        f"{counters['interventions']} interventions, "
        f"repair {counters['repair_result']}."
    )


def _emit(path_state: dict[str, Any], event: str) -> None:
    """Emit the aggregate trace without ever affecting the request."""
    try:
        logger.info(
            "semantic_join.%s %s",
            event,
            json.dumps(_trace(path_state), sort_keys=True, default=str),
        )
    except Exception:
        logger.debug("semantic join trace could not be emitted", exc_info=True)


__all__ = [
    "TRACE_KEY",
    "debug_summary",
    "record_fallback",
    "record_generated_sql",
    "record_ranked_paths",
    "record_validator_outcome",
    "sql_fingerprint",
]
