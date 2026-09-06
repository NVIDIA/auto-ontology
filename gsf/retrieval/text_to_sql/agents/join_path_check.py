# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proactive (pre-execution) join-path check.

Sibling of ``jsonb_path_check.py``, checking ``JOIN ... ON`` predicates
against the same ``SEMANTIC_FK`` graph that seeds ``attribute_join_paths``,
instead of checking JSONB key paths. A hallucinated join between two
plausible-looking identifier columns is syntactically and semantically valid SQL — it executes
and returns rows — so nothing else in the pipeline catches it: syntax
validation only checks the SQL parses, and ``validate_intent`` reasons over
the same join-path data the generator had, so it shares the same blind spot
when that data is what led the model astray in the first place.

Also runs ``find_case_dirty_join_mismatches``: same graph, opposite failure
shape — not a fabricated join, but a *correct* join with an extra, unverified
equality predicate tacked on that silently rejects real matches purely
because the extra column disagrees in casing/whitespace between the two
tables. Unlike a fabricated join, the fix is never to drop the extra
predicate (it may be doing real disambiguation work) — only to normalize it,
and only once the probe has confirmed every rejected row agrees once
normalized. Both checks share this node/flag since they're two sides of the
same "trust but verify JOIN predicates against the graph" idea.

Gated by ``DB_PROBE_JOIN_PATH_CHECK`` since, when the graph has no edge for a
predicate at all, this falls back to a live value-overlap probe (a couple of
extra DB round-trips). Runs at most ``_MAX_REPAIR_ATTEMPTS`` times per
request — unlike the JSONB check's single-shot guard, a join fix sometimes
needs a second pass (the model can apply a partial fix, e.g. correcting one
predicate but not the table set), but an unbounded loop risks repeating the
same wrong join forever if reconstruction doesn't converge.
"""

from __future__ import annotations

from typing import Any, Dict

from gsf.dal.datasources import fetch_tables_by_ids, find_table_id_by_name
from gsf.retrieval.text_to_sql.agents.empty_like_result_check import (
    _get_sql_code,
    _set_sql_code,
)
from gsf.retrieval.text_to_sql.agents.sql_parse_validation import (
    detect_degenerate_sql,
    detect_missing_aggregation,
    detect_vacuous_group_by,
    quote_known_mixed_case_identifiers,
)
from gsf.retrieval.text_to_sql.base import BaseAgent
from gsf.retrieval.text_to_sql.connector_routing import resolve_connector_from_tables
from gsf.retrieval.text_to_sql.db_probe.executor import ProbeExecutor
from gsf.retrieval.text_to_sql.db_probe.join_path_check import (
    build_case_dirty_join_repair_error,
    build_join_path_repair_error,
    find_case_dirty_join_mismatches,
    find_join_path_mismatches,
    try_self_apply_missing_bridge_fixes,
    try_self_apply_wrong_column_fixes,
)
from gsf.retrieval.text_to_sql.state import AgentState

_MAX_REPAIR_ATTEMPTS = 2


class JoinPathCheckAgent(BaseAgent):
    """Check JOIN predicates against the semantic FK graph before executing."""

    def __init__(self) -> None:
        super().__init__("join_path_check")

    def validate_input(self, state: AgentState) -> bool:
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        path_state = dict(state.get("path_state", {}))
        sql_code = _get_sql_code(path_state)

        attempts = path_state.get("join_path_repair_attempts", 0)
        if not sql_code.strip() or attempts >= _MAX_REPAIR_ATTEMPTS:
            return {"decision": "valid_sql", "path_state": path_state}

        connectors = state.get("connectors") or []
        relevant_tables = list(path_state.get("relevant_tables") or [])
        connector = resolve_connector_from_tables(relevant_tables, connectors)
        dialect = getattr(connector, "dialect", None)
        database_name = path_state.get("target_db")

        with ProbeExecutor(connector) as executor:
            mismatches = find_join_path_mismatches(
                executor, dialect, sql_code, database_name
            )
            case_dirty = find_case_dirty_join_mismatches(
                executor, dialect, sql_code, database_name
            )

        if not mismatches and not case_dirty:
            return {"decision": "valid_sql", "path_state": path_state}

        # Self-apply "wrong_column" mismatches by patching the predicate in
        # place — same two tables/aliases, just the wrong column on one or
        # both sides, so it's a safe local edit (see
        # try_self_apply_wrong_column_fixes for why "missing_bridge" and
        # "unverified" are deliberately excluded — no fixed graph edge to
        # inject a whole new JOIN clause from, or no fix to apply at all).
        # Not counted against join_path_repair_attempts below: a fully
        # self-applied fix isn't a reconstruction round spent.
        sql_code, mismatches = try_self_apply_wrong_column_fixes(
            mismatches, sql_code, dialect
        )
        if sql_code != _get_sql_code(path_state):
            self._write_sql_code(path_state, sql_code)
            self.logger.info(
                "[%s] Join path check — self-applied wrong_column fix(es), "
                "%d join(s) still need reconstruction",
                path_state.get("task_id", "?"),
                len(mismatches),
            )

        # Also self-apply the narrow, unambiguous subclass of
        # "missing_bridge" mismatches (see try_self_apply_missing_bridge_fixes
        # for exactly which shape qualifies). Unlike the wrong_column swap
        # above, this inserts a brand-new JOIN clause, which is graph-legal
        # by construction but can still fan out into an unintended
        # many-to-many pairing if both sides have multiple rows per hub row
        # — so the result is checked with the same cheap, LLM-free static
        # checks validate_sql_query already runs (degenerate SQL, vacuous
        # GROUP BY, missing aggregation) before being accepted. No DB
        # round-trip, no LLM call. Any mismatch that fails this check (or
        # doesn't qualify for self-apply at all) falls straight back to the
        # normal reconstruction path below, exactly as if this had never run.
        if any(m["verdict"] == "missing_bridge" for m in mismatches):
            candidate_sql, candidate_mismatches = try_self_apply_missing_bridge_fixes(
                mismatches, sql_code, dialect
            )
            if candidate_sql != sql_code:
                reason = (
                    detect_degenerate_sql(candidate_sql, dialect)
                    or detect_vacuous_group_by(candidate_sql, dialect, database_name)
                    or detect_missing_aggregation(candidate_sql, dialect, database_name)
                )
                if reason:
                    self.logger.info(
                        "[%s] Join path check — self-applied bridge fix "
                        "rejected by post-apply check (%s), leaving it for "
                        "reconstruction instead",
                        path_state.get("task_id", "?"),
                        reason[:150],
                    )
                else:
                    sql_code = candidate_sql
                    mismatches = candidate_mismatches
                    self._write_sql_code(path_state, sql_code)
                    self.logger.info(
                        "[%s] Join path check — self-applied missing_bridge "
                        "fix(es), %d join(s) still need reconstruction",
                        path_state.get("task_id", "?"),
                        len(mismatches),
                    )

        if not mismatches and not case_dirty:
            # A self-applied fix above (wrong_column or missing_bridge) can
            # substitute in a column whose quoting need differs from the one
            # it replaced. Re-run quote_known_mixed_case_identifiers .
            requoted_sql = quote_known_mixed_case_identifiers(
                sql_code, relevant_tables, dialect
            )
            if requoted_sql != sql_code:
                self._write_sql_code(path_state, requoted_sql)
                self.logger.info(
                    "[%s] Join path check — quoted mixed-case identifier(s) "
                    "introduced by self-repair",
                    path_state.get("task_id", "?"),
                )
            return {"decision": "valid_sql", "path_state": path_state}

        path_state["join_path_repair_attempts"] = attempts + 1
        error_sections: list[str] = []
        # Vacuously true when there are no fabricated-join mismatches at all;
        # narrowed below when there are — case_dirty never narrows it, since
        # both branches of this check hand reconstruction an already-known
        # fix, so error_known_fixable only turns false when something here
        # couldn't be resolved from the graph/probe alone.
        known_fixable = True

        if mismatches:
            repairable = [m for m in mismatches if m["verdict"] != "unverified"]
            if repairable:
                # We already know the exact fix from the graph (either the
                # real column pair, or the bridge table(s) plus the join keys
                # through them) — merge it in ourselves rather than making
                # reconstruction re-discover it via its own LLM-driven search.
                new_hops = [m["hops"] for m in repairable]
                attribute_join_paths = list(
                    path_state.get("attribute_join_paths") or []
                )
                attribute_join_paths.extend({"path": hops} for hops in new_hops)
                path_state["attribute_join_paths"] = attribute_join_paths

                known_names = {(t.get("name") or "").lower() for t in relevant_tables}
                missing_names = {
                    name
                    for m in repairable
                    for name in m["bridge_tables"]
                    if name.lower() not in known_names
                }
                self._merge_missing_bridge_tables(
                    path_state,
                    relevant_tables,
                    repairable,
                    missing_names,
                    database_name,
                )

            self.logger.info(
                "[%s] Join path check — routing to reconstruction to fix %d join(s): %s",
                path_state.get("task_id", "?"),
                len(mismatches),
                [
                    f"{m['table_a']}.{m['col_a']} = {m['table_b']}.{m['col_b']} "
                    f"({m['verdict']})"
                    for m in mismatches
                ],
            )
            error_sections.append(build_join_path_repair_error(mismatches))
            # "unverified" mismatches have no known repair — leave
            # known_fixable false so reconstruction's normal LLM
            # error-classification (and its own VDB table-discovery
            # fallback) gets a chance to find something this check couldn't.
            known_fixable = known_fixable and len(repairable) == len(mismatches)

        if case_dirty:
            self.logger.info(
                "[%s] Join path check — routing to reconstruction to normalize %d "
                "case-dirty join predicate(s): %s",
                path_state.get("task_id", "?"),
                len(case_dirty),
                [
                    f"{m['table_a']}/{m['table_b']} extra={m['extra_columns']} "
                    f"({m['rejected_rows']} row(s) rejected, casing-only)"
                    for m in case_dirty
                ],
            )
            error_sections.append(build_case_dirty_join_repair_error(case_dirty))
            # Always a concretely-known fix (normalize the named predicate) —
            # doesn't narrow known_fixable.

        path_state["error"] = "\n\n".join(error_sections)
        if known_fixable:
            path_state["error_known_fixable"] = True

        return {"decision": "invalid_sql", "path_state": path_state}

    @staticmethod
    def _write_sql_code(path_state: Dict[str, Any], sql_code: str) -> None:
        """Store a self-applied *sql_code* — see ``_set_sql_code`` for why
        this must update every field a downstream consumer might read it
        from, not just one."""
        _set_sql_code(path_state, sql_code)

    @staticmethod
    def _merge_missing_bridge_tables(
        path_state: Dict[str, Any],
        relevant_tables: list[dict],
        repairable: list[dict],
        missing_names: set[str],
        database_name: str | None,
    ) -> None:
        """Fetch and merge bridge tables named in *repairable* but not yet in relevant_tables."""
        if not missing_names:
            path_state["relevant_tables"] = relevant_tables
            return

        # find_join_path's hop dicts carry table *names* only, not ids, so
        # resolve each bridge table name back to an id — scoped to
        # database_name to avoid matching a same-named table in a different
        # co-resident database (see find_table_id_by_name).
        ids: list[str] = []
        for name in missing_names:
            table_id = find_table_id_by_name(name, database_name)
            if table_id:
                ids.append(table_id)

        if not ids:
            path_state["relevant_tables"] = relevant_tables
            return

        bridge_tables = fetch_tables_by_ids(ids)
        existing_ids = {t.get("id") for t in relevant_tables}
        for tbl in bridge_tables:
            if tbl.get("id") not in existing_ids:
                relevant_tables.append(tbl)
                existing_ids.add(tbl.get("id"))
        path_state["relevant_tables"] = relevant_tables


__all__ = ["JoinPathCheckAgent"]
