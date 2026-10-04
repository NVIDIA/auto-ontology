# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Candidate Preparation Agent

This agent prepares and fetches all candidates needed for SQL construction.
It runs before SQL generation agents to gather all necessary context.

Responsibilities:
- Fetch relevant tables from candidates
- Filter tables by LLM-based relevance check
- Retrieve relevant queries for context
- Filter and process complex candidates (custom analyses)
- Store all prepared data in path_state for downstream agents

Design Decisions:
- Runs before SQL generation to separate data fetching from SQL construction logic
- Stores fetched data in path_state for reusability across multiple SQL agents
- Handles embeddings and conversation history lookup
- LLM relevance filter removes noise tables before SQL construction
- Extra table retrieve runs in parallel with the anchor-column LLM
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from auto_ontology.dal.attributes import (
    fetch_attr_column_contexts,
    find_anchor_hub_siblings,
    find_connected_junction_tables,
    find_join_path,
    find_kept_table_bridges,
)
from auto_ontology.dal.custom_analyses import (
    fetch_custom_analyses_with_sql,
    fetch_tables_from_custom_analyses,
)
from auto_ontology.dal.datasources import fetch_tables_by_ids, find_table_id_by_name
from auto_ontology.dal.sql_attributes import (
    fetch_sql_attributes_with_sql,
    fetch_tables_from_sql_attributes,
)
from auto_ontology.dal.terms import fetch_term_synonyms, fetch_term_table_pairs
from auto_ontology.retrieval.data_access.relevant_tables import (
    dedupe_merge_relevant_tables,
    get_relevant_tables,
    get_relevant_tables_from_candidates,
)
from auto_ontology.retrieval.data_access.semantic_search import search_semantic_index
from auto_ontology.retrieval.text_to_sql.base import BaseAgent, record_thought
from auto_ontology.retrieval.text_to_sql.formatters_util import qualify_table
from auto_ontology.retrieval.text_to_sql.models import (
    AnchorColumnModel,
    CustomAnalysisRelevanceModel,
    TableRelevanceModel,
)
from auto_ontology.retrieval.text_to_sql.prompts import (
    CUSTOM_ANALYSIS_RELEVANCE_FILTER_PROMPT,
    SQL_GEN_MAX_ENTITIES,
    TABLE_RELEVANCE_FILTER_PROMPT,
)
from auto_ontology.retrieval.text_to_sql.state import (
    AgentState,
    get_question_for_processing,
    rules_to_text,
)
from auto_ontology.retrieval.text_to_sql.value_anchors import (
    anchor_tables_missing_from_scope,
    split_schema_named_anchors,
)
from auto_ontology.utils.llm_invoke import invoke_with_structured_output


def _qualified_name(t: dict) -> str:
    """Build a database/schema-qualified table name for deduplication.

    Includes database_name (not just schema.name) so two same-named tables
    in different databases (e.g. two "public.orders") don't collide during
    dedup and silently merge into one.

    Deliberately dialect-blind, unlike the prompt formatters: these names are
    identity keys that get matched back against what the relevance filter
    returns, never interpolated into SQL, so only self-consistency matters.
    """
    return qualify_table(
        t.get("database_name", ""),
        t.get("schema_name", ""),
        t.get("name", ""),
    )


# The relevance filter sees each table's columns, with sample values for
# JSONB columns specifically, since their key names alone (e.g. "Res_Scr")
# can decoy-match unrelated tables. Flat columns get their description when
# one is present. JSONB sample values are already stored on the Column row
# and reach here via fetch_tables_by_ids's nested `columns`, so this adds no
# extra DB round trip, only extra prompt tokens.
#
# Truncation is silent and costs the answer when the cut column is the one
# that justified the table: on BIRD dev, 5 wrong drops all named a column
# past position 25 in a 44-column table. Env-overridable because the right
# value is schema-dependent — it has to exceed the widest table the filter
# must reason about, not the median one.
_RELEVANCE_FILTER_MAX_COLS = int(os.environ.get("RELEVANCE_FILTER_MAX_COLS", "25"))

# Off by default. find_anchor_hub_siblings() pulls in tables that share a
# hub with the anchor's own table via FK — see its docstring for why. Opt-in
# via env flag (this deployment's .env sets it to true) so new/other
# deployments aren't defaulted into the extra store + embedding-search cost
# without an explicit choice.
_HUB_SIBLING_EXPANSION_ENABLED = os.environ.get(
    "HUB_SIBLING_EXPANSION_ENABLED", "false"
).strip().lower() in ("1", "true", "yes")

# Pools this size or smaller skip the relevance filter. Default 2 is the
# long-standing behaviour. The filter drops a uniquely-needed table at a
# near-constant rate whatever the pool size (BIRD dev: 2.0% at 2-4 candidates,
# 2.5% at 5-7, 2.2% at 8+), so raising this buys recoveries and perturbations
# in roughly equal measure — 4 was the only value measured to recover more
# than it risks, and every larger bypass lands at the same break-even.
_RELEVANCE_FILTER_BYPASS_MAX_TABLES = int(
    os.environ.get("RELEVANCE_FILTER_BYPASS_MAX_TABLES", "2")
)

# How many siblings per hub survive the cap — see _rank_and_cap_hub_siblings.
# Raised from 5 (find_anchor_hub_siblings' old built-in default) to 6 after
# an audit of 28.8's run: capped-out siblings matched a GT-required table in
# ~30% of truncation events, and re-ranking by embedding score (rather than
# raising the cap alone) recovered most of those within the *same* cap size —
# see _rank_and_cap_hub_siblings' docstring.
_HUB_SIBLING_CAP = int(os.environ.get("HUB_SIBLING_CAP", "6"))

# How many ColumnAttribute hits to pull per entity when scoring hub siblings
# for the rank — generous relative to per-hub sibling counts (median 2, rare
# mega-hubs up to ~20) so a sibling several entities down the ranking still
# has a chance to be seen, without unbounded index-scan cost.
_HUB_SIBLING_RANK_K = 30

#: Concurrent join-path lookups. Held below the DAL pool (5 + 5 overflow) so a
#: wide candidate set cannot starve the rest of the request.
_JOIN_PATH_WORKERS = 4


def _format_relevance_filter_column(c: dict) -> str:
    name = c.get("name", "")
    ctype = c.get("data_type") or "unknown"
    desc = c.get("description")
    sv = c.get("sample_values")
    extra = ""
    # sample_values is already normalized to list[str] | None by
    # fetch_tables_by_ids (via parse_sample_values) — no JSON decoding here.
    if sv and "json" in str(ctype).lower():
        extra = f" | JSONB keys: {', '.join(str(v) for v in sv[:12])}"
    if desc:
        extra = (extra + f" | {desc}") if extra else f" | {desc}"
    return f"    - {name} ({ctype}){extra}"


def _build_relevance_tables_summary(tables: list[dict]) -> str:
    lines = []
    for t in tables:
        lines.append(
            f"- {_qualified_name(t)}: {t.get('description', '(no description)')}"
        )
        cols = t.get("columns") or []
        if cols:
            lines.append("  Columns:")
            lines.extend(
                _format_relevance_filter_column(c)
                for c in cols[:_RELEVANCE_FILTER_MAX_COLS]
            )
    return "\n".join(lines)


def _merge_tables(base: list[dict], additions: list[dict]) -> list[dict]:
    """Merge *additions* into *base*, enriching existing entries.

    Tables already present in *base* (matched by ``id``) are merged via
    :func:`dedupe_merge_relevant_tables` so per-column fields such as
    ``sample_values`` survive even when the candidate row arrived without
    them.  New tables are appended after the original base order.
    """
    if not additions:
        return list(base)
    base_by_id = {str(t.get("id") or ""): i for i, t in enumerate(base)}
    result = list(base)
    for tbl in additions:
        tid = str(tbl.get("id") or "")
        if tid and tid in base_by_id:
            merged = dedupe_merge_relevant_tables([result[base_by_id[tid]], tbl])
            result[base_by_id[tid]] = merged[0]
        else:
            if tid:
                base_by_id[tid] = len(result)
            result.append(tbl)
    return result


def _needs_column_metadata_backfill(table: dict) -> bool:
    """Whether a relevant table lacks samples or nullability metadata."""
    columns = table.get("columns") or []
    has_samples = any(
        isinstance(column, dict) and column.get("sample_values") for column in columns
    )
    has_complete_nullability = bool(columns) and all(
        isinstance(column, dict) and "is_nullable" in column for column in columns
    )
    return not has_samples or not has_complete_nullability


logger = logging.getLogger(__name__)

# Graph node name this agent is registered under in ``text_to_sql_graph.create_graph``
# (NOT ``self.agent_name``, which is a separate internal/logging name) — must match
# so ``stream_agent_response`` can attribute this agent's recorded thoughts to the
# right step event and ``NODE_LABELS`` entry.
_GRAPH_NODE_NAME = "prepare_candidates"


class CandidatePreparationAgent(BaseAgent):
    """
    Agent that prepares and fetches all candidates for SQL construction.

    This agent gathers all necessary context before SQL generation:
    - Relevant tables
    - Relevant queries for context
    - Similar questions from conversation history


    Output:
    - path_state["candidates"]: Flat list of candidate dicts (same as retrieved, enriched)
    - path_state["relevant_tables"]: Deduplicated list of relevant table dicts
        (same per-table dict shape as ``get_relevant_tables``), including the
        tables value anchors point at, which the relevance filter then judges
    - path_state["relevant_queries"]: Relevant queries for context
    - path_state["similar_questions"]: Similar questions from history
    - path_state["custom_analyses"]: Filtered complex candidates
    - path_state["custom_analyses_str"]: String representation for prompts
    - path_state["sql_attributes"]: Retrieved SqlAttribute details
    - path_state["sql_attributes_str"]: String representation of SqlAttributes for prompts
    - value_anchors: Caller-supplied anchors, narrowed to those naming data
        rather than a table/column in scope (only when anchors were supplied)
    """

    def __init__(self):
        super().__init__("candidate_preparation")

    def validate_input(self, state: AgentState) -> bool:
        """Validate that retrieval produced at least one hit."""
        path_state = state.get("path_state", {})
        has_col_attrs = bool(path_state.get("retrieved_column_attributes"))
        has_custom = bool(path_state.get("retrieved_custom_analyses"))
        has_sql_attrs = bool(path_state.get("retrieved_sql_attributes"))
        if not has_col_attrs and not has_custom and not has_sql_attrs:
            connectors = state.get("connectors") or []
            if path_state.get("target_db") or (
                len(connectors) == 1 and getattr(connectors[0], "database_name", None)
            ):
                return True
            self.logger.warning(
                "No candidates for preparation: expected retrieved_column_attributes, "
                "retrieved_custom_analyses, or retrieved_sql_attributes in path_state"
            )
            return False
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Prepare and fetch all candidates for SQL construction.

        Gathers tables, queries, similar questions, and processes complex candidates.

        Args:
            state: Current agent state

        Returns:
            Dictionary with:
            - path_state: Contains all prepared candidate data
        """
        path_state = state.get("path_state", {})
        question = get_question_for_processing(state)
        target_db = path_state.get("target_db")
        if not target_db:
            connectors = state.get("connectors") or []
            if len(connectors) == 1:
                target_db = getattr(connectors[0], "database_name", None)
        custom_analyses = list(path_state.get("retrieved_custom_analyses") or [])
        column_attributes = list(path_state.get("retrieved_column_attributes") or [])
        sql_attributes_raw = list(path_state.get("retrieved_sql_attributes") or [])
        candidates = custom_analyses + column_attributes + sql_attributes_raw

        # --- 1. Custom analyses ---
        self.logger.info("Retrieved %d custom analyses", len(custom_analyses))

        analysis_ids = [str(ca["id"]) for ca in custom_analyses if ca.get("id")]
        relevant_queries = fetch_custom_analyses_with_sql(analysis_ids)
        self.logger.info(
            "Found %d relevant queries from custom analyses", len(relevant_queries)
        )

        custom_analyses_str = self._build_custom_analyses_str(relevant_queries)

        # Extra-table retrieval only needs the question + entities — independent
        # of the anchor-column LLM/join-path work below. Kick it off in the
        # background so the two multi-second steps overlap instead of stacking.
        extra_pool = ThreadPoolExecutor(max_workers=1)
        extra_future = extra_pool.submit(
            self._retrieve_additional_tables,
            state.get("data_retriever"),
            question,
            path_state.get("entities") or [],
            target_db,
        )

        # --- 2. Enrich ColumnAttributes with store context and build join paths ---
        primary_attribute: dict | None = None
        attribute_join_paths: list[dict] = []
        attr_contexts: dict[str, dict] = {}
        term_synonyms: dict[str, list[str]] = {}
        # Table ids to force back into relevant_tables after the relevance
        # filter runs (§5b-§5c), regardless of what it decides — see rationale
        # at the junction, hub-sibling, and pairwise-bridge computations below.
        forced_table_ids: set[str] = set()

        try:
            if column_attributes:
                attr_ids = [
                    str(hit.get("id") or "")
                    for hit in column_attributes
                    if hit.get("id")
                ]
                attr_ids = list(dict.fromkeys(attr_ids))

                attr_contexts = fetch_attr_column_contexts(
                    attr_ids,
                    database_name=target_db,
                )
                self.logger.info(
                    "Fetched store context for %d/%d column attributes",
                    len(attr_contexts),
                    len(attr_ids),
                )
                term_synonyms = fetch_term_synonyms(attr_ids)
                self.logger.info("Fetched synonyms for %d term(s)", len(term_synonyms))

                anchor_id, anchor_reasoning = self._identify_anchor(
                    state, question, attr_contexts
                )
                self.logger.info("Anchor attribute id: %s", anchor_id)
                if anchor_reasoning:
                    record_thought(path_state, _GRAPH_NODE_NAME, anchor_reasoning)

                if anchor_id and anchor_id in attr_contexts:
                    anchor_ctx = attr_contexts[anchor_id]
                    primary_attribute = {
                        "id": anchor_id,
                        "attr_name": anchor_ctx["attr_name"],
                        "col_name": anchor_ctx["col_name"],
                        "table_name": anchor_ctx["table_name"],
                        "schema_name": anchor_ctx["schema_name"],
                        "database_name": anchor_ctx["database_name"],
                        "datatype": anchor_ctx.get("datatype") or "",
                    }

                    dest_items = [
                        (did, dctx)
                        for did, dctx in attr_contexts.items()
                        if did != anchor_id
                    ]
                    # Bounded, not one worker per destination. Each worker runs
                    # find_join_path, which is several sequential checkouts from a
                    # 10-connection pool; a wide fan-out exhausts it, and
                    # find_join_path catches the QueuePool timeout and returns []
                    # -- so the failure shows up as missing joins in the prompt
                    # rather than as an error.
                    workers = min(len(dest_items) or 1, _JOIN_PATH_WORKERS)
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        futures = {
                            pool.submit(
                                find_join_path, anchor_ctx["col_id"], dctx["col_id"]
                            ): (did, dctx)
                            for did, dctx in dest_items
                        }
                        for future in as_completed(futures):
                            dest_id, dest_ctx = futures[future]
                            join_path = future.result()
                            attribute_join_paths.append(
                                {
                                    "id": dest_id,
                                    "attr_name": dest_ctx["attr_name"],
                                    "col_name": dest_ctx["col_name"],
                                    "table_name": dest_ctx["table_name"],
                                    "schema_name": dest_ctx["schema_name"],
                                    "database_name": dest_ctx["database_name"],
                                    "datatype": dest_ctx.get("datatype") or "",
                                    "path": join_path,
                                }
                            )
                            self.logger.info(
                                "Join path to %s (%s): %d hop(s)",
                                dest_ctx["attr_name"],
                                dest_id,
                                len(join_path),
                            )

                    # Looser, discovery-only signal: find_join_path above cannot
                    # reach a sibling table that shares a hub with the anchor
                    # (forward-only, by design — see its docstring). Surface
                    # those siblings (and the hub itself) separately so the
                    # relevance filter doesn't drop a structurally-connected
                    # table it has no other way to recognize. Scoped to the
                    # anchor's own outgoing FKs only. These are also force-kept
                    # in relevant_tables below (§5b-§5c) rather than merely shown to
                    # the relevance filter, since it's unreliable at preserving
                    # structurally-connected tables even when given this info.
                    anchor_table_id = anchor_ctx.get("table_id")
                    if anchor_table_id and _HUB_SIBLING_EXPANSION_ENABLED:
                        # Uncapped here (max_siblings=None) — capping now happens
                        # after ranking, in _rank_and_cap_hub_siblings, instead of
                        # on the DAL's arbitrary return order.
                        hub_sibling_hops, _ = find_anchor_hub_siblings(
                            anchor_table_id, max_siblings=None
                        )
                        hub_sibling_hops, hub_sibling_truncated = (
                            self._rank_and_cap_hub_siblings(
                                state,
                                path_state.get("entities") or [],
                                target_db,
                                hub_sibling_hops,
                                cap=_HUB_SIBLING_CAP,
                            )
                        )
                        if hub_sibling_hops:
                            attribute_join_paths.append({"path": hub_sibling_hops})
                            forced_table_ids.update(
                                h["id"] for h in hub_sibling_hops if h.get("id")
                            )
                            self.logger.info(
                                "Found %d hub-sibling table(s) via anchor's own FK "
                                "(hub included): %s%s",
                                len(hub_sibling_hops),
                                [h["target_table"] for h in hub_sibling_hops],
                                f" ({hub_sibling_truncated} sibling(s) truncated by cap)"
                                if hub_sibling_truncated
                                else "",
                            )
                else:
                    self.logger.warning(
                        "No valid anchor attribute found — skipping join path computation"
                    )

            # --- 4. Retrieve relevant tables ---
            relevant_tables = get_relevant_tables_from_candidates(candidates)

            if attr_contexts:
                ca_table_ids = list(
                    dict.fromkeys(
                        ctx["table_id"]
                        for ctx in attr_contexts.values()
                        if ctx.get("table_id")
                    )
                )
                ca_tables = fetch_tables_by_ids(ca_table_ids)
                existing_ids = {t.get("id") for t in relevant_tables}
                for tbl in ca_tables:
                    if tbl.get("id") not in existing_ids:
                        relevant_tables.append(tbl)
                        existing_ids.add(tbl.get("id"))

            self.logger.info(
                "Tables from candidates: %s", [t["name"] for t in relevant_tables]
            )

            additional_tables: list[dict] = []
            try:
                additional_tables = extra_future.result()
            except Exception:
                self.logger.warning("Additional table retrieval failed", exc_info=True)
        finally:
            extra_pool.shutdown()

        seen_qnames: set[str] = set()
        deduped_tables: list[dict] = []
        for t in relevant_tables + additional_tables:
            qn = _qualified_name(t).lower()
            if qn in seen_qnames:
                continue
            seen_qnames.add(qn)
            deduped_tables.append(t)
        relevant_tables = deduped_tables

        self.logger.debug(
            "Found %d relevant tables (after dedupe): %s",
            len(relevant_tables),
            [_qualified_name(t) for t in relevant_tables],
        )

        # --- 4a. Back-fill rich per-column metadata when it is incomplete ---
        # Tables retrieved from the vector index carry only name/data_type/description;
        # sample_values and is_nullable live in the store's row for that table.
        # Fetch rich rows whenever either signal is missing, then merge per-column
        # so nothing already present is overwritten.
        metadata_incomplete_ids = [
            str(t["id"])
            for t in relevant_tables
            if t.get("id") and _needs_column_metadata_backfill(t)
        ]
        if metadata_incomplete_ids:
            enriched = fetch_tables_by_ids(metadata_incomplete_ids)
            relevant_tables = _merge_tables(relevant_tables, enriched)
            self.logger.info(
                "Back-filled column metadata for %d/%d table(s)",
                len(enriched),
                len(metadata_incomplete_ids),
            )

        # --- 4b. Add tables referenced by custom analyses via the store ---
        if custom_analyses:
            ca_ids = [str(ca["id"]) for ca in custom_analyses if ca.get("id")]
            ca_linked_tables = fetch_tables_from_custom_analyses(ca_ids)
            prev_len = len(relevant_tables)
            relevant_tables = _merge_tables(relevant_tables, ca_linked_tables)
            self.logger.info(
                "Added %d table(s) from custom analyses SQL references: %s",
                len(relevant_tables) - prev_len,
                [t["name"] for t in ca_linked_tables],
            )

        # --- 4c. Enrich SqlAttributes with SQL + term from the store ---
        sql_attributes: list[dict] = []
        if sql_attributes_raw:
            sa_ids = [
                str(hit.get("id") or "") for hit in sql_attributes_raw if hit.get("id")
            ]
            sa_ids = list(dict.fromkeys(sa_ids))
            sql_attributes = fetch_sql_attributes_with_sql(sa_ids)
            self.logger.info(
                "Fetched %d/%d SqlAttribute details from the store",
                len(sql_attributes),
                len(sa_ids),
            )

            sa_linked_tables = fetch_tables_from_sql_attributes(sa_ids)
            prev_len = len(relevant_tables)
            relevant_tables = _merge_tables(relevant_tables, sa_linked_tables)
            self.logger.info(
                "Added %d table(s) from SqlAttribute SQL references: %s",
                len(relevant_tables) - prev_len,
                [t["name"] for t in sa_linked_tables],
            )

        # --- 4d. Add tables linked to the subject Term ---
        subject_term = path_state.get("retrieved_subject_term")
        subject_term_id = (
            str(subject_term.get("id") or "") if isinstance(subject_term, dict) else ""
        )
        if subject_term_id:
            pairs = fetch_term_table_pairs(term_ids=[subject_term_id])
            subject_table_ids = list(
                dict.fromkeys(str(p["table_id"]) for p in pairs if p.get("table_id"))
            )
            subject_tables = fetch_tables_by_ids(subject_table_ids)
            existing_ids = {t.get("id") for t in relevant_tables}
            added = 0
            for tbl in subject_tables:
                if tbl.get("id") not in existing_ids:
                    relevant_tables.append(tbl)
                    existing_ids.add(tbl.get("id"))
                    added += 1
            self.logger.info(
                "Added %d table(s) from subject Term %r: %s",
                added,
                subject_term.get("name") or subject_term_id,
                [t["name"] for t in subject_tables],
            )

        sql_attributes_str = self._build_sql_attributes_str(sql_attributes)

        # --- 4e. Add the tables value anchors point at ---
        # An anchor reports the table its value was matched in, and retrieval
        # reaches that table by its own route or not at all — §6 below can only
        # discard anchors, never act on one. Added here rather than beside §5c's
        # force-include so the relevance filter sees them as candidates: most
        # anchor tables are not wanted, and deciding that is the filter's job.
        incoming_anchors = list(state.get("value_anchors") or [])
        anchor_table_ids = [
            table_id
            for table_id in (
                find_table_id_by_name(name, target_db)
                for name in anchor_tables_missing_from_scope(
                    incoming_anchors, relevant_tables
                )
            )
            if table_id
        ]
        if anchor_table_ids:
            anchor_tables = fetch_tables_by_ids(anchor_table_ids)
            prev_len = len(relevant_tables)
            relevant_tables = _merge_tables(relevant_tables, anchor_tables)
            self.logger.info(
                "Added %d table(s) a value anchor points at: %s",
                len(relevant_tables) - prev_len,
                [t["name"] for t in anchor_tables],
            )

        # Term rows are global (no database_name), so a Term shared across two
        # databases (e.g. the same domain ingested as both a full DB and a
        # "_template" variant) can pull wrong-DB tables into relevant_tables via
        # the subject-Term/candidate expansion above. Drop them before the
        # relevance filter and bridge reconciliation ever see them, rather than
        # relying on those steps to notice a table that doesn't belong.
        if target_db:
            relevant_tables = [
                table
                for table in relevant_tables
                if table.get("database_name") == target_db
            ]

        # Snapshot the candidate pool BEFORE the relevance filter runs. §5c's
        # bridge reconciliation must only ever restore a table that was
        # already a candidate here (and that the filter had a chance to see)
        # — never surface a table the filter was never shown, which would be
        # discovering new information rather than enforcing the filter's own
        # "don't remove a needed bridge" rule.
        pre_filter_candidate_ids = {t["id"] for t in relevant_tables if t.get("id")}

        # --- 5. Filter tables by relevance ---
        relevant_tables, table_relevance_reasoning = self._filter_tables_by_relevance(
            state,
            question,
            relevant_tables,
            custom_analyses,
            attribute_join_paths,
        )
        self.logger.debug(
            "Kept %d relevant tables (after relevance filter): %s",
            len(relevant_tables),
            [_qualified_name(t) for t in relevant_tables],
        )
        if table_relevance_reasoning:
            record_thought(path_state, _GRAPH_NODE_NAME, table_relevance_reasoning)

        # --- 5b. One-hop junction-table expansion ---
        # Junction tables are join structure, not relevance candidates: once
        # the filter has chosen a table, include every catalog-classified
        # junction whose FK columns point into it. This intentionally discovers
        # tables that were not in the pre-filter candidate pool.
        kept_ids = [t["id"] for t in relevant_tables if t.get("id")]
        connected_junctions, connected_junction_paths = find_connected_junction_tables(
            kept_ids
        )
        if connected_junctions:
            forced_table_ids.update(t["id"] for t in connected_junctions)
            self.logger.info(
                "Connected-junction expansion added %d table(s): %s",
                len(connected_junctions),
                [t["name"] for t in connected_junctions],
            )
        if connected_junction_paths:
            attribute_join_paths.extend(
                {"path": hops} for hops in connected_junction_paths
            )
            self.logger.info(
                "Connected-junction expansion added %d one-hop join path(s)",
                len(connected_junction_paths),
            )

        # --- 5c. Deterministic bridge-table reconciliation ---
        # The relevance filter is unreliable at preserving join-chain bridge
        # tables even when its prompt shows it the exact connection, so force
        # these back in by code instead of relying on it. Two sources:
        #   (a) the anchor's hub + capped siblings, already computed above
        #       and collected into forced_table_ids;
        #   (b) any bridge table needed to connect pairs of tables the
        #       filter itself decided to KEEP — this only ever restores
        #       connectivity between tables the filter already judged
        #       relevant, it never second-guesses which tables matter, and
        #       (via pre_filter_candidate_ids) never introduces a table the
        #       filter was never shown in the first place.
        kept_ids = [t["id"] for t in relevant_tables if t.get("id")]
        bridge_tables, bridge_paths, skipped_pairs = find_kept_table_bridges(
            kept_ids, pre_filter_candidate_ids
        )
        if bridge_tables:
            forced_table_ids.update(t["id"] for t in bridge_tables)
            self.logger.info(
                "Pairwise bridge reconciliation added %d table(s) between "
                "kept tables: %s%s",
                len(bridge_tables),
                [t["name"] for t in bridge_tables],
                f" ({skipped_pairs} pair(s) skipped after cap)"
                if skipped_pairs
                else "",
            )
        # A bridge table with no join hops reaching SQL-gen is a table the
        # model can see but not connect — without the real FK chain, it has
        # to guess the join condition and can fabricate one between unrelated
        # columns. Surface the real FK chain the same way attribute_join_paths
        # already does for verified semantic joins.
        if bridge_paths:
            attribute_join_paths.extend({"path": hops} for hops in bridge_paths)
            self.logger.info(
                "Pairwise bridge reconciliation added %d join path(s) for "
                "bridge table(s)",
                len(bridge_paths),
            )

        forced_table_ids -= {t.get("id") for t in relevant_tables}
        if forced_table_ids:
            forced_tables = fetch_tables_by_ids(list(forced_table_ids))
            relevant_tables = _merge_tables(relevant_tables, forced_tables)
            self.logger.info(
                "Force-included %d table(s) after relevance filter (deterministic "
                "reconciliation, not the LLM's choice): %s",
                len(forced_tables),
                [t["name"] for t in forced_tables],
            )

        self.logger.info(
            "Final %d table(s) reaching SQL generation: %s",
            len(relevant_tables),
            [_qualified_name(t) for t in relevant_tables],
        )

        # --- 6. Drop value anchors that name schema rather than data ---
        # Anchors were matched against stored values without the schema, so a
        # phrase belonging to a column name still arrives pointing at whichever
        # unrelated column holds it as data. The tables in scope are only known
        # here, which is what makes the two separable — see value_anchors.py.
        kept_anchors, schema_named_anchors = split_schema_named_anchors(
            incoming_anchors, relevant_tables
        )
        if schema_named_anchors:
            self.logger.info(
                "Dropped %d value anchor(s) naming a table/column in scope "
                "rather than a stored value: %s",
                len(schema_named_anchors),
                [
                    f"{a.get('phrase')!r} -> {a.get('tbl')}.{a.get('col')}"
                    for a in schema_named_anchors
                ],
            )

        result: Dict[str, Any] = {
            "path_state": {
                **path_state,
                "relevant_tables": relevant_tables,
                "relevant_queries": [
                    r["sql"] for r in relevant_queries if r.get("sql")
                ],
                "custom_analyses": custom_analyses,
                "custom_analyses_str": custom_analyses_str,
                "sql_attributes": sql_attributes,
                "sql_attributes_str": sql_attributes_str,
                "table_relevance_reasoning": table_relevance_reasoning,
                "primary_attribute": primary_attribute,
                "attribute_join_paths": attribute_join_paths,
                "term_synonyms": term_synonyms,
            }
        }
        if incoming_anchors:
            result["value_anchors"] = kept_anchors
        return result

    def _rank_and_cap_hub_siblings(
        self,
        state: AgentState,
        entities: list[str],
        target_db: str | None,
        hub_sibling_hops: list[dict],
        cap: int,
    ) -> tuple[list[dict], int]:
        """Rank a hub's sibling tables by embedding similarity to the
        question's extracted entities, then cap — replacing
        ``find_anchor_hub_siblings``'s old behavior of capping in whatever
        arbitrary order the DAL query happened to return.

        That arbitrary order was a real bug, not just a theoretical one: an
        audit of a full eval run found that when a hub had more than the cap
        (siblings dropped), the dropped sibling was the one GT actually
        needed in ~30% of those events — e.g. a "risk_and_moderation" table
        losing out to "monitoring" purely because of the DAL's return order,
        despite both being plausible siblings of the same "accounts" hub.
        Re-ranking by embedding score (rather than just raising the cap)
        recovered most of those within the *same* cap size, since the
        dropped table was usually still a strong match once actually
        compared against the question — it just never got the chance.

        Scoring reuses the exact same vector index and per-entity search
        ``candidate_retrieval`` already runs every turn (``ColumnAttribute``
        label, same retriever) — no LLM call, and no new embedding calls
        beyond what this turn already pays for elsewhere; only the
        request/response size for these specific lookups is new. For each
        sibling table, its score is the *best* (lowest-distance) hit across
        all entities among that table's own ColumnAttributes — i.e. "does
        any extracted entity match any column on this table well" — not a
        weighted blend across entities, to keep this cheap and legible.
        The hub itself is never scored or dropped (see
        ``find_anchor_hub_siblings``'s docstring for why it's always kept).

        Falls back to the prior (unranked, arbitrary-order) cap on any
        failure — retriever missing, search error, etc. — so this can only
        ever do as well as or better than the old behavior, never worse.

        Returns ``(kept_hops, truncated_count)`` in the same shape
        ``find_anchor_hub_siblings`` returns.
        """
        hub_entries = [h for h in hub_sibling_hops if h.get("is_hub")]
        sibling_entries = [h for h in hub_sibling_hops if not h.get("is_hub")]

        def _unranked_fallback() -> tuple[list[dict], int]:
            kept = sibling_entries[:cap]
            return hub_entries + kept, len(sibling_entries) - len(kept)

        if len(sibling_entries) <= cap or not entities:
            return _unranked_fallback()

        retriever = state.get("data_retriever")
        if retriever is None:
            return _unranked_fallback()

        sibling_table_ids = {h["id"] for h in sibling_entries if h.get("id")}
        best_score: dict[str, float] = {}
        try:
            for entity in entities:
                rows = search_semantic_index(
                    retriever,
                    entity,
                    label_filter=["ColumnAttribute"],
                    per_label_k=_HUB_SIBLING_RANK_K,
                    database_name=target_db,
                )
                attr_ids = [r["id"] for r in rows if r.get("id")]
                if not attr_ids:
                    continue
                ctx = fetch_attr_column_contexts(
                    attr_ids,
                    database_name=target_db,
                )
                for r in rows:
                    c = ctx.get(r.get("id"))
                    tid = c.get("table_id") if c else None
                    if tid not in sibling_table_ids:
                        continue
                    score = r.get("score")
                    if score is None:
                        continue
                    if tid not in best_score or score < best_score[tid]:
                        best_score[tid] = score
        except Exception:
            self.logger.warning(
                "Hub-sibling relevance ranking failed — falling back to unranked cap",
                exc_info=True,
            )
            return _unranked_fallback()

        # Siblings with no scored hit at all (never matched any entity)
        # sort last rather than being dropped outright — they still get a
        # chance to fill remaining cap slots after scored ones.
        ranked = sorted(
            sibling_entries,
            key=lambda h: best_score.get(h.get("id"), float("inf")),
        )
        kept = ranked[:cap]
        truncated = len(sibling_entries) - len(kept)
        return hub_entries + kept, truncated

    def _retrieve_additional_tables(
        self,
        retriever: Any,
        question: str,
        entities: list[str],
        target_db: str | None,
    ) -> list[dict]:
        """Embed the question and entities and search for extra Table hits.

        Independent of the anchor-column LLM/join-path work in ``execute`` —
        intended to run in a background thread in parallel with it.
        """
        search_queries = [question] + list(entities)
        k_per_query = max(1, SQL_GEN_MAX_ENTITIES // len(search_queries))

        def _fetch_tables_for_query(query: str) -> list[dict]:
            return get_relevant_tables(
                retriever,
                query,
                k=k_per_query,
                database_name=target_db,
            )

        additional_tables: list[dict] = []
        with ThreadPoolExecutor(max_workers=len(search_queries) or 1) as pool:
            futures = {
                pool.submit(_fetch_tables_for_query, q): q for q in search_queries
            }
            for future in as_completed(futures):
                query = futures[future]
                try:
                    additional_tables.extend(future.result())
                except Exception:
                    self.logger.warning(
                        "Table retrieval failed for query: %s", query, exc_info=True
                    )
        return dedupe_merge_relevant_tables(additional_tables)[:20]

    def _filter_custom_analyses_by_relevance(
        self,
        state: AgentState,
        question: str,
        analyses: list[dict],
    ) -> list[dict]:
        """Use the LLM to decide which retrieved custom analyses are relevant."""
        if len(analyses) <= 1:
            return analyses

        try:
            llm = state["llm"]
        except KeyError:
            self.logger.warning(
                "No LLM in state — skipping custom analysis relevance filter"
            )
            return analyses

        analyses_summary = "\n".join(
            f"- {a.get('name', '(unnamed)')}: "
            f"{(a.get('description') or '(no description)').strip()}"
            f"{('  SQL: ' + a['sql'].strip()) if a.get('sql') else ''}"
            for a in analyses
        )

        prompt_text = CUSTOM_ANALYSIS_RELEVANCE_FILTER_PROMPT.format(
            question=question,
            analyses_summary=analyses_summary,
        )

        messages = [
            SystemMessage(
                content="You are a database domain expert that filters custom analyses."
            ),
            HumanMessage(content=prompt_text),
        ]

        try:
            result = invoke_with_structured_output(
                llm, messages, CustomAnalysisRelevanceModel
            )
        except Exception as e:
            self.logger.warning(
                "Custom analysis relevance LLM call failed: %s — keeping all",
                e,
            )
            return analyses

        if result is None:
            self.logger.warning(
                "Custom analysis relevance filter returned None — keeping all"
            )
            return analyses

        names_to_remove = {name.lower() for name in result.analyses_to_remove}

        filtered = [
            a for a in analyses if (a.get("name") or "").lower() not in names_to_remove
        ]
        removed = [
            a.get("name")
            for a in analyses
            if (a.get("name") or "").lower() in names_to_remove
        ]

        reasoning = (result.reasoning or "").strip()
        self.logger.info(
            "Custom analysis filter reasoning: %s",
            reasoning if reasoning else "(empty)",
        )
        if removed:
            self.logger.info("Custom analysis filter removed: %s", removed)
        self.logger.info(
            "Custom analysis filter kept: %s", [a.get("name") for a in filtered]
        )

        if not filtered:
            self.logger.warning(
                "Custom analysis filter removed ALL analyses — keeping all"
            )
            return analyses

        return filtered

    def _filter_tables_by_relevance(
        self,
        state: AgentState,
        question: str,
        tables: list[dict],
        custom_analyses: list[dict] | None = None,
        attribute_join_paths: list[dict] | None = None,
    ) -> tuple[list[dict], str]:
        """Use the LLM to decide which candidate tables are actually needed."""
        if len(tables) <= _RELEVANCE_FILTER_BYPASS_MAX_TABLES:
            self.logger.info(
                "Relevance filter bypassed: %d candidate table(s) <= bypass max %d",
                len(tables),
                _RELEVANCE_FILTER_BYPASS_MAX_TABLES,
            )
            return tables, ""

        try:
            llm = state["llm"]
        except KeyError:
            self.logger.warning("No LLM in state — skipping relevance filter")
            return tables, ""

        tables_summary = _build_relevance_tables_summary(tables)

        domain_rules_text = rules_to_text(state.get("domain_rules", []))
        domain_rules_section = ""
        if domain_rules_text:
            domain_rules_section = (
                "Domain-specific rules (use these to decide relevance):\n"
                f"{domain_rules_text}\n"
            )

        ca_section = ""
        if custom_analyses:
            ca_lines = []
            for a in custom_analyses:
                line = f"- {a.get('name', '(unnamed)')}"
                desc = (a.get("description") or "").strip()
                if desc:
                    line += f": {desc}"
                sql = (a.get("sql") or "").strip()
                if sql:
                    line += f"  SQL: {sql}"
                ca_lines.append(line)
            ca_section = (
                "Selected custom analyses (their SQL references tables that MUST be kept):\n"
                + "\n".join(ca_lines)
                + "\n\n"
            )

        join_paths_section = ""
        chains: set[str] = set()
        if attribute_join_paths:
            for entry in attribute_join_paths:
                for hop in entry.get("path") or []:
                    src = hop.get("source_table")
                    tgt = hop.get("target_table")
                    if src and tgt and src != tgt:
                        chains.add(f"{src} <-> {tgt}")
            if chains:
                join_paths_section = (
                    "Known join paths between candidate tables:\n"
                    + "\n".join(sorted(chains))
                    + "\n\n"
                )
            self.logger.debug(
                "Relevance filter join_paths_section: %s",
                sorted(chains) if chains else "(no cross-table chains found)",
            )

        # Optional and not specific to any one caller — only present when an
        # upstream flow (e.g. interactive clarification) populated
        # state["enriched_question"]. Deliberately NOT the full
        # state["evidence"] blob: that also carries sort/scalar/output-shape
        # hints and a raw Phase 1 SQL reference, which are noise for a
        # table-relevance decision. Guard sentence lives inside this block
        # (not a standalone Rules bullet) so the prompt is byte-identical to
        # before when no enriched question is present.
        enriched_question_text = (state.get("enriched_question") or "").strip()
        enriched_question_section = ""
        if enriched_question_text:
            enriched_question_section = (
                "The expanded question below is the same question with all formulas, "
                "thresholds, and terms resolved. Use it for context when deciding "
                "which tables are relevant.\n"
                "Expanded question:\n"
                f"{enriched_question_text}\n\n"
            )

        prompt_text = TABLE_RELEVANCE_FILTER_PROMPT.format(
            question=question,
            tables_summary=tables_summary,
            domain_rules=domain_rules_section,
            custom_analyses=ca_section,
            join_paths=join_paths_section,
            enriched_question=enriched_question_section,
        )

        messages = [
            SystemMessage(
                content="You are a database schema expert that filters candidate tables."
            ),
            HumanMessage(content=prompt_text),
        ]

        try:
            result = invoke_with_structured_output(llm, messages, TableRelevanceModel)
        except Exception as e:
            top_n = tables[:10]
            self.logger.warning(
                "Table relevance LLM call failed: %s — falling back to top %d/%d tables",
                e,
                len(top_n),
                len(tables),
            )
            return top_n, ""

        if result is None:
            top_n = tables[:10]
            self.logger.warning(
                "Table relevance filter returned None (LLM parsing failed). "
                "Falling back to top %d/%d tables by retrieval order. "
                "Check ERROR logs above for parsing/validation details.",
                len(top_n),
                len(tables),
            )
            return top_n, ""

        reasoning = (result.reasoning or "").strip()

        # A removal needs both checks to hold, and the model answers them per
        # table (see TableRemovalModel). Enforcing them here rather than
        # trusting the prose is the point of asking: the failure mode being
        # targeted is a model that is confident, not uncertain, so it will
        # state a clean column-provenance case for a table the query needs to
        # restrict rows. Declining such a removal costs one extra table in the
        # prompt; honouring it costs the answer.
        names_to_remove: set[str] = set()
        unjustified: list[str] = []
        for removal in result.tables_to_remove:
            name = (removal.table or "").strip()
            if not name:
                continue
            if removal.supplies_no_needed_column and (
                removal.cannot_change_qualifying_rows
            ):
                names_to_remove.add(name.lower())
            else:
                unjustified.append(
                    f"{name} (no_needed_column="
                    f"{removal.supplies_no_needed_column}, "
                    f"cannot_change_rows={removal.cannot_change_qualifying_rows}: "
                    f"{(removal.justification or '').strip()})"
                )
        if unjustified:
            self.logger.info(
                "Relevance filter proposed %d removal(s) it could not justify on "
                "both checks — keeping them: %s",
                len(unjustified),
                unjustified,
            )

        filtered = [
            t for t in tables if _qualified_name(t).lower() not in names_to_remove
        ]
        removed = [
            _qualified_name(t)
            for t in tables
            if _qualified_name(t).lower() in names_to_remove
        ]

        self.logger.info(
            "Relevance filter reasoning: %s", reasoning if reasoning else "(empty)"
        )
        if removed:
            self.logger.info("Relevance filter removed tables: %s", removed)

        # Cross-reference the join-chain facts we showed the LLM (computed in
        # full before the call, above) against what it actually kept — lets
        # us measure whether the prompt wording is doing anything, rather
        # than assume it from a handful of manually-inspected runs.
        if chains:
            kept_names = {(t.get("name") or "").lower() for t in filtered}
            preserved, broken = [], []
            for chain in sorted(chains):
                a, b = (part.strip().lower() for part in chain.split("<->"))
                (preserved if a in kept_names and b in kept_names else broken).append(
                    chain
                )
            self.logger.debug(
                "Relevance filter join-chain outcome — preserved: %s | broken "
                "(a table on this chain was removed, before §5c reconciliation "
                "restores it): %s",
                preserved if preserved else "(none)",
                broken if broken else "(none)",
            )

        if not filtered:
            self.logger.warning("Relevance filter removed ALL tables — keeping all")
            return tables, reasoning

        return filtered, reasoning

    def _identify_anchor(
        self,
        state: "AgentState",
        question: str,
        contexts: dict[str, dict],
    ) -> tuple[str | None, str]:
        """Use the LLM to pick the primary (anchor) ColumnAttribute for the question.

        Returns ``(anchor_id, reasoning)`` — reasoning is empty when no LLM
        call was needed (0 or 1 candidates) or the call failed.
        """
        ids = list(contexts.keys())
        if not ids:
            return None, ""
        if len(ids) == 1:
            return ids[0], ""

        try:
            llm = state["llm"]
        except KeyError:
            self.logger.warning(
                "_identify_anchor: no LLM in state — using first attribute"
            )
            return ids[0], ""

        def _physical_column(ctx: dict) -> str:
            parts = (
                ctx.get("schema_name"),
                ctx.get("table_name"),
                ctx.get("col_name"),
            )
            return ".".join(str(part) for part in parts if part) or "(unknown)"

        attrs_block = "\n".join(
            f"- id: {aid} | physical column: {_physical_column(ctx)}"
            + (f" — {ctx['attr_description']}" if ctx.get("attr_description") else "")
            for aid, ctx in contexts.items()
        )
        evidence = str(state.get("evidence") or "").strip()
        evidence_block = f"\n\nAuthoritative evidence:\n{evidence}" if evidence else ""
        messages = [
            SystemMessage(
                content=(
                    "You are identifying the primary column attribute that is the main focus "
                    "of a user's analytical question."
                )
            ),
            HumanMessage(
                content=(
                    f"Question: {question}{evidence_block}\n\n"
                    f"Available column attributes:\n{attrs_block}\n\n"
                    "Return the id of the single column attribute that best represents "
                    "the primary subject of the question. Map evidence concepts to "
                    "the listed schema.table.column physical references, not to "
                    "semantic attribute names."
                )
            ),
        ]

        try:
            result = invoke_with_structured_output(
                llm.bind(max_tokens=1024), messages, AnchorColumnModel
            )
        except Exception:
            self.logger.warning(
                "_identify_anchor: LLM call failed — using first attribute",
                exc_info=True,
            )
            return ids[0], ""

        if result and result.anchor_id and result.anchor_id in contexts:
            self.logger.info(
                "Anchor column identified: %s (%s)", result.anchor_id, result.reasoning
            )
            return result.anchor_id, (result.reasoning or "").strip()

        self.logger.warning(
            "_identify_anchor: LLM returned invalid id — using first attribute"
        )
        return ids[0], ""

    def _build_custom_analyses_str(self, relevant_queries: list[dict]) -> list[str]:
        """Build string representation of custom analyses for prompts."""
        parts_list: list[str] = []
        for x in relevant_queries:
            name = (x.get("name") or "").strip()
            if not name:
                continue
            entry = f"name: {name}"
            desc = (x.get("description") or "").strip()
            if desc:
                entry += f", description: {desc}"
            sql = (x.get("sql") or "").strip()
            if sql:
                entry += f", sql: {sql}"
            parts_list.append(entry)
        return parts_list

    def _build_sql_attributes_str(self, sql_attributes: list[dict]) -> list[str]:
        """Build string representation of sql attributes for prompts."""
        parts_list: list[str] = []
        for x in sql_attributes:
            name = (x.get("name") or "").strip()
            if not name:
                continue
            entry = f"name: {name}"
            desc = (x.get("description") or "").strip()
            if desc:
                entry += f", description: {desc}"
            expr = (x.get("expression") or "").strip()
            sql = (x.get("sql") or "").strip()
            if expr:
                entry += f", expression: {expr}"
            if sql and sql != expr:
                entry += f", full_query: {sql}"
            term = (x.get("term_name") or "").strip()
            if term:
                entry += f", term: {term}"
            parts_list.append(entry)
        return parts_list
