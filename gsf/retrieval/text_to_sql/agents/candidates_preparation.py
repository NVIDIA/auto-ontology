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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from gsf.dal.attributes import fetch_attr_column_contexts, find_join_path
from gsf.dal.custom_analyses import (
    fetch_custom_analyses_with_sql,
    fetch_tables_from_custom_analyses,
)
from gsf.dal.datasources import fetch_tables_by_ids
from gsf.dal.sql_attributes import (
    fetch_sql_attributes_with_sql,
    fetch_tables_from_sql_attributes,
)
from gsf.dal.terms import fetch_term_synonyms, fetch_term_table_pairs
from gsf.retrieval.data_access.relevant_tables import (
    dedupe_merge_relevant_tables,
    get_relevant_tables,
    get_relevant_tables_from_candidates,
)
from gsf.retrieval.text_to_sql.base import BaseAgent, record_thought
from gsf.retrieval.text_to_sql.formatters_util import qualify_table
from gsf.retrieval.text_to_sql.models import (
    AnchorColumnModel,
    CustomAnalysisRelevanceModel,
    TableRelevanceModel,
)
from gsf.retrieval.text_to_sql.prompts import (
    CUSTOM_ANALYSIS_RELEVANCE_FILTER_PROMPT,
    TABLE_RELEVANCE_FILTER_PROMPT,
)
from gsf.retrieval.text_to_sql.state import (
    AgentState,
    get_question_for_processing,
    rules_to_text,
)
from gsf.utils.llm_invoke import invoke_with_structured_output


def _qualified_name(t: dict) -> str:
    """Build a database/schema-qualified table name for deduplication."""
    return qualify_table(
        t.get("database_name", ""),
        t.get("schema_name", ""),
        t.get("name", ""),
    )


logger = logging.getLogger(__name__)

#: Concurrent join-path lookups. Held below the DAL pool (5 + 5 overflow) so a
#: wide candidate set cannot starve the rest of the request.
_JOIN_PATH_WORKERS = 4

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
        (same per-table dict shape as ``get_relevant_tables``)
    - path_state["relevant_queries"]: Relevant queries for context
    - path_state["similar_questions"]: Similar questions from history
    - path_state["custom_analyses"]: Filtered complex candidates
    - path_state["custom_analyses_str"]: String representation for prompts
    - path_state["sql_attributes"]: Retrieved SqlAttribute details
    - path_state["sql_attributes_str"]: String representation of SqlAttributes for prompts
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

        # Extra table retrieve only needs the question + entities. Kick it
        # off before the anchor LLM so the two ~3s steps overlap.
        extra_entities = path_state.get("entities") or []
        with ThreadPoolExecutor(max_workers=1) as extra_pool:
            extra_future = extra_pool.submit(
                self._retrieve_additional_tables,
                state.get("data_retriever"),
                question,
                extra_entities,
                target_db,
            )

            # --- 2. Enrich ColumnAttributes with the store context and build join paths ---
            primary_attribute: dict | None = None
            attribute_join_paths: list[dict] = []
            attr_contexts: dict[str, dict] = {}
            term_synonyms: dict[str, list[str]] = {}

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
                    "Fetched the store context for %d/%d column attributes",
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
                                    "path": join_path,
                                }
                            )
                            self.logger.info(
                                "Join path to %s (%s): %d hop(s)",
                                dest_ctx["attr_name"],
                                dest_id,
                                len(join_path),
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
                self.logger.warning(
                    "Additional table retrieval failed",
                    exc_info=True,
                )

        seen_qnames: set[str] = set()
        deduped_tables: list[dict] = []
        for t in relevant_tables + additional_tables:
            qn = _qualified_name(t).lower()
            if qn in seen_qnames:
                continue
            seen_qnames.add(qn)
            deduped_tables.append(t)
        relevant_tables = deduped_tables

        self.logger.info(
            "Found %d relevant tables (after dedupe, capped at 20): %s",
            len(relevant_tables),
            [_qualified_name(t) for t in relevant_tables],
        )

        # --- 4b. Add tables referenced by custom analyses via the store ---
        if custom_analyses:
            ca_ids = [str(ca["id"]) for ca in custom_analyses if ca.get("id")]
            ca_linked_tables = fetch_tables_from_custom_analyses(ca_ids)
            existing_ids = {t.get("id") for t in relevant_tables}
            added = 0
            for tbl in ca_linked_tables:
                if tbl.get("id") not in existing_ids:
                    relevant_tables.append(tbl)
                    existing_ids.add(tbl.get("id"))
                    added += 1
            self.logger.info(
                "Added %d table(s) from custom analyses SQL references: %s",
                added,
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
            existing_ids = {t.get("id") for t in relevant_tables}
            added = 0
            for tbl in sa_linked_tables:
                if tbl.get("id") not in existing_ids:
                    relevant_tables.append(tbl)
                    existing_ids.add(tbl.get("id"))
                    added += 1
            self.logger.info(
                "Added %d table(s) from SqlAttribute SQL references: %s",
                added,
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

        if target_db:
            relevant_tables = [
                table
                for table in relevant_tables
                if table.get("database_name") == target_db
            ]

        # --- 5. Filter tables by relevance ---
        relevant_tables, table_relevance_reasoning = self._filter_tables_by_relevance(
            state,
            question,
            relevant_tables,
            custom_analyses,
        )
        self.logger.info(
            "Kept %d relevant tables (after relevance filter): %s",
            len(relevant_tables),
            [_qualified_name(t) for t in relevant_tables],
        )
        if table_relevance_reasoning:
            record_thought(path_state, _GRAPH_NODE_NAME, table_relevance_reasoning)

        return {
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

    def _retrieve_additional_tables(
        self,
        retriever: Any,
        question: str,
        entities: Any,
        target_db: str | None,
    ) -> list[dict]:
        """Embed the question and entities and search for extra Table hits.

        Independent of the anchor LLM — intended to run in parallel with it.
        """
        search_queries = [question]
        if isinstance(entities, list):
            search_queries.extend(entities)
        k_per_query = max(1, 5 // len(search_queries))
        additional_tables: list[dict] = []

        def _fetch_tables_for_query(query: str) -> list[dict]:
            return get_relevant_tables(
                retriever,
                query,
                k=k_per_query,
                database_name=target_db,
            )

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
        return dedupe_merge_relevant_tables(additional_tables)[:10]

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
    ) -> tuple[list[dict], str]:
        """Use the LLM to decide which candidate tables are actually needed."""
        if len(tables) <= 2:
            return tables, ""

        try:
            llm = state["llm"]
        except KeyError:
            self.logger.warning("No LLM in state — skipping relevance filter")
            return tables, ""

        tables_summary = "\n".join(
            f"- {_qualified_name(t)}: {t.get('description', '(no description)')}"
            for t in tables
        )

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

        prompt_text = TABLE_RELEVANCE_FILTER_PROMPT.format(
            question=question,
            tables_summary=tables_summary,
            domain_rules=domain_rules_section,
            custom_analyses=ca_section,
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
        names_to_remove = {name.lower() for name in result.tables_to_remove}

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

        attrs_block = "\n".join(
            f"- id: {aid} | {ctx['attr_name']} "
            f"(table: {ctx.get('table_name', '?')}, column: {ctx.get('col_name', '?')})"
            + (f" — {ctx['attr_description']}" if ctx.get("attr_description") else "")
            for aid, ctx in contexts.items()
        )
        messages = [
            SystemMessage(
                content=(
                    "You are identifying the primary column attribute that is the main focus "
                    "of a user's analytical question."
                )
            ),
            HumanMessage(
                content=(
                    f"Question: {question}\n\n"
                    f"Available column attributes:\n{attrs_block}\n\n"
                    "Return the id of the single column attribute that best represents "
                    "the primary subject of the question."
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
