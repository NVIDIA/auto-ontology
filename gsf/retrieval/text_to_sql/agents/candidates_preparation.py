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
"""

import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from gsf.neo4j.attributes import fetch_attr_column_contexts, find_join_path
from gsf.neo4j.custom_analyses import (
    fetch_custom_analyses_with_sql,
    fetch_tables_from_custom_analyses,
)
from gsf.neo4j.datasources import fetch_tables_by_ids
from gsf.neo4j.terms import fetch_term_synonyms
from gsf.retrieval.data_access.relevant_tables import (
    dedupe_merge_relevant_tables,
    get_relevant_tables,
    get_relevant_tables_from_candidates,
)
from gsf.retrieval.text_to_sql.base import BaseAgent
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
from gsf.retrieval.text_to_sql.base import BaseAgent
from nemo_retriever.tabular_data.ingestion.model.reserved_words import Edges, Labels
from gsf.retrieval.data_access.graph_schemas import fetch_tables_by_ids
from gsf.retrieval.data_access.relevant_tables import (
    dedupe_merge_relevant_tables,
    get_relevant_tables,
    get_relevant_tables_from_candidates,
)


def _qualified_name(t: dict) -> str:
    """Build schema-qualified table name (e.g. 'public.users') for dedup/filtering."""
    schema = t.get("schema_name", "")
    name = t.get("name", "")
    return f"{schema}.{name}" if schema else name


logger = logging.getLogger(__name__)


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
    """

    def __init__(self):
        super().__init__("candidate_preparation")

    def validate_input(self, state: AgentState) -> bool:
        """Validate that retrieval produced at least one hit."""
        path_state = state.get("path_state", {})
        has_col_attrs = bool(path_state.get("retrieved_column_attributes"))
        has_custom = bool(path_state.get("retrieved_custom_analyses"))
        if not has_col_attrs and not has_custom:
            self.logger.warning(
                "No candidates for preparation: expected retrieved_column_attributes "
                "or retrieved_custom_analyses in path_state"
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
        custom_analyses = list(path_state.get("retrieved_custom_analyses") or [])
        column_attributes = list(path_state.get("retrieved_column_attributes") or [])
        candidates = custom_analyses + column_attributes

        # --- 1. Custom analyses ---
        self.logger.info("Retrieved %d custom analyses", len(custom_analyses))

        analysis_ids = [str(ca["id"]) for ca in custom_analyses if ca.get("id")]
        relevant_queries = fetch_custom_analyses_with_sql(analysis_ids)
        self.logger.info(
            "Found %d relevant queries from custom analyses", len(relevant_queries)
        )

        custom_analyses_str = self._build_custom_analyses_str(relevant_queries)

        # --- 2. Enrich ColumnAttributes with Neo4j context and build join paths ---
        primary_attribute: dict | None = None
        attribute_join_paths: list[dict] = []
        attr_contexts: dict[str, dict] = {}
        term_synonyms: dict[str, list[str]] = {}

        if column_attributes:
            attr_ids = [
                str(hit.get("id") or "") for hit in column_attributes if hit.get("id")
            ]
            attr_ids = list(dict.fromkeys(attr_ids))

            attr_contexts = fetch_attr_column_contexts(attr_ids)
            self.logger.info(
                "Fetched Neo4j context for %d/%d column attributes",
                len(attr_contexts),
                len(attr_ids),
            )
            term_synonyms = fetch_term_synonyms(attr_ids)
            self.logger.info("Fetched synonyms for %d term(s)", len(term_synonyms))

            anchor_id = self._identify_anchor(state, question, attr_contexts)
            self.logger.info("Anchor attribute id: %s", anchor_id)

            if anchor_id and anchor_id in attr_contexts:
                anchor_ctx = attr_contexts[anchor_id]
                primary_attribute = {
                    "id": anchor_id,
                    "attr_name": anchor_ctx["attr_name"],
                    "col_name": anchor_ctx["col_name"],
                    "table_name": anchor_ctx["table_name"],
                    "schema_name": anchor_ctx["schema_name"],
                }

                for dest_id, dest_ctx in attr_contexts.items():
                    if dest_id == anchor_id:
                        continue
                    join_path = find_join_path(anchor_ctx["col_id"], dest_ctx["col_id"])
                    attribute_join_paths.append(
                        {
                            "id": dest_id,
                            "attr_name": dest_ctx["attr_name"],
                            "col_name": dest_ctx["col_name"],
                            "table_name": dest_ctx["table_name"],
                            "schema_name": dest_ctx["schema_name"],
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

        additional_tables = []
        search_queries = [question] + path_state.get("entities", [])
        k_per_query = max(1, 5 // len(search_queries))
        for query in search_queries:
            try:
                tables = get_relevant_tables(
                    state["data_retriever"],
                    query,
                    k=k_per_query,
                )
                additional_tables.extend(tables)
            except Exception:
                self.logger.warning(
                    "Table retrieval failed for query: %s", query, exc_info=True
                )
        additional_tables = dedupe_merge_relevant_tables(additional_tables)[:10]
        relevant_tables.extend(additional_tables)

        self.logger.info(
            "Found %d relevant tables (after dedupe, capped at 20): %s",
            len(relevant_tables),
            [t["name"] for t in relevant_tables],
        )

        # --- 4b. Add tables referenced by custom analyses via Neo4j ---
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
            [t["name"] for t in relevant_tables],
        )

        return {
            "path_state": {
                **path_state,
                "relevant_tables": relevant_tables,
                "relevant_queries": [
                    r["sql"] for r in relevant_queries if r.get("sql")
                ],
                "custom_analyses": custom_analyses,
                "custom_analyses_str": custom_analyses_str,
                "table_relevance_reasoning": table_relevance_reasoning,
                "primary_attribute": primary_attribute,
                "attribute_join_paths": attribute_join_paths,
                "term_synonyms": term_synonyms,
            }
        }

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

    def _fetch_attr_column_contexts(self, attr_ids: list[str]) -> dict[str, dict]:
        """Fetch Column + Table + Schema context for ColumnAttribute IDs from Neo4j.

        Returns a mapping of attr_id -> {attr_name, attr_description, col_id, col_name, table_name, schema_name}.
        """
        if not attr_ids:
            return {}
        query = """
        UNWIND $attr_ids AS attr_id
        MATCH (attr:ColumnAttribute {id: attr_id})
        OPTIONAL MATCH (col:Column)-[:SEMANTIC_FK|HAS_ATTRIBUTE]->(attr)
        OPTIONAL MATCH (col)<-[:CONTAINS]-(tbl:Table)<-[:CONTAINS]-(sch:Schema)
        RETURN attr.id AS attr_id, attr.name AS attr_name, attr.description AS attr_description,
               col.id AS col_id, col.name AS col_name,
               tbl.id AS table_id, tbl.name AS table_name, sch.name AS schema_name
        """
        try:
            rows = get_neo4j_conn().query_read(query, {"attr_ids": attr_ids})
        except Exception:
            self.logger.warning(
                "_fetch_attr_column_contexts: Neo4j query failed", exc_info=True
            )
            return {}
        result: dict[str, dict] = {}
        for row in rows:
            aid = row.get("attr_id")
            if not aid:
                continue
            result[aid] = {
                "attr_name": row.get("attr_name") or "",
                "attr_description": row.get("attr_description") or "",
                "col_id": row.get("col_id"),
                "col_name": row.get("col_name") or "",
                "table_id": row.get("table_id"),
                "table_name": row.get("table_name") or "",
                "schema_name": row.get("schema_name") or "",
            }
        return result

    def _fetch_term_synonyms(self, attr_ids: list[str]) -> dict[str, list[str]]:
        """Fetch synonyms for Terms connected to the given ColumnAttribute IDs.

        Returns a mapping of term_name -> list[synonym].
        """
        if not attr_ids:
            return {}
        query = """
        UNWIND $attr_ids AS attr_id
        MATCH (attr:ColumnAttribute {id: attr_id})-[:PROPERTY_OF]->(term:Term)
        WHERE term.synonyms IS NOT NULL AND size(term.synonyms) > 0
        RETURN DISTINCT term.name AS term_name, term.synonyms AS synonyms
        """
        try:
            rows = get_neo4j_conn().query_read(query, {"attr_ids": attr_ids})
        except Exception:
            self.logger.warning(
                "_fetch_term_synonyms: Neo4j query failed", exc_info=True
            )
            return {}
        result: dict[str, list[str]] = {}
        for row in rows:
            name = row.get("term_name")
            syns = row.get("synonyms") or []
            if name and syns:
                result[name] = [s for s in syns if s]
        return result

    def _fetch_tables_from_custom_analyses(self, analysis_ids: list[str]) -> list[dict]:
        """Fetch Tables referenced by CustomAnalysis nodes via HAS_SQL -> Sql -> SQL -> Table.

        Returns a list of normalized table dicts ready for prompt consumption.
        """
        if not analysis_ids:
            return []
        query = f"""
        UNWIND $ids AS analysis_id
        MATCH (ca:{Labels.CUSTOM_ANALYSIS} {{id: analysis_id}})
              -[:{Edges.HAS_SQL}]->(sql:{Labels.SQL})
              -[:{Edges.SQL}]->(tbl:{Labels.TABLE})
        OPTIONAL MATCH (tbl)<-[:CONTAINS]-(sch:Schema)
        OPTIONAL MATCH (tbl)-[:CONTAINS]->(col:Column)
        WITH tbl, sch, collect({{name: col.name, data_type: col.data_type, description: col.description}}) AS cols
        RETURN tbl.id AS id, tbl.name AS name, tbl.description AS description,
               sch.name AS schema_name, cols
        """
        try:
            rows = get_neo4j_conn().query_read(query, {"ids": analysis_ids})
        except Exception:
            self.logger.warning(
                "_fetch_tables_from_custom_analyses: Neo4j query failed", exc_info=True
            )
            return []
        tables = []
        seen: set[str] = set()
        for row in rows:
            tid = row.get("id")
            if not tid or str(tid) in seen:
                continue
            seen.add(str(tid))
            cols = [c for c in (row.get("cols") or []) if c.get("name")]
            tables.append(
                {
                    "id": tid,
                    "name": row.get("name") or "",
                    "description": row.get("description") or "",
                    "schema_name": row.get("schema_name") or "",
                    "label": Labels.TABLE,
                    "columns": cols,
                }
            )
        return tables

    def _identify_anchor(
        self,
        state: "AgentState",
        question: str,
        contexts: dict[str, dict],
    ) -> str | None:
        """Use the LLM to pick the primary (anchor) ColumnAttribute for the question."""
        ids = list(contexts.keys())
        if not ids:
            return None
        if len(ids) == 1:
            return ids[0]

        try:
            llm = state["llm"]
        except KeyError:
            self.logger.warning(
                "_identify_anchor: no LLM in state — using first attribute"
            )
            return ids[0]

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
            result = invoke_with_structured_output(llm, messages, AnchorColumnModel)
        except Exception:
            self.logger.warning(
                "_identify_anchor: LLM call failed — using first attribute",
                exc_info=True,
            )
            return ids[0]

        if result and result.anchor_id and result.anchor_id in contexts:
            self.logger.info(
                "Anchor column identified: %s (%s)", result.anchor_id, result.reasoning
            )
            return result.anchor_id

        self.logger.warning(
            "_identify_anchor: LLM returned invalid id — using first attribute"
        )
        return ids[0]

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
