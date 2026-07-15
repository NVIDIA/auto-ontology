# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SQL generation from semantic retrieval context.

Builds SQL from graph-backed semantic candidates (custom analyses, columns),
prepared tables, and optional file extraction — not from ad-hoc “snippet”
assembly alone.

Responsibilities:
- Construct SQL using semantic candidates and schema context from CandidatePreparationAgent
- Handle file extraction results (data_for_sql) when present
- Incorporate similar questions from conversation history
- Handle feedback scenarios
- Store SQL response with custom analyses in path_state

Design Decisions:
- Primary path: vector/semantic retrieval + preparation, then LLM SQL synthesis
- Supports text-style answers when the model returns prose instead of SQL
- Optional extracted file data from upstream file steps
"""

import logging
from typing import Any, Dict
from langchain_core.messages import AIMessage, SystemMessage

from gsf.utils.llm_invoke import safe_invoke_with_structured_output
from gsf.retrieval.text_to_sql.base import BaseAgent
from gsf.retrieval.text_to_sql.connector_routing import resolve_connector_from_tables
from gsf.retrieval.data_access.custom_analyses import (
    build_custom_analyses_section,
    get_custom_analyses_ids,
)
from gsf.retrieval.text_to_sql.state import (
    AgentState,
    get_original_question,
    get_question_for_processing,
)
from gsf.retrieval.text_to_sql.prompts import (
    create_sql_from_candidates_prompt,
    create_sql_user_prompt,
    format_dual_question_block,
)
from gsf.retrieval.text_to_sql.models import SQLGenerationModel

logger = logging.getLogger(__name__)


def _hop_column(hop: dict, side: str) -> str:
    """Format a hop endpoint (``side`` is ``"source"`` or ``"target"``) as
    ``schema.table.column`` (or ``table.column`` when the schema is absent)."""
    schema = hop.get(f"{side}_schema", "")
    table = hop.get(f"{side}_table", "")
    column = hop.get(f"{side}_column", "")
    prefix = f"{schema}.{table}" if schema else table
    return f"{prefix}.{column}"


def _format_semantic_context(
    primary_attribute: dict,
    attribute_join_paths: list[dict],
) -> str:
    """Format the semantic anchor + join-path context for the SQL prompt.

    Produces a human-readable block describing the anchor table/column and
    how to reach every other retrieved column via JOIN conditions derived
    from the semantic graph.

    Example output::

        ANCHOR TABLE (primary focus of the question):
          Table: public.orders
          Column: creator_id  (Creator)

        RELATED COLUMNS (accessible via semantic joins):
          User Name: public.users.name
            Join path from anchor:
              public.orders.creator_id = public.users.id
    """
    anchor_schema = primary_attribute.get("schema_name", "")
    anchor_table = primary_attribute.get("table_name", "")
    anchor_col = primary_attribute.get("col_name", "")
    anchor_name = primary_attribute.get("attr_name", "")
    anchor_full = f"{anchor_schema}.{anchor_table}" if anchor_schema else anchor_table

    lines: list[str] = [
        "SEMANTIC HINT — likely starting table (use as a strong hint, not a mandate):",
        f"  Table: {anchor_full}",
        f"  Column: {anchor_col}  ({anchor_name})",
    ]

    if attribute_join_paths:
        lines.append("")
        lines.append(
            "JOIN PATHS (AUTHORITATIVE — derived from the verified semantic model). "
            "This is our most reliable knowledge of how these tables join: use these "
            "exact join conditions almost always, and only deviate if they clearly "
            "cannot answer the question. Use only the hops you need:"
        )
        for entry in attribute_join_paths:
            attr_name = entry.get("attr_name", "")
            col_name = entry.get("col_name", "")
            schema = entry.get("schema_name", "")
            table = entry.get("table_name", "")
            full_table = f"{schema}.{table}" if schema else table
            lines.append(f"  {attr_name}: {full_table}.{col_name}")
            path = entry.get("path") or []
            if path:
                lines.append("    Join path:")
                # The anchor column is the first hop's source; the destination
                # is the last hop's target. Within a hop, source/target are the
                # same table (navigation), so the actual cross-table joins are
                # between consecutive hops: target[i] = source[i+1].
                if len(path) == 1:
                    left = _hop_column(path[0], "source")
                    right = _hop_column(path[0], "target")
                    lines.append(f"      {left} = {right}")
                else:
                    for cur, nxt in zip(path, path[1:]):
                        left = _hop_column(cur, "target")
                        right = _hop_column(nxt, "source")
                        lines.append(f"      {left} = {right}")

    return "\n".join(lines)


def format_tables_for_prompt(tables: list[dict]) -> str:
    """
    Format tables with clear column information to prevent cross-table column confusion.

    Args:
        tables: Table dicts from ``path_state["relevant_tables"]`` — each must expose
            ``columns`` as a list of dicts (from ``_normalize_table_to_relevant_shape`` / prep).

    Returns:
        Formatted string clearly showing which columns belong to each table
    """
    if not tables:
        return "No tables available"

    formatted_tables = []
    for table in tables:
        table_parts = []

        # Table identifier
        table_name = table.get("name", "UNKNOWN")
        table_label = table.get("label", "")
        table_description = table.get("description", "")

        # Database and schema info
        database_name = table.get("database_name", "")
        schema_name = table.get("schema_name", "")

        # Build table header
        if database_name and schema_name:
            full_name = f"{database_name}.{schema_name}.{table_name}"
        elif schema_name:
            full_name = f"{schema_name}.{table_name}"
        else:
            full_name = table_name

        table_parts.append(f"TABLE: {full_name}")
        if table_label and table_label != table_name:
            table_parts.append(f"  Label: {table_label}")
        if table_description:
            table_parts.append(f"  Description: {table_description}")

        # Primary key
        if "primary_key" in table:
            table_parts.append(f"  Primary Key: {table['primary_key']}")

        columns = table.get("columns")
        if not isinstance(columns, list):
            columns = []
        if columns:
            table_parts.append(
                "  AVAILABLE COLUMNS (only use these columns for this table):"
            )
            for col in columns:
                # Handle both dict and string column formats
                if isinstance(col, dict):
                    col_name = col.get("name", "UNKNOWN")
                    col_type = col.get("data_type", "UNKNOWN")
                    col_desc = col.get("description", "")
                    sample_values = col.get("sample_values")

                    col_line = f"    - {col_name} ({col_type})"
                    if col_desc:
                        col_line += f" - {col_desc}"
                    if sample_values:
                        col_line += f" | sample values: {sample_values}"
                    table_parts.append(col_line)
                elif isinstance(col, str):
                    # If column is a string, use it directly
                    table_parts.append(f"    - {col}")
                else:
                    # Unknown format, convert to string
                    table_parts.append(f"    - {str(col)}")

        formatted_tables.append("\n".join(table_parts))

    return "\n\n".join(formatted_tables)


class SQLFromCandidatesAgent(BaseAgent):
    """
    Agent that constructs SQL from semantic retrieval and prepared schema context.

    Uses candidates, table groups, and related signals produced by
    CandidatePreparationAgent, then prompts the LLM to produce SQL

    Input Requirements:
    - path_state["retrieved_candidates"]: Candidate dicts from preparation
    - path_state["relevant_tables"]: schema context
    - path_state["relevant_queries"]: Relevant queries (from CandidatePreparationAgent)

    Output:
    - path_state["sql_generation_result"]: SQL response with SQL code or text answer
    - path_state["relevant_tables"]: Relevant tables used
    - path_state["custom_analyses_used"]: Semantic entity IDs used
    - decision: "constructable" or "unconstructable"
    """

    def __init__(self):
        super().__init__("sql_from_semantic")

    def validate_input(self, state: AgentState) -> bool:
        """Validate that CandidatePreparationAgent has run (keys exist in path_state)."""
        path_state = state.get("path_state", {})
        has_context = (
            "primary_attribute" in path_state
            or "attribute_join_paths" in path_state
            or "retrieved_column_attributes" in path_state
        )
        if not has_context:
            self.logger.warning(
                "CandidatePreparationAgent output missing: expected primary_attribute, "
                "attribute_join_paths, or retrieved_column_attributes in path_state"
            )
            return False
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Construct SQL from semantic candidates and prepared schema context.

        Uses CandidatePreparationAgent outputs (candidates, tables, queries,
        similar questions). May return a text response when the model does not emit SQL.

        Args:
            state: Current agent state

        Returns:
            Dictionary with:
            - path_state: Contains SQL response, tables, connection, custom analyses
            - messages: Adds SQL response to messages
            - decision: "constructable" or "unconstructable"
        """
        path_state = state.get("path_state", {})
        llm = state["llm"]
        connectors = state.get("connectors") or []
        original_question = get_original_question(state)
        sanitized_question = get_question_for_processing(state)
        main_question = format_dual_question_block(
            original_question, sanitized_question
        )

        primary_attribute: dict | None = path_state.get("primary_attribute")
        attribute_join_paths: list[dict] = path_state.get("attribute_join_paths") or []
        relevant_tables = path_state.get("relevant_tables", [])
        relevant_queries = path_state.get("relevant_queries", [])
        similar_questions = path_state.get("similar_questions", [])
        custom_analyses = path_state.get("custom_analyses", [])
        custom_analyses_str = path_state.get("custom_analyses_str", [])
        sql_attributes = path_state.get("sql_attributes", [])
        sql_attributes_str = path_state.get("sql_attributes_str", [])
        term_synonyms: dict = path_state.get("term_synonyms") or {}
        value_entities: list[str] = path_state.get("value_entities") or []

        connector = resolve_connector_from_tables(relevant_tables, connectors)
        dialect = getattr(connector, "dialect", None)

        self.logger.info(
            "Semantic context: anchor=%s, join_paths=%d, custom_analyses=%d, "
            "sql_attributes=%d, fallback_tables=%d",
            primary_attribute.get("attr_name") if primary_attribute else None,
            len(attribute_join_paths),
            len(custom_analyses),
            len(sql_attributes),
            len(relevant_tables),
        )

        # Format similar questions for prompt
        similar_questions_txt = "\n".join(
            f"question: {x[0]}\nanswer: {x[1]}" for x in similar_questions
        )
        self.logger.info(
            f"Using {len(similar_questions)} similar questions from conversations."
        )

        def build_messages() -> list:
            """
            Build messages for SQL construction.

            Includes semantic candidate context, similar questions, and optionally
            extracted file data or file excerpts.
            """
            relevance_reasoning = path_state.get("table_relevance_reasoning", "")
            observation_block = ""
            if relevance_reasoning:
                observation_block += (
                    f"\nTable selection reasoning:\n{relevance_reasoning}\n"
                )
            observation_block += f"\nlist of important semantic entities with sql snippets:\n{custom_analyses_str}\n"
            if sql_attributes_str:
                observation_block += (
                    f"\nlist of sql attributes (derived metrics/formulas):\n"
                    f"{sql_attributes_str}\n"
                )
            if term_synonyms:
                gloss_lines = ["TERM GLOSSARY (alternate names users may use):"]
                for term_name, syns in term_synonyms.items():
                    gloss_lines.append(
                        f"  {term_name}: also known as {', '.join(syns)}"
                    )
                observation_block += "\n" + "\n".join(gloss_lines) + "\n"

            # Build the search-values section (mandatory text-search filters).
            search_values = ""
            value_terms = [v.strip() for v in value_entities if v and v.strip()]
            if value_terms:
                value_lines = [
                    "**Search Values** (text-search filter values):",
                    "- Every LIKE/ILIKE or text-equality WHERE filter in the SQL "
                    "MUST use one of these terms, and EVERY term listed below MUST "
                    "appear as such a filter. Do NOT invent other text-search "
                    "filters.",
                    "- EXCEPTION: if a term is clearly the wrong word for this "
                    "domain/context, do NOT filter on it — omit that term entirely "
                    "rather than force a match that would miss valid rows.",
                    "- Only filter on columns that ACTUALLY EXIST in the AVAILABLE "
                    "TABLES above — never assume a column exists.",
                    "- Pick whichever existing column best fits each term. If "
                    "unsure, prefer a description-type column for the main entity, "
                    "or a features-type column for adjective/feature values — only "
                    "when such a column actually exists.",
                    "- Avoid matching a term against title or name columns when the "
                    "term is more than 2 words.",
                    "",
                    "Terms:",
                ]
                value_lines.extend(f"  - {term}" for term in value_terms)
                search_values = "\n".join(value_lines) + "\n\n"

            # Build custom analyses section for user prompt
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
                        line += f"\n  SQL: {sql}"
                    ca_lines.append(line)
                ca_section = (
                    "DOMAIN-SPECIFIC CUSTOM ANALYSES (use their SQL patterns as guidance):\n"
                    + "\n".join(ca_lines)
                    + "\n\n"
                )

            # Build sql attributes section for user prompt
            sa_section = ""
            if sql_attributes:
                sa_lines = []
                for a in sql_attributes:
                    line = f"- {a.get('name', '(unnamed)')}"
                    desc = (a.get("description") or "").strip()
                    if desc:
                        line += f": {desc}"
                    expr = (a.get("expression") or "").strip()
                    sql = (a.get("sql") or "").strip()
                    if expr:
                        line += f"\n  Expression: {expr}"
                    if sql and sql != expr:
                        line += f"\n  Full query: {sql}"
                    sa_lines.append(line)
                sa_section = (
                    "SQL ATTRIBUTES (derived metrics/formulas — "
                    "use their expressions and SQL as guidance):\n"
                    + "\n".join(sa_lines)
                    + "\n\n"
                )

            # Build the join-paths section (semantic hint + suggested joins).
            join_paths = ""
            if primary_attribute:
                join_paths = (
                    "## Semantic Hints & Join Paths\n"
                    + _format_semantic_context(primary_attribute, attribute_join_paths)
                    + "\n\n"
                )

            # Build the available-tables schema section.
            tables_section = (
                "AVAILABLE TABLES (schema context):\n"
                + format_tables_for_prompt(relevant_tables)
                if relevant_tables
                else "No tables available."
            )

            # Build user prompt
            user_prompt = create_sql_user_prompt.format(
                dialect=dialect,
                main_question=main_question,
                observation_block=observation_block,
                queries=relevant_queries,
                qa_from_conversations=similar_questions_txt,
                tables=tables_section,
                join_paths=join_paths,
                search_values=search_values,
                custom_analyses=ca_section + sa_section,
            )

            # Choose system prompt based on context
            system_prompt = create_sql_from_candidates_prompt()

            messages = state["messages"] + [
                SystemMessage(content=system_prompt),
                AIMessage(content=user_prompt),
            ]

            # Add calendar time window reminder if needed
            if any(
                phrase in sanitized_question.lower()
                for phrase in ["last week", "last month", "last year"]
            ):
                messages.append(
                    SystemMessage(
                        content="Apply only calendar time windows. DO NOT apply rolling time windows."
                    )
                )

            return messages

        # Choose schema based on context
        # Use SQLGenerationModel for new flow (without formatting)
        # Keep old models for feedback scenarios

        schema = SQLGenerationModel

        def run_with_context() -> tuple:
            """Invoke LLM with messages, optionally including file snippets and extracted data."""
            messages = build_messages()
            try:
                response = safe_invoke_with_structured_output(llm, messages, schema)
            except Exception as e:
                self.logger.error(
                    "LLM structured output failed: %s: %s",
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
                return None, messages
            if response and hasattr(response, "response") and response.response:
                self.logger.info(
                    "LLM response generated: %s...",
                    response.response[:100],
                )
            return response, messages

        MAX_RETRIES = 3
        response, messages = None, []
        for attempt in range(1, MAX_RETRIES + 1):
            response, messages = run_with_context()
            if response is not None:
                break
            self.logger.warning(
                "LLM returned None on attempt %d/%d — retrying.",
                attempt,
                MAX_RETRIES,
            )

        if response is None:
            self.logger.error("LLM returned None after %d attempts.", MAX_RETRIES)
            return {
                "path_state": {
                    **path_state,
                    "unconstructable_explanation": "LLM failed to produce a response.",
                },
                "decision": "unconstructable",
            }

        # Check if we have a valid response (either SQL or text-based answer from file contents)
        has_sql = bool(response.sql_code and response.sql_code.strip())
        has_response = bool(response.response and response.response.strip())

        if has_sql:
            custom_analyses_used = []
            if (
                hasattr(response, "custom_analyses_used")
                and response.custom_analyses_used
            ):
                # Filter custom analyses to keep only those found in candidates
                candidates_ids = {
                    c.get("id") if isinstance(c, dict) else getattr(c, "id", None)
                    for c in path_state.get("custom_analyses", [])
                }
                filtered_elements = [
                    elem
                    for elem in response.custom_analyses_used
                    if (elem.id if hasattr(elem, "id") else elem.get("id"))
                    in candidates_ids
                ]
                response.custom_analyses_used = filtered_elements
                custom_analyses_used = get_custom_analyses_ids(
                    response.custom_analyses_used
                )

            return {
                "messages": messages,  # Don't add formatted response here - formatting agent will do it
                "path_state": {
                    **path_state,
                    "sql_generation_result": response,  # Keep as object (Pydantic model)
                    "relevant_tables": relevant_tables if has_sql else [],
                    "custom_analyses_used": custom_analyses_used,
                },
                "decision": "constructable",
            }
        elif has_response:
            custom_analyses_used = []
            if hasattr(response, "custom_analyses_used"):
                response.response += build_custom_analyses_section(
                    response.custom_analyses_used, path_state.get("custom_analyses", [])
                )
                custom_analyses_used = get_custom_analyses_ids(
                    response.custom_analyses_used
                )

            return {
                "messages": messages + [AIMessage(content=response.response)],
                "path_state": {
                    **path_state,
                    "sql_generation_result": response,
                    "relevant_tables": relevant_tables if has_sql else [],
                    "custom_analyses_used": custom_analyses_used,
                },
                "decision": "constructable",
            }
        else:
            # SQL could not be generated
            return {
                "path_state": {
                    **path_state,
                    "unconstructable_explanation": response.response
                    or "Unable to construct response.",
                },
                "decision": "unconstructable",
            }
