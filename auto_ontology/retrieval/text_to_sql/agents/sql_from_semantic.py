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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from auto_ontology.utils.llm_invoke import safe_invoke_with_structured_output
from auto_ontology.retrieval.text_to_sql.base import BaseAgent, record_thought
from auto_ontology.retrieval.text_to_sql.connector_routing import (
    resolve_connector_from_tables,
)
from auto_ontology.retrieval.data_access.custom_analyses import (
    build_custom_analyses_section,
    get_custom_analyses_ids,
)
from auto_ontology.retrieval.entity_coverage.prompts import format_glossary_section
from auto_ontology.retrieval.text_to_sql.formatters_util import (
    format_important_columns_for_prompt,
    format_semantic_context,
    format_tables_for_prompt,
)
from auto_ontology.retrieval.text_to_sql.state import (
    AgentState,
    get_original_question,
    get_question_for_processing,
)
from auto_ontology.retrieval.text_to_sql.prompts import (
    SCHEMA_ONLY_SQL_SYSTEM_PROMPT,
    SCHEMA_ONLY_SQL_USER_PROMPT,
    create_sql_from_candidates_prompt,
    create_sql_user_prompt,
    format_authoritative_evidence,
    format_custom_analyses_section,
    format_dialect_rules,
    format_dual_question_block,
    format_projection_rules,
    format_sql_examples_section,
    format_value_anchors_section,
)
from auto_ontology.retrieval.text_to_sql.agents.sql_candidate_selection import (
    SelectionContext,
    SQLCandidate,
    select_candidate,
)
from auto_ontology.retrieval.text_to_sql.models import SQLGenerationModel

logger = logging.getLogger(__name__)

# Graph node name this agent is registered under in ``text_to_sql_graph.create_graph``
# (NOT ``self.agent_name``, which is a separate internal/logging name) — must match
# so ``stream_agent_response`` can attribute this agent's recorded thoughts to the
# right step event and ``NODE_LABELS`` entry.
_GRAPH_NODE_NAME = "construct_sql_from_candidates"

_MAX_GENERATION_ATTEMPTS = 3


def format_calculation_sql_template(path_state: dict[str, Any]) -> str:
    """Render the intent subtype's SQL example as structural guidance only."""
    sql_template = (path_state.get("sql_template") or "").strip()
    if not sql_template:
        return ""
    subtype = path_state.get("calculation_subtype") or "calculation"
    return f"""## Calculation SQL shape reference

The question was classified as `{subtype}`. Use this example only as a structural
pattern. Adapt it to the available schema and SQL dialect. Never copy its table names,
column names, or literal values unless the schema and question independently require
the same identifiers or values.

```sql
{sql_template}
```
"""


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
    - path_state["sql_candidates"]: Semantic and schema-only candidates with the
      signals used to choose between them (see ``sql_candidate_selection``);
      path_state["sql_candidate_selected"] names the one kept and
      path_state["sql_candidate_selection"] the stage that decided
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
        llm = state.get("reasoning_llm") or state["llm"]
        connectors = state.get("connectors") or []
        original_question = get_original_question(state)
        sanitized_question = get_question_for_processing(state)
        evidence = state.get("evidence", "")
        authoritative_evidence_section = format_authoritative_evidence(
            evidence, evidence_first_sql=True
        )
        sql_examples_section = format_sql_examples_section(state.get("sql_examples"))
        value_anchors_section = format_value_anchors_section(state.get("value_anchors"))
        calculation_template_section = format_calculation_sql_template(path_state)
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
        if sql_examples_section:
            self.logger.info(
                "Injecting %d reference query pattern(s) into the SQL prompt.",
                len(state.get("sql_examples") or []),
            )
        if value_anchors_section:
            self.logger.info(
                "Injecting %d verified database value(s) into the SQL prompt.",
                len(state.get("value_anchors") or []),
            )
        if calculation_template_section:
            self.logger.info(
                "Injecting the %s calculation SQL shape into the SQL prompt.",
                path_state.get("calculation_subtype"),
            )

        target_db = path_state.get("target_db")

        def build_semantic_messages() -> list[BaseMessage]:
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
            glossary_section = format_glossary_section(state.get("glossary") or [])
            if glossary_section:
                observation_block += f"\n{glossary_section}"
            if authoritative_evidence_section:
                observation_block += f"\n{authoritative_evidence_section}\n"
            # Build custom analyses section for user prompt
            ca_section = format_custom_analyses_section(custom_analyses)

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
                    + format_semantic_context(
                        primary_attribute,
                        attribute_join_paths,
                        target_db=target_db,
                        dialect=dialect,
                    )
                    + "\n\n"
                )

            # Build the available-tables schema section.
            tables_section = (
                "AVAILABLE TABLES (schema context):\n"
                + format_tables_for_prompt(
                    relevant_tables, target_db=target_db, dialect=dialect
                )
                if relevant_tables
                else "No tables available."
            )
            important_columns = format_important_columns_for_prompt(
                primary_attribute,
                attribute_join_paths,
                relevant_tables,
                target_db=target_db,
                dialect=dialect,
            )

            # Build user prompt
            user_prompt = create_sql_user_prompt.format(
                dialect=dialect,
                dialect_rules=format_dialect_rules(dialect),
                projection_rules=format_projection_rules(
                    state.get("shorten_answer", False)
                ),
                main_question=main_question,
                observation_block=observation_block,
                queries=relevant_queries,
                qa_from_conversations=similar_questions_txt,
                tables=tables_section,
                important_columns=important_columns,
                join_paths=join_paths,
                custom_analyses=ca_section + sa_section,
            )

            # Choose system prompt based on context
            system_prompt = create_sql_from_candidates_prompt(
                dialect=dialect,
                target_db=target_db,
                has_sql_examples=bool(sql_examples_section),
            )

            messages = state["messages"] + [SystemMessage(content=system_prompt)]
            if authoritative_evidence_section:
                messages.append(SystemMessage(content=authoritative_evidence_section))
            if calculation_template_section:
                messages.append(SystemMessage(content=calculation_template_section))
            # Before the query patterns: anchors state what this database
            # contains, which constrains the SQL more tightly than precedent
            # from another database does.
            if value_anchors_section:
                messages.append(SystemMessage(content=value_anchors_section))
            # After evidence, so that on any conflict the authoritative block is
            # the one the model read first and the advisory one qualifies it.
            if sql_examples_section:
                messages.append(SystemMessage(content=sql_examples_section))
            messages.append(HumanMessage(content=user_prompt))

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

        def build_schema_only_messages() -> list[BaseMessage]:
            """Build the minimal question/evidence/table-view prompt."""
            tables = format_tables_for_prompt(
                relevant_tables, target_db=target_db, dialect=dialect
            )
            user_prompt = SCHEMA_ONLY_SQL_USER_PROMPT.format(
                question=sanitized_question,
                evidence=evidence,
                dialect=dialect or "SQL",
                tables=tables,
            )
            return [
                SystemMessage(content=SCHEMA_ONLY_SQL_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]

        def generate(
            name: str, messages: list[BaseMessage]
        ) -> tuple[SQLGenerationModel | None, list[BaseMessage]]:
            """One structured SQL + description + response call, with retries."""
            for attempt in range(1, _MAX_GENERATION_ATTEMPTS + 1):
                try:
                    response = safe_invoke_with_structured_output(
                        llm, messages, SQLGenerationModel
                    )
                except Exception as e:
                    self.logger.error(
                        "%s SQL generation failed: %s: %s",
                        name,
                        type(e).__name__,
                        e,
                        exc_info=True,
                    )
                    response = None
                if response is not None:
                    return response, messages
                self.logger.warning(
                    "%s SQL generation returned no response on attempt %d/%d.",
                    name,
                    attempt,
                    _MAX_GENERATION_ATTEMPTS,
                )
            return None, messages

        with ThreadPoolExecutor(max_workers=2) as pool:
            semantic_future = pool.submit(
                generate, "semantic", build_semantic_messages()
            )
            schema_future = pool.submit(
                generate, "schema-only", build_schema_only_messages()
            )
            semantic_response, semantic_messages = semantic_future.result()
            schema_response, schema_messages = schema_future.result()

        candidates = [
            (source, response, messages)
            for source, response, messages in (
                ("semantic", semantic_response, semantic_messages),
                ("schema-only", schema_response, schema_messages),
            )
            if response is not None
        ]
        for source, candidate, _ in candidates:
            self.logger.info(
                "%s SQL description: %s", source, candidate.description[:200]
            )

        if not candidates:
            self.logger.error("Both SQL generation candidates failed.")
            return {
                "path_state": {
                    **path_state,
                    "unconstructable_explanation": "LLM failed to produce a response.",
                },
                "decision": "unconstructable",
            }

        sql_candidates = [
            SQLCandidate(source=source, response=candidate, messages=messages)
            for source, candidate, messages in candidates
        ]
        selection = select_candidate(
            sql_candidates,
            SelectionContext(
                llm=llm,
                question=sanitized_question,
                evidence=evidence,
                tables_text=format_tables_for_prompt(
                    relevant_tables, target_db=target_db, dialect=dialect
                ),
                connector=connector,
                dialect=dialect,
                target_db=target_db,
                relevant_tables=relevant_tables,
            ),
        )
        self.logger.info(
            "SQL candidate selected: %s (stage=%s: %s).",
            selection.selected.source,
            selection.stage,
            selection.reason,
        )

        response = selection.selected.response
        messages = selection.selected.messages
        path_state = {
            **path_state,
            "sql_candidates": [c.summary() for c in sql_candidates],
            "sql_candidate_selected": selection.selected.source,
            "sql_candidate_selection": {
                "stage": selection.stage,
                "reason": selection.reason,
            },
        }

        thought = (getattr(response, "thought", "") or "").strip()
        if thought:
            record_thought(path_state, _GRAPH_NODE_NAME, thought)

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
