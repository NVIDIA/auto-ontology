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
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from auto_ontology.utils.llm_invoke import (
    get_llm_client,
    safe_invoke_with_structured_output,
)
from auto_ontology.retrieval.text_to_sql.base import BaseAgent, record_thought
from auto_ontology.retrieval.text_to_sql.connector_routing import (
    resolve_connector_from_tables,
)
from auto_ontology.retrieval.text_to_sql.agents.sql_execution import _run_sql
from auto_ontology.retrieval.text_to_sql import candidate_flags as flags
from auto_ontology.retrieval.text_to_sql import candidate_strategies as strategies
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
from auto_ontology.retrieval.text_to_sql.models import (
    SQLDecompositionModel,
    SQLDecompositionTreeModel,
    SQLGenerationModel,
    SQLQueryPlanModel,
    SyntheticSQLExamplesModel,
)

logger = logging.getLogger(__name__)

# Sampling clients are keyed by temperature and reused across questions:
# building a ChatNVIDIA per candidate per question is pure overhead, and the
# client carries no per-request state.
_sampling_llm_cache: dict[float, Any] = {}


def _get_sampling_llm(temperature: float):
    """A cached LLM client at *temperature*, for candidate diversity.

    gpt-5.x / o-series ignore an explicit temperature because their provider
    default already samples, so for those the schema shuffle is the diversity
    lever; nemotron and claude need the raised temperature to actually
    diverge.
    """
    client = _sampling_llm_cache.get(temperature)
    if client is None:
        client = get_llm_client(temperature=temperature)
        _sampling_llm_cache[temperature] = client
    return client


# Graph node name this agent is registered under in ``text_to_sql_graph.create_graph``
# (NOT ``self.agent_name``, which is a separate internal/logging name) — must match
# so ``stream_agent_response`` can attribute this agent's recorded thoughts to the
# right step event and ``NODE_LABELS`` entry.
_GRAPH_NODE_NAME = "construct_sql_from_candidates"


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
        evidence = state.get("evidence", "")
        sql_examples = list(state.get("sql_examples") or [])
        sql_examples_section = format_sql_examples_section(sql_examples)
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

        def build_messages(
            tables_variant: list[dict] | None = None,
            *,
            examples_variant: list[dict] | None = None,
            schema_directive: str = "",
            strategy: str = "",
        ) -> list:
            """
            Build messages for SQL construction.

            Includes semantic candidate context, similar questions, and optionally
            extracted file data or file excerpts.

            The four keyword arguments are what makes one candidate slot differ
            from another: a reordered schema, a different slice of reference
            examples, a schema-reading directive, and a strategy instruction.
            All default to the single-candidate behaviour.
            """
            tables_for_prompt = (
                relevant_tables if tables_variant is None else tables_variant
            )
            examples_section = (
                sql_examples_section
                if examples_variant is None
                else format_sql_examples_section(examples_variant)
            )
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

            target_db = path_state.get("target_db")

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
                    tables_for_prompt, target_db=target_db, dialect=dialect
                )
                if tables_for_prompt
                else "No tables available."
            )
            important_columns = format_important_columns_for_prompt(
                primary_attribute,
                attribute_join_paths,
                tables_for_prompt,
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
                has_sql_examples=bool(examples_section),
            )

            messages = state["messages"] + [SystemMessage(content=system_prompt)]
            # Before the evidence and example blocks: these say how this
            # candidate should differ from its siblings, and they have to hold
            # even where the shared context would push every slot the same way.
            if strategy:
                messages.append(SystemMessage(content=strategy))
            if schema_directive:
                messages.append(SystemMessage(content=schema_directive))
            if evidence:
                messages.append(
                    SystemMessage(content=format_authoritative_evidence(evidence))
                )
            if calculation_template_section:
                messages.append(SystemMessage(content=calculation_template_section))
            # Before the query patterns: anchors state what this database
            # contains, which constrains the SQL more tightly than precedent
            # from another database does.
            if value_anchors_section:
                messages.append(SystemMessage(content=value_anchors_section))
            # After evidence, so that on any conflict the authoritative block is
            # the one the model read first and the advisory one qualifies it.
            if examples_section:
                messages.append(SystemMessage(content=examples_section))
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

        # Choose schema based on context
        # Use SQLGenerationModel for new flow (without formatting)
        # Keep old models for feedback scenarios

        schema = SQLGenerationModel

        def build_synthetic_examples(
            artifact: SyntheticSQLExamplesModel,
        ) -> list[dict]:
            """Keep only the invented demonstrations the target database accepts.

            A generated example is instruction, not decoration: an invalid one
            actively teaches the final call a table or column that does not
            exist. Executing each is cheap next to the LLM call that wrote it.
            """
            kept: list[dict] = []
            for example in artifact.examples:
                checked = _run_sql(example.sql, connector)
                if checked.error:
                    continue
                kept.append(
                    {
                        "question": example.question,
                        "sql": example.sql,
                        "evidence": getattr(example, "reasoning", "") or "",
                        "db": path_state.get("target_db") or "",
                    }
                )
            return kept

        def generate_candidate(index: int) -> tuple:
            """Produce one SQL candidate, returning ``(response, messages, tag)``.

            Slot 0 uses the base client and the untouched schema order, so
            ``BIRD_NCAND=1`` reproduces the previous behaviour exactly. Other
            slots run a two-stage method: build an explicit intermediate
            artifact, then translate that artifact into SQL in a second call.
            """
            started = time.monotonic()
            tag = strategies.strategy_for_slot(index, n_candidates)

            if strategies.slot_samples(index, n_candidates):
                tables_variant = strategies.shuffled_tables(
                    relevant_tables, random.Random(1000 + index)
                )
                client = _get_sampling_llm(candidate_temp)
            else:
                tables_variant = None
                client = llm

            examples_variant = (
                None
                if n_candidates < 2
                else strategies.rotate_examples(sql_examples, index)
            )
            directive = (
                strategies.schema_directive(index, path_state.get("entity_columns"))
                if n_candidates > 1
                else ""
            )

            def stage1(stage1_messages: list, artifact_schema):
                try:
                    return safe_invoke_with_structured_output(
                        client, stage1_messages, artifact_schema
                    )
                except Exception as exc:
                    self.logger.warning(
                        "Candidate %d [%s] stage 1 failed: %s: %s",
                        index,
                        tag,
                        type(exc).__name__,
                        exc,
                    )
                    return None

            if tag == "baseline":
                messages = build_messages(
                    tables_variant,
                    examples_variant=examples_variant,
                    schema_directive=directive,
                )
            elif tag.startswith("query_plan"):
                stage1_messages = build_messages(
                    tables_variant,
                    examples_variant=examples_variant,
                    schema_directive=directive,
                    strategy=strategies.QUERY_PLAN_ARTIFACT_PROMPT,
                )
                artifact = stage1(stage1_messages, SQLQueryPlanModel)
                if artifact is None:
                    return None, stage1_messages, tag
                messages = build_messages(
                    tables_variant,
                    examples_variant=examples_variant,
                    schema_directive=directive,
                    strategy=strategies.QUERY_PLAN_TRANSLATION_PROMPT.format(
                        artifact=artifact.plan
                    ),
                )
            elif tag == "decomposition":
                tree_mode = flags.decomposition_tree()
                stage1_messages = build_messages(
                    tables_variant,
                    examples_variant=examples_variant,
                    schema_directive=directive,
                    strategy=(
                        strategies.DECOMPOSITION_TREE_ARTIFACT_PROMPT
                        if tree_mode
                        else strategies.DECOMPOSITION_ARTIFACT_PROMPT
                    ),
                )
                artifact = stage1(
                    stage1_messages,
                    SQLDecompositionTreeModel if tree_mode else SQLDecompositionModel,
                )
                if artifact is None:
                    return None, stage1_messages, tag
                artifact_text = (
                    strategies.format_decomposition_tree(artifact)
                    if tree_mode
                    else strategies.format_decomposition_list(artifact)
                )
                translation = (
                    strategies.DECOMPOSITION_TREE_TRANSLATION_PROMPT
                    if tree_mode
                    else strategies.DECOMPOSITION_TRANSLATION_PROMPT
                )
                messages = build_messages(
                    tables_variant,
                    examples_variant=examples_variant,
                    schema_directive=directive,
                    strategy=translation.format(artifact=artifact_text),
                )
            elif tag == "synthetic_examples":
                # Cross-database examples stay hidden while inventing the
                # same-schema ones: they would anchor the generator back onto
                # the very signal this strategy exists to replace.
                stage1_messages = build_messages(
                    tables_variant,
                    examples_variant=[],
                    schema_directive=directive,
                    strategy=strategies.synthetic_artifact_prompt(),
                )
                artifact = stage1(stage1_messages, SyntheticSQLExamplesModel)
                if artifact is None:
                    return None, stage1_messages, tag
                synthetic = build_synthetic_examples(artifact)
                if not synthetic:
                    self.logger.info(
                        "Candidate %d [%s]: no invented example survived execution.",
                        index,
                        tag,
                    )
                    return None, stage1_messages, tag
                messages = build_messages(
                    tables_variant,
                    examples_variant=synthetic,
                    schema_directive=directive,
                    strategy=strategies.SYNTHETIC_USE_PROMPT,
                )
            elif tag == "alt_table_set":
                # One-shot SQL behind a hard table-set divergence directive.
                # The goal is to fail on different questions than slot 0, not
                # merely to word the same wrong reading differently.
                messages = build_messages(
                    tables_variant,
                    examples_variant=examples_variant,
                    schema_directive=(f"{directive}\n\n" if directive else "")
                    + strategies.ALT_TABLE_SET_STRATEGY,
                )
            else:
                raise AssertionError(f"Unknown candidate strategy: {tag}")

            try:
                response = safe_invoke_with_structured_output(client, messages, schema)
            except Exception as e:
                self.logger.error(
                    "LLM structured output failed (candidate %d): %s: %s",
                    index,
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
                return None, messages, tag

            if response is not None:
                self.logger.info(
                    "Candidate %d [%s] produced SQL in %.1fs.",
                    index,
                    tag,
                    time.monotonic() - started,
                )
            return response, messages, tag

        n_candidates = flags.num_candidates()
        candidate_temp = flags.candidate_temperature()
        candidates: list = []
        messages: list = []

        if n_candidates < 2:
            # Single-candidate path, retry-on-None preserved exactly.
            MAX_RETRIES = 3
            response = None
            for attempt in range(1, MAX_RETRIES + 1):
                response, messages, _ = generate_candidate(0)
                if response is not None:
                    break
                self.logger.warning(
                    "LLM returned None on attempt %d/%d — retrying.",
                    attempt,
                    MAX_RETRIES,
                )
            if response is not None:
                candidates = [response]
        else:
            # The shared in-flight semaphore in llm_invoke is what keeps real
            # endpoint concurrency polite, so this pool only bounds threads.
            results: list = [None] * n_candidates
            base_messages: list = []
            with ThreadPoolExecutor(
                max_workers=min(n_candidates, flags.max_parallel()),
                thread_name_prefix="sqlcand",
            ) as pool:
                futures = {
                    pool.submit(generate_candidate, i): i for i in range(n_candidates)
                }
                for future in as_completed(futures):
                    index = futures[future]
                    candidate, candidate_messages, _tag = future.result()
                    results[index] = candidate
                    if index == 0:
                        base_messages = candidate_messages
            candidates = [
                candidate
                for candidate in results
                if candidate is not None
                and candidate.sql_code
                and candidate.sql_code.strip()
            ]
            messages = base_messages
            self.logger.info(
                "Generated %d/%d usable SQL candidates.", len(candidates), n_candidates
            )

        response = candidates[0] if candidates else None

        if response is None:
            self.logger.error("No usable SQL candidate produced.")
            return {
                "path_state": {
                    **path_state,
                    "unconstructable_explanation": "LLM failed to produce a response.",
                },
                "decision": "unconstructable",
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
                    # The whole pool, for the selection node and for callers
                    # measuring the best-of-N ceiling. Holds one entry on the
                    # single-candidate path.
                    "sql_candidates": candidates,
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
