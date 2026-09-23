# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SQL Reconstruction Agent

Reconstructs SQL queries that failed validation. On the first
reconstruction attempt the agent classifies the error type via an LLM
call. When the error is MISSING_DATA, it searches the data VDB for
additional tables using LLM-suggested queries, enriches them with full
column info from the store, and merges them into the available context.
For FIXABLE errors it proceeds directly to SQL reconstruction with the
existing tables.

All failed attempts are tracked and included in the reconstruction
prompt so the LLM does not repeat the same broken SQL.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from auto_ontology.retrieval.data_access.custom_analyses import get_custom_analyses_ids
from auto_ontology.dal.datasources import fetch_tables_by_ids
from auto_ontology.retrieval.data_access.relevant_tables import (
    dedupe_merge_relevant_tables,
    get_relevant_tables,
)
from auto_ontology.utils.llm_invoke import invoke_with_structured_output
from auto_ontology.retrieval.text_to_sql.formatters_util import format_tables_for_prompt
from auto_ontology.retrieval.text_to_sql.connector_routing import (
    resolve_connector_from_tables,
)
from auto_ontology.retrieval.text_to_sql.base import BaseAgent, record_thought
from auto_ontology.retrieval.text_to_sql.models import SQLGenerationModel
from auto_ontology.retrieval.text_to_sql.prompts import (
    format_authoritative_evidence,
    format_dual_question_block,
)
from auto_ontology.retrieval.text_to_sql.state import (
    AgentState,
    get_original_question,
    get_question_for_processing,
)

logger = logging.getLogger(__name__)

# Graph node name this agent is registered under in ``text_to_sql_graph.create_graph``
# (NOT ``self.agent_name``, which is a separate internal/logging name) — must match
# so ``stream_agent_response`` can attribute this agent's recorded thoughts to the
# right step event and ``NODE_LABELS`` entry.
_GRAPH_NODE_NAME = "reconstruct_sql"

# ------------------------------------------------------------------
# Known-column grounding (from the semantic resolution done up front)
# ------------------------------------------------------------------


def _collect_known_columns(
    primary_attribute: dict | None,
    attribute_join_paths: list[dict] | None,
) -> tuple[list[str], list[str]]:
    """Shared line-collection for known columns/join keys, used by both the
    repair prompt and the (more hedged) classification prompt renderers
    below. See ``_format_known_columns`` for the column/hop line format.
    """
    entries = list(attribute_join_paths or [])
    if primary_attribute:
        entries = [primary_attribute] + entries

    column_lines = []
    seen_columns = set()
    hop_lines = []
    seen_hops = set()
    for entry in entries:
        col_name = entry.get("col_name")
        table_name = entry.get("table_name")
        if col_name and table_name:
            key = (table_name, col_name)
            if key not in seen_columns:
                seen_columns.add(key)
                label = entry.get("attr_name") or col_name
                # datatype is only populated for attrs (re-)ingested since
                # this field was added — older rows just omit the tag.
                datatype = entry.get("datatype")
                tag = f" ({datatype})" if datatype else ""
                column_lines.append(f"  {label}: {table_name}.{col_name}{tag}")

        # Multi-hop entries also carry the exact join key on each side of
        # every hop (e.g. treatmentbasics.encref = encounters.enckey) — this
        # is what actually fixes wrong-FK-column guesses like "enc_ref" vs
        # "encref", which the flat column mapping above can't cover since
        # the FK column isn't itself a resolved business attribute.
        for hop in entry.get("path") or []:
            src_table = hop.get("source_table")
            src_col = hop.get("source_column")
            tgt_table = hop.get("target_table")
            tgt_col = hop.get("target_column")
            if not (src_table and src_col and tgt_table and tgt_col):
                continue
            hop_key = (src_table, src_col, tgt_table, tgt_col)
            if hop_key in seen_hops:
                continue
            seen_hops.add(hop_key)
            hop_lines.append(f"  {src_table}.{src_col} = {tgt_table}.{tgt_col}")

    return column_lines, hop_lines


def _format_known_columns(
    primary_attribute: dict | None,
    attribute_join_paths: list[dict] | None,
) -> str:
    """Render a short, repair-focused reminder of already-resolved column
    names, so reconstruction doesn't have to re-guess casing/existence from
    the error message alone (e.g. "exch_spot" vs "EXCH_SPOT" when the real
    column is "quote_depth_snapshot"). Terse by design — this is a repair
    prompt, not the first-pass generation prompt.
    """
    column_lines, hop_lines = _collect_known_columns(
        primary_attribute, attribute_join_paths
    )
    if not column_lines and not hop_lines:
        return ""

    sections = []
    if column_lines:
        sections.append(
            "\nKNOWN COLUMN MAPPINGS (already resolved — use these exact "
            "names/casing, do not guess):\n" + "\n".join(column_lines)
        )
    if hop_lines:
        sections.append(
            "\nKNOWN JOIN KEYS (already resolved — use these exact join "
            "conditions, do not guess FK column names):\n" + "\n".join(hop_lines)
        )

    return "\n".join(sections) + "\n\n"


def _format_known_columns_for_classification(
    primary_attribute: dict | None,
    attribute_join_paths: list[dict] | None,
) -> str:
    """Render the same known-columns data for the error CLASSIFIER prompt,
    with deliberately hedged framing.

    Unlike the repair prompt's "already resolved, use these exact names"
    (appropriate there — it's just telling the LLM what names to write), this
    context feeds a root-cause judgment call: is the data missing, or is the
    SQL just wrong? The anchor/join-path resolution that produces this data
    is itself a best-effort LLM+graph process that has been wrong before
    (see candidates_preparation.py's anchor-selection fallback path and its
    "relevance filter has repeatedly proven unreliable" note) — so this must
    NOT be framed as ground truth, or the classifier could be biased toward
    "fixable" even when the resolution itself picked the wrong table/column
    and a real missing_data rediscovery is needed.
    """
    column_lines, hop_lines = _collect_known_columns(
        primary_attribute, attribute_join_paths
    )
    if not column_lines and not hop_lines:
        return ""

    sections = []
    if column_lines:
        sections.append(
            "\nColumns resolved by an earlier step (may be "
            "incomplete). The datatype is in parentheses "
            "where known):\n" + "\n".join(column_lines)
        )
    if hop_lines:
        sections.append(
            "\nJoin keys tentatively resolved by an earlier step (same "
            "caveat):\n" + "\n".join(hop_lines)
        )
    sections.append(
        "\nIf the error concerns one of the above, prefer 'fixable' unless "
        "you have a specific reason to believe this resolution itself is "
        "wrong (e.g. it points at a table/column that doesn't plausibly "
        "answer the question), in which case 'missing_data' is still "
        "correct."
    )

    return "\n".join(sections) + "\n\n"


# ------------------------------------------------------------------
# Deterministic fast-path classification
# ------------------------------------------------------------------

# Postgres raises this when a CAST/::numeric hits a formatted string value
# (e.g. "45.2%" or "USD 81,931.00") instead of a bare number — always fixable
# with the same tables by stripping non-numeric characters before casting,
# never a missing_data situation. Matching here skips the LLM classification
# call, same as the existing db_probe pre-classification checks below.
# Postgres-specific error text; other dialects just won't match (fails safe),
# could be generalized with per-dialect patterns later.
_NUMERIC_FORMAT_CAST_ERROR_RE = re.compile(
    r"invalid input syntax for type (?:numeric|double precision|integer|bigint):"
    r'\s*"[^"]*(?:%|\$|USD|EUR|GBP)[^"]*"',
    re.IGNORECASE,
)


def _is_numeric_format_cast_error(error: str) -> bool:
    """Whether ``error`` is a Postgres cast failure on a %/currency-formatted
    string value (see ``_NUMERIC_FORMAT_CAST_ERROR_RE``)."""
    return bool(_NUMERIC_FORMAT_CAST_ERROR_RE.search(error or ""))


class ErrorType(str, Enum):
    """Root-cause classification for reconstruction.

    The only decision that matters is: do we need more tables or can we
    fix the SQL with what we already have?  The error may have been
    raised by unified SQL validation or SQL execution
    -- but the *symptom* (syntax / runtime / wrong intent) doesn't
    always match the *root cause*.  So we let the LLM decide.
    """

    MISSING_DATA = "missing_data"
    FIXABLE = "fixable"


class ErrorAnalysis(BaseModel):
    """LLM output: classify the root cause and optionally suggest VDB queries."""

    model_config = ConfigDict(extra="forbid")

    error_type: ErrorType = Field(
        description=(
            "Classify the root cause:\n"
            "- missing_data: the available tables/columns are insufficient "
            "to answer the question — new data must be discovered.\n"
            "- fixable: the SQL has a syntax error, wrong column, runtime "
            "failure, or wrong logic — it can be fixed with the same tables."
        ),
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description=(
            "Only populate when error_type is 'missing_data'. "
            "2-4 semantic search queries to find the missing data. "
            "Each query should describe a concept, entity, or relationship "
            "needed but not in the current tables."
        ),
    )
    explanation: str = Field(
        default="",
        description="Brief explanation of the diagnosis.",
    )


_ANALYSIS_PROMPT_TEMPLATE = """\
You are diagnosing why a SQL query could not be constructed or is incorrect.

Question the user asked:
"{question}"

Tables available:
{table_summary}
{known_columns_section}
Error / previous attempt:
{error_context}

Classify the ROOT CAUSE (not the symptom):
- missing_data: the tables above do NOT contain the data needed to answer \
the question — even a perfect SQL rewrite would fail because the right \
tables/columns are absent. Provide 2-4 semantic search queries to find \
the missing concepts in our ontology database (focus on entity names and \
relationships, NOT SQL syntax).
- fixable: the SQL can be corrected using the SAME tables (syntax error, \
wrong column reference, wrong aggregation, bad logic, etc.). Leave \
search_queries empty."""


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------


class SQLReconstructionAgent(BaseAgent):
    """Reconstruct failed SQL, diagnosing root cause and tracking history."""

    def __init__(self) -> None:
        super().__init__("sql_reconstruction")

    def validate_input(self, state: AgentState) -> bool:
        """Validate that error and previous response are available."""
        path_state = state.get("path_state", {})
        if not path_state.get("error"):
            self.logger.warning("No error found for SQL reconstruction")
            return False
        if not path_state.get("sql_generation_result"):
            self.logger.warning("No previous SQL response found for reconstruction")
            return False
        return True

    def _format_attempt_history(self, attempts: list[dict], *, label: str) -> list[str]:
        """Render past ``{"sql", "error"}`` attempts as numbered prompt lines."""
        lines: list[str] = []
        for i, attempt in enumerate(attempts, 1):
            if len(attempt["sql"]) > 600 or len(attempt["error"]) > 600:
                self.logger.info(
                    "History %s %d truncated for prompt (sql=%d chars, error=%d chars)",
                    label,
                    i,
                    len(attempt["sql"]),
                    len(attempt["error"]),
                )
            lines.append(
                f"  {label} {i}: {attempt['sql'][:600]}\n"
                f"  Error: {attempt['error'][:600]}"
            )
        return lines

    # ------------------------------------------------------------------
    # Error analysis
    # ------------------------------------------------------------------

    def _analyze_error(
        self,
        state: AgentState,
        question: str,
        error_context: str,
        existing_tables: list[dict],
    ) -> ErrorAnalysis:
        """Ask the LLM to classify the error and suggest search queries."""
        llm = state["llm"]
        path_state = state["path_state"]

        if ext_err := path_state.get("error"):
            error_context = f"External feedback: {ext_err}\n\n{error_context}"

        table_summary = (
            ", ".join(t.get("name", "?") for t in existing_tables) or "(none)"
        )

        # Cheap, no extra LLM/DB call — same data _format_known_columns() uses
        # for the repair prompt, just rendered with hedged framing here since
        # this feeds a root-cause judgment call rather than a "what to write"
        # instruction. See _format_known_columns_for_classification's
        # docstring for why the framing differs.
        known_columns_section = _format_known_columns_for_classification(
            path_state.get("primary_attribute"),
            path_state.get("attribute_join_paths"),
        )

        prompt = _ANALYSIS_PROMPT_TEMPLATE.format(
            question=question,
            table_summary=table_summary,
            known_columns_section=known_columns_section,
            error_context=error_context,
        )

        result = invoke_with_structured_output(
            llm,
            [SystemMessage(content=prompt)],
            ErrorAnalysis,
        )

        if result is None:
            return ErrorAnalysis(
                error_type=ErrorType.FIXABLE,
                explanation="LLM analysis returned None — defaulting to fixable.",
            )
        return result

    # ------------------------------------------------------------------
    # VDB discovery (MISSING_DATA path)
    # ------------------------------------------------------------------

    def _discover_tables(
        self,
        state: AgentState,
        search_queries: list[str],
        existing_tables: list[dict],
    ) -> list[dict]:
        """Search the data VDB for additional tables, enrich with the store columns."""
        data_retriever = state.get("data_retriever")
        if data_retriever is None:
            self.logger.warning("No data_retriever — skipping table discovery")
            return []

        existing_ids = {str(t.get("id", "")) for t in existing_tables if t.get("id")}

        path_state_for_db = state.get("path_state") or {}
        database_name = path_state_for_db.get("target_db") or path_state_for_db.get(
            "retrieval_database"
        )
        combined: list[dict] = []
        for query_text in search_queries:
            try:
                hits = get_relevant_tables(
                    data_retriever,
                    query_text,
                    k=3,
                    database_name=database_name,
                )
                combined.extend(hits)
            except Exception:
                self.logger.warning(
                    "Table discovery failed for query: %s",
                    query_text,
                    exc_info=True,
                )

        if not combined:
            self.logger.info("VDB returned 0 hits for discovery queries")
            return []

        combined = dedupe_merge_relevant_tables(combined)

        new_ids = [
            str(t["id"])
            for t in combined
            if t.get("id") and str(t["id"]) not in existing_ids
        ]
        if not new_ids:
            self.logger.info("Discovery found no tables beyond existing set")
            return []

        enriched = fetch_tables_by_ids(new_ids)
        self.logger.info(
            "Discovery found %d new table(s): %s",
            len(enriched),
            [t["name"] for t in enriched],
        )
        return enriched

    @staticmethod
    def _merge_tables(existing: list[dict], discovered: list[dict]) -> list[dict]:
        """Merge discovered tables into existing, avoiding duplicates by id."""
        by_id: dict[str, dict] = {}
        for t in existing:
            tid = str(t.get("id") or "")
            if tid:
                by_id[tid] = t
        for tbl in discovered:
            tid = str(tbl.get("id") or "")
            if not tid:
                continue
            if tid in by_id:
                old_cols = {c["name"] for c in by_id[tid].get("columns", [])}
                for col in tbl.get("columns", []):
                    if col.get("name") and col["name"] not in old_cols:
                        by_id[tid].setdefault("columns", []).append(col)
                        old_cols.add(col["name"])
            else:
                by_id[tid] = tbl
        return list(by_id.values())

    # ------------------------------------------------------------------
    # Main execute
    # ------------------------------------------------------------------

    def execute(self, state: AgentState) -> Dict[str, Any]:
        path_state = state.get("path_state", {})
        llm = state["llm"]
        error = path_state.get("error", "")
        incorrect_response = path_state.get("sql_generation_result")
        original_question = get_original_question(state)
        sanitized_question = get_question_for_processing(state)
        # Evidence now arrives as its own state field rather than embedded in
        # the question text, so question_block never contains it and needs
        # no stripping.
        evidence = state["evidence"]
        question_block = format_dual_question_block(
            original_question,
            sanitized_question,
        )

        messages = state["messages"]
        relevant_tables = list(path_state.get("relevant_tables") or [])

        sql_code = getattr(incorrect_response, "sql_code", "") or ""
        previous_thought = (getattr(incorrect_response, "thought", "") or "").strip()

        # Accumulate every prior `thought` (not just the last one) so an
        # assumption made two retries ago that the latest thought doesn't
        # repeat isn't silently lost from the repair prompt.
        interpretation_history = list(path_state.get("interpretation_history") or [])
        if previous_thought and previous_thought not in interpretation_history:
            interpretation_history.append(previous_thought)
        path_state["interpretation_history"] = interpretation_history

        # --- Step 1: Classify the error (once per reconstruction chain) ---
        if not path_state.get("error_analysis_done"):
            path_state["error_analysis_done"] = True

            # A db_probe check (literal_check / jsonb_path_check) already
            # confirmed the correct value/key live in a table the query is
            # already joining — that can never be a missing_data situation,
            # so skip the LLM classification (and the table-discovery
            # detour it could trigger) and go straight to fixable.
            if path_state.pop("error_known_fixable", False):
                self.logger.info(
                    "Error pre-classified fixable (db_probe check) — skipping "
                    "LLM error analysis and table discovery"
                )
                path_state["error_type"] = ErrorType.FIXABLE.value
                analysis = None
            elif _is_numeric_format_cast_error(error):
                self.logger.info(
                    "Error pre-classified fixable (numeric/currency-format cast) "
                    "— skipping LLM error analysis and table discovery"
                )
                path_state["error_type"] = ErrorType.FIXABLE.value
                analysis = None
            else:
                response_text = getattr(incorrect_response, "response", "") or ""
                error_context = (
                    f"SQL: {sql_code}\n"
                    f"Response: {response_text}\n"
                    f"Actual validation/execution error: {error}"
                )

                analysis = self._analyze_error(
                    state, question_block, error_context, relevant_tables
                )
                path_state["error_type"] = analysis.error_type.value
                self.logger.info(
                    "Error analysis: %s — %s",
                    analysis.error_type.value,
                    analysis.explanation[:150],
                )
                if analysis.explanation:
                    record_thought(path_state, _GRAPH_NODE_NAME, analysis.explanation)

            if (
                analysis is not None
                and analysis.error_type == ErrorType.MISSING_DATA
                and analysis.search_queries
            ):
                new_tables = self._discover_tables(
                    state, analysis.search_queries, relevant_tables
                )
                if new_tables:
                    relevant_tables = self._merge_tables(relevant_tables, new_tables)
                    self.logger.info(
                        "Tables after discovery: %d (%s)",
                        len(relevant_tables),
                        [t["name"] for t in relevant_tables],
                    )

        # --- Step 2: Accumulate failed attempts ---
        failed_attempts: list[dict] = list(path_state.get("failed_attempts") or [])
        failed_attempts.append({"sql": sql_code, "error": error})
        path_state["failed_attempts"] = failed_attempts

        # --- Step 3: Build reconstruction prompt ---
        # Resolved after table discovery above, so a newly merged table is
        # covered too. Without the dialect the table list would drop the
        # catalog on a catalog-qualified engine, and since the model copies
        # these names verbatim every retry would reproduce the unresolvable
        # name it is being asked to fix.
        connector = resolve_connector_from_tables(
            relevant_tables, state.get("connectors") or []
        )
        dialect = getattr(connector, "dialect", None)

        tables_section = ""
        if relevant_tables:
            formatted_tables = format_tables_for_prompt(
                relevant_tables,
                target_db=path_state.get("target_db"),
                dialect=dialect,
            )
            tables_section = (
                "\nAvailable tables and columns (use ONLY these):\n\n"
                f"{formatted_tables}\n\n"
            )

        history_section = ""
        # Round 1's own reconstruction lineage, carried forward read-only by
        # a debug turn's seed (see coordinator.py::_apply_debug_seed) — kept
        # under a separate key so it never touches len(failed_attempts),
        # which the graph's routers use as the reconstruction-budget cap.
        # Every entry here is already a *past* attempt (none is "the one
        # currently being reconstructed"), unlike failed_attempts below.
        prior_round_failed_attempts: list[dict] = list(
            path_state.get("prior_round_failed_attempts") or []
        )
        history_lines = [
            *self._format_attempt_history(
                prior_round_failed_attempts, label="Round 1 attempt"
            ),
            *self._format_attempt_history(failed_attempts[:-1], label="Attempt"),
        ]
        if history_lines:
            history_section = (
                "\nPREVIOUS FAILED ATTEMPTS (do NOT repeat any of these):\n"
                + "\n".join(history_lines)
                + "\n\n"
            )

        known_columns_section = _format_known_columns(
            path_state.get("primary_attribute"),
            path_state.get("attribute_join_paths"),
        )

        # Anchor ambiguous-term interpretation across repair attempts: without
        # this, each reconstruction call independently re-derives things like
        # "recently" from scratch and silently drifts (e.g. 5 months → 4
        # months) even when the time window was never the flagged problem.
        # Accumulate the FULL history (not just the last thought) so an
        # assumption made two retries ago isn't lost just because the latest
        # thought didn't happen to repeat it.
        prior_interpretation_section = ""
        if interpretation_history:
            interpretation_text = "\n".join(
                f"  {index}. {thought}"
                for index, thought in enumerate(interpretation_history, 1)
            )
            prior_interpretation_section = (
                "\nINTERPRETATION HISTORY:\n"
                f"{interpretation_text}\n\n"
                "Keep every valid assumption above. Change one only when required "
                "by the question, schema, or error, and state the change "
                "explicitly. Omission does not remove an assumption. Do not treat "
                "SQL implementation choices—such as joins, columns, aliases, or "
                "query structure—as fixed. Restate all active assumptions in "
                "`thought`; revise only parts contradicted by the question, "
                "available schema, or validation error.\n\n"
            )

        error_prompt = (
            "The following SQL contains an ERROR:\n\n"
            f"```sql\n{sql_code}\n```\n\n"
            f"Validation failed with the following message:\n{error}\n\n"
            f"{history_section}"
            f"{prior_interpretation_section}"
            "Please correct the SQL. Do not return the same SQL — "
            "it is invalid.\n"
            "Fix what the error requires while preserving every still-valid "
            "user-intent assumption. Do not silently reinterpret an ambiguous "
            "term merely because you're rewriting the query. In `thought`, "
            "restate the assumptions that remain valid and clearly state any "
            "assumption that had to change because it conflicted with the "
            "user's question, the available schema, or the validation error.\n"
            "Do not explain how you corrected the sql, like you were "
            "never wrong.\n"
            f"{tables_section}"
            f"{known_columns_section}"
            f"The user's question was:\n{question_block}\n"
            "You must include corrected sql in your final answer.\n"
            "Follow the rules defined in the previous messages for "
            "writing the final answer."
        )

        messages = list(messages)
        if evidence:
            messages.append(
                SystemMessage(content=format_authoritative_evidence(evidence))
            )
        messages.append(HumanMessage(content=error_prompt))

        response = invoke_with_structured_output(llm, messages, SQLGenerationModel)

        if response is None:
            self.logger.warning(
                "SQL reconstruction returned None — marking unconstructable"
            )
            return {
                "decision": "unconstructable",
                "path_state": path_state,
            }

        sql_preview = (getattr(response, "sql_code", "") or "")[:300]
        self.logger.info("SQL reconstructed: %s", sql_preview)

        thought = getattr(response, "thought", "No explanation")
        response_explanation = getattr(response, "response", "") or thought
        self.logger.info(
            "Reconstruction explanation: %s...",
            response_explanation[:100],
        )
        if thought and thought != "No explanation":
            record_thought(path_state, _GRAPH_NODE_NAME, thought)

        custom_analyses_used: list = []
        if hasattr(response, "custom_analyses_used"):
            custom_analyses_used = get_custom_analyses_ids(
                response.custom_analyses_used
            )

        return {
            "messages": messages,
            "path_state": {
                **path_state,
                "sql_generation_result": response,
                "relevant_tables": relevant_tables,
                "custom_analyses_used": custom_analyses_used,
            },
        }
