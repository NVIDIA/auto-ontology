# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Empty-result LIKE check after SQL execution.

When execution succeeds but returns an empty result set and the SQL contains
LIKE/ILIKE predicates, use an LLM to identify non-essential filters and route
once to reconstruction with explicit removal guidance.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from gsf.retrieval.text_to_sql.base import BaseAgent
from gsf.retrieval.text_to_sql.prompts import (
    create_empty_like_check_prompt,
    format_dual_question_block,
)
from gsf.retrieval.text_to_sql.state import (
    AgentState,
    get_original_question,
    get_question_for_processing,
)
from gsf.utils.llm_invoke import invoke_with_structured_output

logger = logging.getLogger(__name__)

_LIKE_PATTERN = re.compile(r"\bI?LIKE\b", re.IGNORECASE)

_EMPTY_LIKE_SYSTEM_PROMPT = """You classify LIKE/ILIKE predicates in SQL queries.
Be conservative: only mark a predicate non-essential when it clearly filters on
a feature, preference, or additional attribute rather than the main subject."""


class EmptyLikeCheckModel(BaseModel):
    """LLM classification of LIKE/ILIKE predicates after an empty result set."""

    model_config = ConfigDict(extra="forbid")

    non_essential_like_predicates: list[str] = Field(
        default_factory=list,
        description=(
            "LIKE/ILIKE predicates to remove or relax. Each entry should quote "
            "the full predicate exactly as it appears in the SQL."
        ),
    )
    essential_like_predicates: list[str] = Field(
        default_factory=list,
        description=(
            "LIKE/ILIKE predicates that identify the main subject and must be "
            "preserved. Each entry should quote the full predicate exactly as "
            "it appears in the SQL."
        ),
    )


def _is_empty_db_result(result: Any) -> bool:
    if result is None:
        return False
    if isinstance(result, list):
        if not result:
            return True
        if len(result) == 1 and isinstance(result[0], str):
            try:
                parsed = json.loads(result[0])
                return isinstance(parsed, list) and len(parsed) == 0
            except json.JSONDecodeError:
                return result[0].strip() in ("[]", "")
        return all(_is_empty_db_result(item) for item in result)
    if isinstance(result, str):
        stripped = result.strip()
        if stripped in ("[]", ""):
            return True
        try:
            parsed = json.loads(stripped)
            return isinstance(parsed, list) and len(parsed) == 0
        except json.JSONDecodeError:
            return False
    return False


def _sql_has_like(sql: str) -> bool:
    return bool(_LIKE_PATTERN.search(sql or ""))


def _get_sql_code(path_state: dict) -> str:
    sql_code = path_state.get("sql_code")
    if sql_code and str(sql_code).strip():
        return str(sql_code)
    llm_result = path_state.get("sql_generation_result")
    return getattr(llm_result, "sql_code", "") or ""


def build_empty_like_reconstruction_error(
    essential: list[str],
    non_essential: list[str],
) -> str:
    essential_lines = "\n".join(f"- {item}" for item in essential) or "- (none)"
    non_essential_lines = "\n".join(f"- {item}" for item in non_essential)
    return (
        "SQL executed successfully but returned an empty result set. The SQL "
        "contains LIKE/ILIKE filters. Reconstruct the SQL by removing or "
        "relaxing ONLY the non-essential LIKE/ILIKE predicates listed below. "
        "Never remove predicates that identify the main subject of the question. "
        "Do not remove joins, numeric thresholds, or other essential filters.\n\n"
        "Remove these non-essential LIKE/ILIKE predicates:\n"
        f"{non_essential_lines}\n\n"
        "Preserve these essential LIKE/ILIKE predicates:\n"
        f"{essential_lines}"
    )


class EmptyLikeResultCheckAgent(BaseAgent):
    """Use LLM to detect removable LIKE filters after an empty execution result."""

    def __init__(self) -> None:
        super().__init__("empty_like_result_check")

    def validate_input(self, state: AgentState) -> bool:
        path_state = state.get("path_state", {})
        if path_state.get("sql_response_from_db") is None:
            self.logger.warning("No SQL execution result found for empty LIKE check")
            return False
        return True

    def execute(self, state: AgentState) -> dict[str, Any]:
        path_state = dict(state.get("path_state", {}))
        sql_code = _get_sql_code(path_state)
        db_result = path_state.get("sql_response_from_db")

        if not _is_empty_db_result(db_result):
            self.logger.info("SQL result is non-empty — skipping empty LIKE check")
            return {"decision": "valid_sql", "path_state": path_state}

        if not _sql_has_like(sql_code):
            self.logger.info("Empty result without LIKE/ILIKE — passing through")
            return {"decision": "valid_sql", "path_state": path_state}

        if path_state.get("empty_like_retry_attempted"):
            self.logger.info(
                "Empty LIKE retry already attempted — passing through empty result"
            )
            return {"decision": "valid_sql", "path_state": path_state}

        original_question = get_original_question(state)
        sanitized_question = get_question_for_processing(state)
        question_block = format_dual_question_block(
            original_question, sanitized_question
        )

        prompt = create_empty_like_check_prompt(question_block, sql_code)
        messages = [
            SystemMessage(content=_EMPTY_LIKE_SYSTEM_PROMPT),
            SystemMessage(content=prompt),
        ]

        try:
            result = invoke_with_structured_output(
                state["llm"], messages, EmptyLikeCheckModel
            )
        except Exception as exc:
            self.logger.warning("Empty LIKE LLM check failed: %s", exc)
            return {"decision": "valid_sql", "path_state": path_state}

        if result is None:
            self.logger.warning("Empty LIKE LLM check returned None")
            return {"decision": "valid_sql", "path_state": path_state}

        non_essential = [
            item.strip()
            for item in result.non_essential_like_predicates
            if item.strip()
        ]
        essential = [
            item.strip() for item in result.essential_like_predicates if item.strip()
        ]

        if not non_essential:
            self.logger.info(
                "LLM found no non-essential LIKE/ILIKE predicates — passing through"
            )
            return {"decision": "valid_sql", "path_state": path_state}

        path_state["empty_like_retry_attempted"] = True
        path_state["error"] = build_empty_like_reconstruction_error(
            essential, non_essential
        )
        self.logger.info(
            "Empty result — routing to reconstruction to remove %d non-essential "
            "LIKE/ILIKE predicate(s)",
            len(non_essential),
        )
        return {"decision": "invalid_sql", "path_state": path_state}
