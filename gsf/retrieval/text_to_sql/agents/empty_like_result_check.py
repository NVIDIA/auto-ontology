# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Empty-result LIKE check after SQL execution.

When execution succeeds but returns an empty result set and the SQL contains
LIKE/ILIKE predicates, deterministically rewrite it and re-execute directly
(no LLM involved):
- Remove any LIKE/ILIKE predicate that contains a word not present in the
  extracted value entities (a filter not grounded in what the user searched for).
- Relax every remaining multi-word predicate by matching each word separately
  (col ILIKE '%w1%' AND col ILIKE '%w2%' ...).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from gsf.retrieval.text_to_sql.base import BaseAgent
from gsf.retrieval.text_to_sql.state import AgentState

logger = logging.getLogger(__name__)

_LIKE_PATTERN = re.compile(r"\bI?LIKE\b", re.IGNORECASE)

# Matches a single ``<column> LIKE|ILIKE '<pattern>'`` predicate. The column may
# be a simple or dotted/quoted identifier (e.g. p.description, "My Col") or a
# function-wrapped one (e.g. LOWER(details), LOWER(p.description)).
_LIKE_PREDICATE_RE = re.compile(
    r"(?P<col>\w+\s*\(\s*[\w.\"]+\s*\)|[\w.\"]+)\s+(?P<op>I?LIKE)\s+'(?P<pat>[^']*)'",
    re.IGNORECASE,
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


def _pattern_words(pattern: str) -> list[str]:
    """Return the non-wildcard words of a LIKE pattern (e.g. '%a b%' -> [a, b])."""
    return [
        token.strip("%_")
        for token in re.split(r"\s+", pattern.strip())
        if token.strip("%_")
    ]


def _value_words(value_entities: list[str]) -> set[str]:
    """Lowercased set of individual words across all value entities."""
    words: set[str] = set()
    for entity in value_entities:
        for token in re.split(r"\W+", (entity or "").lower()):
            if token:
                words.add(token)
    return words


def remove_unbacked_likes(sql: str, allowed_words: set[str]) -> tuple[str, int]:
    """Remove LIKE/ILIKE predicates containing any word not in *allowed_words*.

    A predicate is kept only when EVERY one of its pattern words appears in
    *allowed_words* (the value-entity vocabulary); if it contains even one word
    that is not in there, the whole predicate is removed along with one adjacent
    boolean connector. A lone unbacked predicate is replaced with ``TRUE`` to
    keep the WHERE clause valid.

    Returns the rewritten SQL and the number of predicates removed.
    """
    if not allowed_words:
        return sql, 0

    removed = 0
    for match in list(_LIKE_PREDICATE_RE.finditer(sql)):
        text = match.group(0)
        words = _pattern_words(match.group("pat"))
        if words and all(word.lower() in allowed_words for word in words):
            continue  # every word is backed by a value entity — keep it

        escaped = re.escape(text)
        new_sql, hits = re.subn(
            r"\s+(?:AND|OR)\s+" + escaped, "", sql, flags=re.IGNORECASE
        )
        if hits == 0:
            new_sql, hits = re.subn(
                escaped + r"\s+(?:AND|OR)\s+", "", sql, flags=re.IGNORECASE
            )
        if hits == 0:
            new_sql = sql.replace(text, "TRUE")
            hits = 1 if new_sql != sql else 0
        if hits:
            removed += 1
            sql = new_sql

    return sql, removed


def split_multiword_likes(sql: str) -> tuple[str, int]:
    """Rewrite every multi-word LIKE/ILIKE predicate into per-word ANDed matches.

    ``col ILIKE '%empty playing box%'`` becomes
    ``(col ILIKE '%empty%' AND col ILIKE '%playing%' AND col ILIKE '%box%')``.
    Single-word predicates are left untouched.

    Returns the rewritten SQL and the number of predicates that were split.
    """
    count = 0

    def _replace(match: re.Match) -> str:
        nonlocal count
        words = _pattern_words(match.group("pat"))
        if len(words) < 2:
            return match.group(0)
        count += 1
        col = match.group("col")
        op = match.group("op")
        parts = [f"{col} {op} '%{word}%'" for word in words]
        return "(" + " AND ".join(parts) + ")"

    return _LIKE_PREDICATE_RE.sub(_replace, sql or ""), count


def _get_sql_code(path_state: dict) -> str:
    sql_code = path_state.get("sql_code")
    if sql_code and str(sql_code).strip():
        return str(sql_code)
    llm_result = path_state.get("sql_generation_result")
    return getattr(llm_result, "sql_code", "") or ""


class EmptyLikeResultCheckAgent(BaseAgent):
    """Deterministically relax multi-word LIKE filters after an empty result."""

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

        value_entities: list[str] = path_state.get("value_entities") or []
        allowed_words = _value_words(value_entities)

        # 1) Drop LIKE/ILIKE predicates not grounded in the value entities.
        rewritten_sql, removed_count = remove_unbacked_likes(sql_code, allowed_words)
        # 2) Relax the remaining multi-word predicates into per-word matches.
        rewritten_sql, split_count = split_multiword_likes(rewritten_sql)

        if removed_count == 0 and split_count == 0:
            self.logger.info(
                "No unbacked or multi-word LIKE/ILIKE predicates — passing through"
            )
            return {"decision": "valid_sql", "path_state": path_state}

        path_state["empty_like_retry_attempted"] = True
        path_state["sql_code"] = rewritten_sql
        # Keep the generation result in sync so downstream/display show the SQL
        # that actually ran.
        gen_result = path_state.get("sql_generation_result")
        if gen_result is not None and hasattr(gen_result, "sql_code"):
            try:
                gen_result.sql_code = rewritten_sql
            except Exception:  # noqa: BLE001 - best-effort sync, never fatal
                self.logger.debug("Could not sync sql_generation_result.sql_code")

        self.logger.info(
            "Empty result — removed %d unbacked and split %d multi-word LIKE/ILIKE "
            "predicate(s) in place; re-executing rewritten SQL",
            removed_count,
            split_count,
        )
        return {"decision": "re_execute", "path_state": path_state}
