# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Knobs for the ReAct text-to-SQL agent.

Every limit here bounds an episode, which is what makes the agent both safe to
serve and usable as an RL environment: a policy that can spend unbounded turns
or unbounded context has no reward signal worth learning from. All are
environment-driven.
"""

from __future__ import annotations

import os

_TRUTHY = {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def max_tool_calls() -> int:
    """Tool calls an episode may make before it has to answer.

    Sized for search, a follow-up search, a value check, a draft and a fix,
    with room for one more of each.
    """
    return _int_env("TEXT_TO_SQL_REACT_MAX_TOOL_CALLS", 24)


def model_retries() -> int:
    """Attempts at one model call before the episode gives up on it.

    A saturated endpoint answers 429 for much longer than a few seconds, and
    losing the episode there loses every search and check made so far. Six
    attempts wait 2, 4, 8, 16 and 32 seconds: about a minute in all.
    """
    return _int_env("TEXT_TO_SQL_REACT_MODEL_RETRIES", 6)


def max_sql_runs() -> int:
    """``run_sql`` calls allowed per episode, submissions excluded."""
    return _int_env("TEXT_TO_SQL_REACT_MAX_SQL_RUNS", 10)


def max_observation_chars() -> int:
    """Ceiling on one tool result as it goes back into the conversation.

    A 10,000-column table cannot be shown, and a result that overflows the
    context window loses the question outright rather than answering it badly.
    """
    return _int_env("TEXT_TO_SQL_REACT_MAX_OBSERVATION_CHARS", 8000)


def max_result_rows() -> int:
    """Rows ``run_sql`` shows back to the model."""
    return _int_env("TEXT_TO_SQL_REACT_MAX_RESULT_ROWS", 100)


def max_tables_per_retrieve() -> int:
    """Tables one ``retrieve`` call may introduce.

    Bounded so a vague query returns the best few and the model narrows with a
    second query, instead of one call flooding the context with every table
    that mentions a word.
    """
    return _int_env("TEXT_TO_SQL_REACT_MAX_TABLES_PER_RETRIEVE", 6)


def siblings_enabled() -> bool:
    """Whether ``retrieve`` names the near-copies of each table it shows.

    On by default; ``TEXT_TO_SQL_REACT_SIBLINGS=false`` turns it off so the
    two can be compared on the same questions.
    """
    value = os.environ.get("TEXT_TO_SQL_REACT_SIBLINGS", "").strip().lower()
    return value in _TRUTHY or not value


def submit_tool_enabled() -> bool:
    """Whether the model answers SQL with a separate ``submit`` tool.

    On by default; ``TEXT_TO_SQL_REACT_SUBMIT_TOOL=false`` drops it, and the
    model answers with ``run_sql(final=true)`` instead.
    """
    value = os.environ.get("TEXT_TO_SQL_REACT_SUBMIT_TOOL", "").strip().lower()
    return value in _TRUTHY or not value


def grep_enabled() -> bool:
    """Whether a ``run_sql`` result too long to show is kept for ``grep``.

    Off by default; ``TEXT_TO_SQL_REACT_GREP=true`` saves such a result as a
    page and gives the model a ``grep`` tool to search it.
    """
    return os.environ.get("TEXT_TO_SQL_REACT_GREP", "").strip().lower() in _TRUTHY


def max_page_rows() -> int:
    """Rows of one result kept for ``grep``; the rest are dropped."""
    return _int_env("TEXT_TO_SQL_REACT_MAX_PAGE_ROWS", 100_000)


def max_columns_per_table() -> int:
    """Columns shown per table when the table is wider than this.

    Wider tables show the columns the search matched, the key and the columns
    that reference other tables; the rest are reachable by a narrower query.
    """
    return _int_env("TEXT_TO_SQL_REACT_MAX_COLUMNS_PER_TABLE", 25)


def prediction_enabled() -> bool:
    """Whether this deployment can answer with a KumoRFM prediction.

    ``KUMO_RFM_API_URL`` must name a Universal TFM NIM serving the
    ``kumo-relational`` model; without it the ``predict`` tool is never offered.
    """
    return bool(os.environ.get("KUMO_RFM_API_URL", "").strip())


def max_predict_calls() -> int:
    """``predict`` calls allowed per episode.

    Each one reads the tables and builds a graph, which takes far longer than
    a search, so a refused build gets one retry after retrieving what it lacked.
    """
    return _int_env("TEXT_TO_SQL_REACT_MAX_PREDICT_CALLS", 2)
