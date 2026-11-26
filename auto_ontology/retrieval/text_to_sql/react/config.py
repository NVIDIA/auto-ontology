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


#: Spelled once because the conversation ceiling has to ask whether the
#: general one was set before it can decide to stand aside for it.
_OBSERVATION_CAP = "TEXT_TO_SQL_REACT_MAX_OBSERVATION_CHARS"
_CONVERSATION_CAP = "TEXT_TO_SQL_REACT_MAX_CONVERSATION_OBSERVATION_CHARS"


def max_observation_chars() -> int:
    """Ceiling on one tool result as it goes back into the conversation.

    A 10,000-column table cannot be shown, and a result that overflows the
    context window loses the question outright rather than answering it badly.
    """
    return _int_env(_OBSERVATION_CAP, 8000)


def max_conversation_observation_chars() -> int:
    """Ceiling on one conversation search, which needs more room than a table.

    A catalog hit is a table's name and its columns; a transcript hit is a
    question and an answer, often with a pasted code block, and averages some
    4,700 characters. Three of them pass the general ceiling on their own, so
    one number for both was always going to be wrong for one of them. Measured
    on BEAM: the general default answers 0.828 and 24,000 answers 0.868.

    A caller that set the general ceiling meant it -- a smaller context window
    is the usual reason -- so that setting still decides. This one applies
    only when nothing was said.
    """
    if os.environ.get(_CONVERSATION_CAP, "").strip():
        return _int_env(_CONVERSATION_CAP, 24000)
    if os.environ.get(_OBSERVATION_CAP, "").strip():
        return max_observation_chars()
    return 24000


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


def max_sibling_counts() -> int:
    """Row counts one ``retrieve`` call may run for its SIMILAR lines.

    An exact count is a full scan on most engines and the number is only a
    hint, so a table past the limit is named without one.
    """
    return _int_env("TEXT_TO_SQL_REACT_MAX_SIBLING_COUNTS", 4)


def sibling_count_timeout_s() -> int:
    """Seconds one SIMILAR-line row count may run before it is dropped."""
    return _int_env("TEXT_TO_SQL_REACT_SIBLING_COUNT_TIMEOUT_S", 5)


def submit_tool_enabled() -> bool:
    """Whether the model answers SQL with a separate ``submit`` tool.

    On by default; ``TEXT_TO_SQL_REACT_SUBMIT_TOOL=false`` drops it, and the
    model answers with ``run_sql(final=true)`` instead.
    """
    value = os.environ.get("TEXT_TO_SQL_REACT_SUBMIT_TOOL", "").strip().lower()
    return value in _TRUTHY or not value


def fanout_check_enabled() -> bool:
    """Whether fanout profiling, refusals and advisory notes are enabled.

    Off by default; ``TEXT_TO_SQL_REACT_FANOUT_CHECK=true`` turns all three on
    so deployments can opt in or measure an evaluation arm.
    """
    value = os.environ.get("TEXT_TO_SQL_REACT_FANOUT_CHECK", "").strip().lower()
    return value in _TRUTHY


def empty_result_check_enabled() -> bool:
    """Whether empty-result filter probes and their retry are enabled.

    Off by default; ``TEXT_TO_SQL_REACT_EMPTY_RESULT_CHECK=true`` turns on
    probing and may consume another model turn.
    """
    value = os.environ.get("TEXT_TO_SQL_REACT_EMPTY_RESULT_CHECK", "").strip().lower()
    return value in _TRUTHY


def grep_enabled() -> bool:
    """Whether a ``run_sql`` result too long to show is kept for ``grep``.

    Off by default. ``TEXT_TO_SQL_REACT_GREP=true`` turns it on: such a result
    is then saved as a page and the model gets a ``grep`` tool to search it.
    """
    value = os.environ.get("TEXT_TO_SQL_REACT_GREP", "").strip().lower()
    return value in _TRUTHY


def ask_user_enabled() -> bool:
    """Whether an episode whose caller can reach the user offers ``ask_user``.

    Off by default. ``TEXT_TO_SQL_REACT_ASK_USER=true`` turns it on; the tool
    is still offered only when the caller sets ``ToolContext.ask_user``.
    """
    value = os.environ.get("TEXT_TO_SQL_REACT_ASK_USER", "").strip().lower()
    return value in _TRUTHY


def max_page_rows() -> int:
    """Rows of one result kept for ``grep``; the rest are dropped."""
    return _int_env("TEXT_TO_SQL_REACT_MAX_PAGE_ROWS", 100_000)


def grep_timeout_s() -> float:
    """Seconds one ``grep`` call may spend matching its pattern against a page.

    The pattern is written by the model, and a backtracking one can run for
    minutes on a long row, so the whole search shares one budget.
    """
    return float(_int_env("TEXT_TO_SQL_REACT_GREP_TIMEOUT_S", 5))


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
