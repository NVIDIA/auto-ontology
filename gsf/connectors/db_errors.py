# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classify driver errors by what the caller can actually do about them.

Drivers report "your SQL was wrong" and "the connection is gone" through the
same exception types, so telling them apart means reading the message. Two
callers need that distinction:

* a connector deciding whether to reopen its session and retry, and
* the text-to-SQL graph deciding whether rewriting the query could possibly
  help -- rewriting cannot fix a dead session, so looping on one only burns
  LLM calls before failing anyway.

Matching on message text is unpleasant but unavoidable here. HiveServer2 and
Thrift-based engines (Kyuubi, Databricks, Spark ThriftServer) all return a
retired session as a normal statement error carrying an ``Invalid
SessionHandle`` message, not as a distinct exception class or SQLSTATE.
"""

from __future__ import annotations

# The server is up and answered, but it no longer knows the session the caller
# is using -- typically because an idle Spark engine was retired behind it.
# Reopening the session and re-running the statement resolves this.
_SESSION_LOST_MARKERS = (
    "invalid sessionhandle",
    "invalid session handle",
    "session does not exist",
    "session handle does not exist",
    "session is closed",
    "session has been closed",
    "session not found",
)

# Spark reports a statement it could not parse by echoing it after this
# delimiter, and Kyuubi passes that message through untouched.
_SQL_ECHO_DELIMITER = "== sql =="

# The statement never reached a working engine. Distinct from a lost session in
# that reopening is not necessarily enough, so these are surfaced rather than
# retried -- but they are still not something a different query would fix.
_CONNECTIVITY_MARKERS = (
    "connection refused",
    "connection reset",
    "connection aborted",
    "connection timed out",
    "broken pipe",
    "could not connect",
    "failed to connect",
    "no connector available",
    "ttransportexception",
    "socket is closed",
    "socket timed out",
    "name or service not known",
    "temporary failure in name resolution",
)


def _diagnosis(error: BaseException | str, statement: str | None = None) -> str:
    """Reduce *error* to what the server said about itself, lowercased.

    The echoed statement is removed first. A marker that appears inside the SQL
    the caller sent is not a diagnosis: a filter on the literal ``'connection
    reset'`` says nothing about the connection, and reading it as one would
    abandon a query that a rewrite could have fixed. Pass *statement* whenever
    it is known -- an exact echo is far more precise to strip than a guess at
    where the message ends.
    """
    text = (error if isinstance(error, str) else str(error)).lower()
    text = text.split(_SQL_ECHO_DELIMITER, 1)[0]
    if statement:
        text = text.replace(statement.strip().lower(), " ")
    return text


def is_session_lost(error: BaseException | str, statement: str | None = None) -> bool:
    """Whether *error* is the server rejecting a session we think is still open.

    Callers use this to reopen and retry once. It deliberately does not cover
    dead sockets: those raise recognisable transport exceptions and are caught
    by type instead.
    """
    text = _diagnosis(error, statement)
    return any(marker in text for marker in _SESSION_LOST_MARKERS)


def is_infrastructure_error(
    error: BaseException | str, statement: str | None = None
) -> bool:
    """Whether *error* is about reaching the database, not about the SQL.

    True for a lost session or a connection that could not be established.
    False for anything a different query might fix -- an unknown column, a bad
    join, a syntax error -- so a caller that retries on False keeps retrying
    exactly the cases where retrying has a point.
    """
    text = _diagnosis(error, statement)
    return any(
        marker in text for marker in _SESSION_LOST_MARKERS + _CONNECTIVITY_MARKERS
    )


__all__ = ["is_infrastructure_error", "is_session_lost"]
