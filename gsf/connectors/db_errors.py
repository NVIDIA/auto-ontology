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


def _text(error: BaseException | str) -> str:
    return (error if isinstance(error, str) else str(error)).lower()


def is_session_lost(error: BaseException | str) -> bool:
    """Whether *error* is the server rejecting a session we think is still open.

    Callers use this to reopen and retry once. It deliberately does not cover
    dead sockets: those raise recognisable transport exceptions and are caught
    by type instead.
    """
    return any(marker in _text(error) for marker in _SESSION_LOST_MARKERS)


def is_infrastructure_error(error: BaseException | str) -> bool:
    """Whether *error* is about reaching the database, not about the SQL.

    True for a lost session or a connection that could not be established.
    False for anything a different query might fix -- an unknown column, a bad
    join, a syntax error -- so a caller that retries on False keeps retrying
    exactly the cases where retrying has a point.
    """
    text = _text(error)
    return is_session_lost(text) or any(
        marker in text for marker in _CONNECTIVITY_MARKERS
    )


__all__ = ["is_infrastructure_error", "is_session_lost"]
