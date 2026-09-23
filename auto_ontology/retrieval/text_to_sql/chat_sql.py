# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the SQL a chat turn issues: logged, and capped in time.

Both behaviours deliberately live here rather than in the connectors. A
connector's ``execute`` is the chokepoint for *all* SQL — ingestion
introspection and column profiling included — where neither behaviour is
wanted: logging there buries a question's few statements under hundreds of
housekeeping queries, and a short timeout would break metadata scans over a
large ``information_schema``, which legitimately take minutes.

Only agent-issued SQL reaches the calls here.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# A user is waiting on a chat answer and the pipeline allows one stream at a
# time, so a runaway query would pin the slot. Databricks enforces this server
# side and cancels the statement; connect and warehouse scheduling sit outside
# the cap, so expect roughly this plus a couple of seconds of wall clock.
CHAT_STATEMENT_TIMEOUT_S = 30

# Generated SQL is normally short, but a pathological statement shouldn't flood
# the log; keep enough to be recognisable and say how much was cut.
_MAX_LOGGED_SQL_CHARS = 4000


def truncate_sql(sql: str) -> str:
    """Return *sql* capped at a length that is safe to log."""
    if len(sql) <= _MAX_LOGGED_SQL_CHARS:
        return sql
    omitted = len(sql) - _MAX_LOGGED_SQL_CHARS
    return f"{sql[:_MAX_LOGGED_SQL_CHARS]}… [{omitted} more chars]"


def log_chat_sql(connector: Any, sql: str, *, kind: str = "chat SQL") -> None:
    """Log one agent-issued statement at INFO, with the credential in use.

    ``auth_mode`` is only exposed by connectors that can run as the calling user
    (currently Databricks); for the rest the credential is fixed and the clause
    is left off rather than stating something misleading.
    """
    database = getattr(connector, "database_name", None) or "unknown"
    auth_mode = getattr(connector, "auth_mode", None)

    if auth_mode:
        logger.info(
            "[%s] Executing %s as %s:\n%s", database, kind, auth_mode, truncate_sql(sql)
        )
    else:
        logger.info("[%s] Executing %s:\n%s", database, kind, truncate_sql(sql))


def execute_chat_sql(
    connector: Any,
    sql: str,
    *,
    kind: str = "chat SQL",
    timeout_s: int | None = CHAT_STATEMENT_TIMEOUT_S,
) -> pd.DataFrame:
    """Log and run one agent-issued statement, capped in time where supported.

    Connectors that don't advertise ``supports_statement_timeout`` are called
    unchanged, so adding a cap here never breaks a connector that has no way to
    honour it.
    """
    log_chat_sql(connector, sql, kind=kind)

    if timeout_s is not None and getattr(
        connector, "supports_statement_timeout", False
    ):
        return connector.execute(sql, timeout_s=timeout_s)
    return connector.execute(sql)
