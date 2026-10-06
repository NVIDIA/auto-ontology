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

from auto_ontology.infra.feature_flags import get_sql_query_timeout_seconds

logger = logging.getLogger(__name__)

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
    timeout_s: float | None = None,
) -> pd.DataFrame:
    """Log and run one agent-issued statement, capped in time.

    A user is waiting on the answer and the pipeline allows one stream at a
    time, so a runaway query would pin the slot. The cap is *timeout_s*, or
    when that is ``None`` the SQL Query Timeout from Settings > Agent Settings
    (30s unless an admin changed it, and always 30s without the frontend).
    Connecting and warehouse scheduling may sit outside it, so expect roughly
    the cap plus a couple of seconds of wall clock.

    Every built-in connector honours the cap -- server side where the engine
    has a statement timeout, client side for DuckDB, SQLite and HeavyDB. One
    that doesn't advertise ``supports_statement_timeout`` is called unchanged,
    so the cap never breaks a connector with no way to honour it.
    """
    log_chat_sql(connector, sql, kind=kind)

    if not getattr(connector, "supports_statement_timeout", False):
        return connector.execute(sql)
    if timeout_s is None:
        timeout_s = get_sql_query_timeout_seconds()
    return connector.execute(sql, timeout_s=timeout_s)
