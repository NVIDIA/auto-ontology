# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a ClickHouse connection's chat SQL as the signed-in user.

Opt-in per connection ("Run chat queries as the signed-in user" in Settings →
Connections, stored as ``execute_as_user``). The connection's stored user
stays the one that authenticates; chat SQL is sent as ``EXECUTE AS <user>
<query>``, where ``<user>`` is the local part of the signed-in user's email, so
the query runs under that ClickHouse user's grants. Ingestion, profiling and
semantic compilation are unaffected and keep running as the stored user, which
is why it has to be able to see everything.

Fail-closed throughout, like :mod:`auto_ontology.connectors.databricks_oauth`: a
flagged connection never falls back to the stored user's privileges. The chat
endpoint refuses a request whose user cannot be resolved, and the worker
re-reads the flag for every question (:func:`apply_execute_as`) rather than
trusting a connector built before the flag was turned on.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Mapping

from auto_ontology.connectors.base import SQLDatabase

logger = logging.getLogger(__name__)

# Kept to what an email local part realistically holds. Anything else -- a
# quote, a backtick, whitespace -- is refused rather than escaped: there is no
# ClickHouse user named like that to run as, and refusing keeps the identifier
# that reaches ``EXECUTE AS`` boring.
_USERNAME = re.compile(r"^[A-Za-z0-9._+-]+$")

# any_connection_uses_execute_as_user() reads the catalog (and Vault) on every
# call; cache the answer briefly so the chat hot path stays cheap. Toggling the
# flag invalidates it, so the TTL only bounds staleness across processes.
_FLAG_TTL_S = 30.0
_flag_result: bool | None = None
_flag_expires: float = 0.0
_flag_lock = threading.Lock()


class ExecuteAsError(RuntimeError):
    """A flagged connection could not be bound to a signed-in user."""


def uses_execute_as_user(connection: Mapping[str, Any]) -> bool:
    """Whether *connection* runs chat queries as the signed-in user.

    JSON round-trips give a real bool, but string forms are accepted so
    hand-edited connections behave.
    """
    value = connection.get("execute_as_user")
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in ("1", "true", "yes")


def any_connection_uses_execute_as_user() -> bool:
    """Whether any configured connection runs chat queries as the signed-in user.

    Drives the chat endpoint's check: when nothing is flagged the request needs
    no user, exactly as before this feature existed. A lookup failure answers
    True, so a storage blip refuses chat instead of letting it run as the
    stored user.
    """
    global _flag_result, _flag_expires

    now = time.monotonic()
    with _flag_lock:
        if _flag_result is not None and now < _flag_expires:
            return _flag_result

    try:
        from auto_ontology.dal.connections import list_connections

        result = any(uses_execute_as_user(conn) for conn in list_connections())
    except Exception:
        logger.exception("Failed to check connections for execute-as-user")
        # Not cached: the next request should look again rather than stay
        # refused for a whole TTL after the store comes back.
        return True

    with _flag_lock:
        _flag_result = result
        _flag_expires = now + _FLAG_TTL_S
    return result


def invalidate_execute_as_cache() -> None:
    """Force the next call to re-query. Call after toggling the flag."""
    global _flag_result
    with _flag_lock:
        _flag_result = None


def username_from_email(email: str) -> str:
    """Return the ClickHouse user to run as for *email*: its local part, as written.

    ``alice.smith@corp.com`` → ``alice.smith``. Case is kept, since ClickHouse
    user names are case-sensitive.

    Raises:
        ValueError: *email* has no usable local part.
    """
    local, at, _domain = email.strip().rpartition("@")
    if not at or not _USERNAME.fullmatch(local):
        raise ValueError(f"Cannot derive a ClickHouse user name from {email!r}")
    return local


def apply_execute_as(
    connectors: list[SQLDatabase], username: str | None
) -> list[SQLDatabase]:
    """Swap every flagged connector in *connectors* for one running as *username*.

    The flag is read from the store on every call rather than taken from the
    connectors: a chat worker builds its connectors once, and one that was busy
    when the flag was turned on would otherwise keep running as the stored user.

    Raises:
        ExecuteAsError: a flagged connection is in *connectors* but there is no
            *username*, or it is not a ClickHouse connection; or the flags
            could not be read while a ClickHouse connector is present.
    """
    from auto_ontology.connectors.clickhouse import ClickHouseDatabase
    from auto_ontology.dal.connections import list_connections

    try:
        connections = list_connections()
    except Exception as exc:
        # Only a ClickHouse connector can be flagged, so without one there is
        # nothing this could have changed and no reason to take chat down.
        if not any(isinstance(c, ClickHouseDatabase) for c in connectors):
            return connectors
        raise ExecuteAsError(
            "Could not check which connections run chat queries as the signed-in user"
        ) from exc

    flagged = {
        str(conn.get("database") or "").strip()
        for conn in connections
        if uses_execute_as_user(conn)
    }
    if not flagged:
        return connectors

    applied: list[SQLDatabase] = []
    for connector in connectors:
        if connector.database_name not in flagged:
            applied.append(connector)
            continue
        if not isinstance(connector, ClickHouseDatabase):
            raise ExecuteAsError(
                f"Connection {connector.database_name!r} is set to run as the "
                "signed-in user, which only ClickHouse connections support"
            )
        if not username:
            raise ExecuteAsError(
                f"Connection {connector.database_name!r} runs chat queries as the "
                "signed-in user, but the request carried no user"
            )
        applied.append(connector.impersonating(username))
    return applied
