# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Driver errors are classified by whether a different query could fix them."""

from __future__ import annotations

import pytest

from gsf.connectors.db_errors import is_infrastructure_error, is_session_lost

# Verbatim shapes seen from Thrift-based engines. The handle error arrives
# wrapped in the driver's response repr, so matching has to survive that.
_SESSION_ERRORS = [
    "Invalid SessionHandle: 1f9c2c1e-0000-4c1e-9f00-2b0d5a7c9e11",
    "TExecuteStatementResp(status=TStatus(statusCode=3, "
    "errorMessage='Invalid SessionHandle: abc123'))",
    "Invalid session handle",
    "org.apache.kyuubi.KyuubiSQLException: Session does not exist",
    "The session is closed",
]

_CONNECTIVITY_ERRORS = [
    "[Errno 61] Connection refused",
    "Connection reset by peer",
    "TTransportException: Could not connect to hive.pdx-aws.data.nvidia.com:10000",
    "No connector available to execute SQL.",
    "Temporary failure in name resolution",
]

# These are the whole point of the distinction: each one is plausibly fixed by
# generating different SQL, so they must stay eligible for reconstruction.
_SQL_ERRORS = [
    "Table or view not found: nvdp.nvdp_telemetry_bot.tbl_Nope",
    "cannot resolve 'activity_typ' given input columns: [activity_type, ts]",
    "[PARSE_SYNTAX_ERROR] Syntax error at or near 'GROUP'",
    "AMBIGUOUS_REFERENCE: Reference 'id' is ambiguous",
    "Cannot cast STRING to TIMESTAMP",
    "PERMISSION_DENIED: User does not have SELECT on table raw.events",
]


@pytest.mark.parametrize("message", _SESSION_ERRORS)
def test_session_loss_is_recognised(message: str) -> None:
    assert is_session_lost(message)
    assert is_infrastructure_error(message)


@pytest.mark.parametrize("message", _CONNECTIVITY_ERRORS)
def test_connectivity_failures_are_infrastructure_but_not_session_loss(
    message: str,
) -> None:
    """Only a lost session is worth reopening; the rest are surfaced as-is."""
    assert is_infrastructure_error(message)
    assert not is_session_lost(message)


@pytest.mark.parametrize("message", _SQL_ERRORS)
def test_sql_errors_stay_reconstructable(message: str) -> None:
    assert not is_infrastructure_error(message)
    assert not is_session_lost(message)


def test_accepts_an_exception_as_well_as_a_string() -> None:
    """Connectors hold the exception; the graph only keeps ``str(exc)``."""
    error = RuntimeError("Invalid SessionHandle: deadbeef")

    assert is_session_lost(error)
    assert is_infrastructure_error(error)
