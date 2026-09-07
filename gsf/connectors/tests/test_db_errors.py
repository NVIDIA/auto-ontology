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


# ----------------------------------------------------------------------
# Markers echoed back inside the caller's own statement
# ----------------------------------------------------------------------


def test_a_marker_in_the_echoed_statement_is_not_a_diagnosis() -> None:
    """Spark pastes the statement it rejected after ``== SQL ==``.

    A filter on the literal 'connection reset' would otherwise be read as a
    verdict on the connection, abandoning a query a rewrite could have fixed.
    """
    statement = "SELECT * FROM logs WHERE error_msg = 'connection reset'"
    error = (
        "org.apache.spark.sql.catalyst.parser.ParseException: "
        "[PARSE_SYNTAX_ERROR] Syntax error at or near 'FORM'\n"
        f"== SQL ==\n{statement}"
    )

    assert not is_infrastructure_error(error, statement)
    assert not is_infrastructure_error(error), "the delimiter alone should be enough"


def test_a_marker_in_an_inline_echo_is_not_a_diagnosis() -> None:
    """Kyuubi often inlines the statement instead of using the delimiter."""
    statement = "SELECT id FROM t WHERE msg = 'Invalid SessionHandle'"
    error = f"Error operating ExecuteStatement: cannot resolve 'msg' in {statement}"

    assert not is_session_lost(error, statement)
    assert not is_infrastructure_error(error, statement)


def test_a_real_diagnosis_survives_an_echoed_statement() -> None:
    """Stripping the echo must not hide what the server said on its own behalf."""
    statement = "SELECT activity_type FROM nvdp.nvdp_telemetry_bot.tbl_CodeActivity"
    error = f"Invalid SessionHandle: deadbeef (while running: {statement})"

    assert is_session_lost(error, statement)
    assert is_infrastructure_error(error, statement)


def test_a_marker_in_a_re_rendered_plan_is_not_a_diagnosis() -> None:
    """Spark answers an analysis failure with the plan, not the SQL.

    There is no ``== SQL ==`` delimiter and the literal loses its quotes, so
    nothing in the message matches the statement verbatim -- but the phrase is
    still the caller's own text, and the query is still worth rewriting.
    """
    statement = "SELECT * FROM logs WHERE msg = 'connection reset'"
    error = (
        "org.apache.spark.sql.AnalysisException: cannot resolve 'activity_typ' "
        "given input columns: [msg, ts]; line 1 pos 7;\n"
        "'Project [*]\n"
        "+- 'Filter (msg = connection reset)\n"
        "   +- 'UnresolvedRelation [logs]"
    )

    assert not is_infrastructure_error(error, statement)
    assert not is_session_lost(error, statement)


def test_a_re_indented_echo_still_strips() -> None:
    """Engines re-wrap the statement they echo; collapsed forms still match."""
    statement = "SELECT id\n  FROM t\n WHERE msg = 'broken pipe'"
    error = (
        "ParseException: Syntax error at or near 'FORM'\n"
        "== SQL ==\n"
        "SELECT id FROM t WHERE msg = 'broken pipe'"
    )

    assert not is_infrastructure_error(error, statement)


def test_a_real_diagnosis_survives_a_re_rendered_plan() -> None:
    """Stripping literals must not swallow the server's own verdict."""
    statement = "SELECT * FROM logs WHERE msg = 'connection reset'"
    error = f"Invalid SessionHandle: deadbeef (while running: {statement})"

    assert is_session_lost(error, statement)
    assert is_infrastructure_error(error, statement)


def test_stripping_a_literal_cannot_splice_a_marker_into_existence() -> None:
    """Fragments are replaced with a space, never deleted outright."""
    statement = "SELECT 'ion ref' FROM t"
    error = "cannot resolve 'connect'ion ref'used'"

    assert not is_infrastructure_error(error, statement)
