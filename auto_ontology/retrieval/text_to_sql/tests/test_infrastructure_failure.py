# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An unreachable database fails once, with a message, instead of looping.

Rewriting SQL cannot reach a database that is down, so the reconstruction loop
would spend every attempt re-issuing statements that fail identically and give
up anyway -- and, before this, give up silently.
"""

from __future__ import annotations

from typing import Any

import pytest

from auto_ontology.retrieval.text_to_sql.agents.sql_execution import (
    _INFRASTRUCTURE_MESSAGE,
    QueryResponse,
    SQLExecutionAgent,
)
from auto_ontology.retrieval.text_to_sql.agents.sql_unconstructable import (
    SQLUnconstructableAgent,
)
from auto_ontology.retrieval.text_to_sql.text_to_sql_graph import (
    route_sql_execution,
    route_sql_validation,
)

_SESSION_LOST = "Invalid SessionHandle: 1f9c2c1e-0000-4c1e-9f00-2b0d5a7c9e11"
_BAD_COLUMN = "cannot resolve 'activity_typ' given input columns: [activity_type]"


def _state(**overrides: Any) -> dict:
    state = {
        "path_state": {"sql_code": "SELECT 1", "relevant_tables": []},
        "connectors": [],
        "messages": [],
        "decision": "",
    }
    state.update(overrides)
    return state


def _run_with_error(monkeypatch: pytest.MonkeyPatch, error: str) -> dict:
    monkeypatch.setattr(
        "auto_ontology.retrieval.text_to_sql.agents.sql_execution._run_sql",
        lambda sql, connector: QueryResponse(result=None, sliced=False, error=error),
    )
    return SQLExecutionAgent().execute(_state())


def test_unreachable_database_gives_up_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_with_error(monkeypatch, _SESSION_LOST)

    assert result["decision"] == "unconstructable"
    assert result["path_state"]["unconstructable_explanation"] == (
        _INFRASTRUCTURE_MESSAGE
    )
    # Kept for the logs even though it is not shown to the user.
    assert result["path_state"]["error"] == _SESSION_LOST


def test_a_bad_column_still_goes_round_the_reconstruction_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fast path must not swallow errors a rewrite can genuinely fix."""
    result = _run_with_error(monkeypatch, _BAD_COLUMN)

    assert result["decision"] == "invalid_sql"
    assert "unconstructable_explanation" not in result["path_state"]


def test_router_honours_a_decision_to_give_up() -> None:
    """Without the pass-through, "not invalid_sql" was read as usable SQL."""
    state = _state(decision="unconstructable", path_state={})

    assert route_sql_validation(state) == "unconstructable"


def test_router_still_retries_invalid_sql() -> None:
    state = _state(decision="invalid_sql", path_state={"failed_attempts": []})

    assert route_sql_validation(state) == "invalid_sql"


def test_short_answer_execution_skips_empty_like_check() -> None:
    state = _state(decision="valid_sql", shorten_answer=True)

    assert route_sql_execution(state) == "valid_sql_without_like_check"


def test_regular_execution_keeps_empty_like_check() -> None:
    state = _state(decision="valid_sql", shorten_answer=False)

    assert route_sql_execution(state) == "valid_sql"


def test_router_gives_up_past_the_attempt_limit() -> None:
    """Counts above the limit must route, not fall through returning None."""
    for attempts in (8, 9, 20):
        state = _state(
            decision="invalid_sql",
            path_state={"failed_attempts": [{} for _ in range(attempts)]},
        )

        assert route_sql_validation(state) == "unconstructable"


def test_the_explanation_reaches_the_final_response() -> None:
    """``_extract_answer`` reads ``final_response``; ``messages`` was a dead end."""
    state = _state(
        path_state={"unconstructable_explanation": _INFRASTRUCTURE_MESSAGE},
    )

    result = SQLUnconstructableAgent().execute(state)

    final = result["path_state"]["final_response"]
    assert final["response"] == _INFRASTRUCTURE_MESSAGE
    # A non-empty response is what makes the server persist the turn at all.
    assert final["response"].strip()
    assert isinstance(result["messages"], list), "a dict here replaces the list"
    assert result["messages"][-1].content == _INFRASTRUCTURE_MESSAGE


def test_falls_back_to_a_generic_explanation() -> None:
    state = _state(path_state={})

    final = SQLUnconstructableAgent().execute(state)["path_state"]["final_response"]

    assert final["response"] == "SQL can't be constructed from the data."


def test_the_loop_uses_seven_reconstructions() -> None:
    """Drive the router the way the graph does: one call per failed attempt.

    ``SQLReconstructionAgent`` appends to ``failed_attempts`` whenever the
    router sends the graph to reconstruction. Replay that state transition
    here without standing up the graph.
    """
    path_state: dict[str, Any] = {"failed_attempts": []}
    routes: list[str] = []

    # Generous bound: the budget has to stop the loop on its own. A change that
    # never returns "unconstructable" fails the assertions below rather than
    # spinning until the graph's recursion limit.
    for _ in range(50):
        route = route_sql_validation(
            _state(decision="invalid_sql", path_state=path_state)
        )
        routes.append(route)
        if route == "unconstructable":
            break
        if route == "invalid_sql":
            path_state["failed_attempts"].append({})

    # Seven attempts go back for reconstruction and the eighth call gives up.
    assert [route for route in routes if route != "unconstructable"] == [
        "invalid_sql",
        "invalid_sql",
        "invalid_sql",
        "invalid_sql",
        "invalid_sql",
        "invalid_sql",
        "invalid_sql",
    ]
    assert routes[-1] == "unconstructable"
    assert len(path_state["failed_attempts"]) == 7
