# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An unanswered question still reaches the user with an explanation."""

from __future__ import annotations

from typing import Any

from auto_ontology.retrieval.text_to_sql.agents.sql_unconstructable import (
    SQLUnconstructableAgent,
)

_EXPLANATION = "I couldn't reach the database, so the query never ran."


def _state(**overrides: Any) -> dict:
    state = {
        "path_state": {},
        "connectors": [],
        "messages": [],
        "decision": "",
    }
    state.update(overrides)
    return state


def test_the_explanation_reaches_the_final_response() -> None:
    """``_extract_answer`` reads ``final_response``; ``messages`` was a dead end."""
    state = _state(path_state={"unconstructable_explanation": _EXPLANATION})

    result = SQLUnconstructableAgent().execute(state)

    final = result["path_state"]["final_response"]
    assert final["response"] == _EXPLANATION
    # A non-empty response is what makes the server persist the turn at all.
    assert final["response"].strip()
    assert isinstance(result["messages"], list), "a dict here replaces the list"
    assert result["messages"][-1].content == _EXPLANATION


def test_falls_back_to_a_generic_explanation() -> None:
    final = SQLUnconstructableAgent().execute(_state())["path_state"]["final_response"]

    assert final["response"] == "SQL can't be constructed from the data."
