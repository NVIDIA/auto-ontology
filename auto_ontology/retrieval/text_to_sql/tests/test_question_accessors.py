# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

from auto_ontology.retrieval.text_to_sql.state import (
    AgentState,
    get_question_for_processing,
    get_standalone_question,
)

SUBMITTED = "and next quarter?"
STANDALONE = "How many orders next quarter?"
NORMALIZED = "order count for next quarter"


def _state(**path_state: str) -> AgentState:
    return cast(
        AgentState,
        {"initial_question": SUBMITTED, "path_state": dict(path_state)},
    )


def test_standalone_question_falls_back_to_submitted_question() -> None:
    assert get_standalone_question(_state()) == SUBMITTED


def test_standalone_question_prefers_resolved_follow_up() -> None:
    state = _state(processing_question=STANDALONE)

    assert get_standalone_question(state) == STANDALONE


def test_question_for_processing_prefers_normalized_over_standalone() -> None:
    state = _state(processing_question=STANDALONE, normalized_question=NORMALIZED)

    assert get_question_for_processing(state) == NORMALIZED


def test_question_for_processing_falls_back_to_standalone() -> None:
    assert get_question_for_processing(_state(processing_question=STANDALONE)) == (
        STANDALONE
    )
