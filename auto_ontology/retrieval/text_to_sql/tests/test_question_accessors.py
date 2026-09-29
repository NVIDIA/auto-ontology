# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import pytest

from auto_ontology.retrieval.text_to_sql.prompts import format_dual_question_block
from auto_ontology.retrieval.text_to_sql.state import (
    AgentState,
    get_original_question,
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
    # The submitted turn stays reachable unchanged.
    assert get_original_question(state) == SUBMITTED


def test_question_for_processing_prefers_normalized_over_standalone() -> None:
    state = _state(processing_question=STANDALONE, normalized_question=NORMALIZED)

    assert get_question_for_processing(state) == NORMALIZED


def test_question_for_processing_falls_back_to_standalone() -> None:
    assert get_question_for_processing(_state(processing_question=STANDALONE)) == (
        STANDALONE
    )


def test_dual_question_block_renders_all_three_when_distinct() -> None:
    block = format_dual_question_block(SUBMITTED, NORMALIZED, STANDALONE)

    assert block == (
        f"Original user request:\n{SUBMITTED}\n\n"
        f"Standalone processing question:\n{STANDALONE}\n\n"
        f"Normalized retrieval question:\n{NORMALIZED}"
    )


@pytest.mark.parametrize("processing_question", ["", SUBMITTED, NORMALIZED])
def test_dual_question_block_omits_redundant_standalone(
    processing_question: str,
) -> None:
    block = format_dual_question_block(SUBMITTED, NORMALIZED, processing_question)

    assert "Standalone processing question" not in block
    assert block == (
        f"Original user request:\n{SUBMITTED}\n\n"
        f"Normalized retrieval question:\n{NORMALIZED}"
    )


def test_dual_question_block_collapses_when_question_was_not_rewritten() -> None:
    assert format_dual_question_block(SUBMITTED, SUBMITTED, STANDALONE) == SUBMITTED
