# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from auto_ontology.retrieval.interactive.clarify import (
    _CLARIFY_PROMPT,
    _format_incomplete_formulas,
)


def test_no_gaps_is_none():
    assert _format_incomplete_formulas([], ["Anything?"]) == "None"


def test_gap_asked_by_a_refused_question_is_marked():
    note = _format_incomplete_formulas(
        [
            ("overall efficiency", "how to combine several readings"),
            ("safety index", "exact formula"),
        ],
        ["How should Overall Efficiency be reduced to one value per equipment?"],
    )

    assert note.splitlines() == [
        "- overall efficiency: how to combine several readings [UNANSWERED]",
        "- safety index: exact formula",
    ]


def test_gaps_are_unmarked_without_refusals():
    note = _format_incomplete_formulas([("downtime score", "exact formula")], [])

    assert note == "- downtime score: exact formula"


def test_hyphenated_term_matches_split_words_in_question():
    note = _format_incomplete_formulas(
        [("order-fill rate", "which quantity is the denominator")],
        ["For the order fill rate, is the denominator the total quantity?"],
    )

    assert note.endswith("[UNANSWERED]")


def test_prompt_rules_reference_the_unanswered_marker():
    assert "never ask about it again in any form" in _CLARIFY_PROMPT
    assert 'lists "None" or only items marked [UNANSWERED]' in _CLARIFY_PROMPT
