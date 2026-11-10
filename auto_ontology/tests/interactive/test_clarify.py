# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from auto_ontology.retrieval.interactive.clarify import (
    _CLARIFY_PROMPT,
    _format_ambiguous_terms,
    _format_incomplete_formulas,
    _open_ambiguous_terms,
)


def test_open_ambiguous_term_lists_candidate_meanings():
    open_terms = _open_ambiguous_terms(
        {"profile growth speed"},
        {
            "profile growth speed": [
                {"text": "ColumnAttribute: FollGrow. Follower growth rate per day"},
                {"text": "ColumnAttribute: FingGrow. Following growth rate per day"},
            ]
        },
    )

    assert _format_ambiguous_terms(open_terms) == (
        '- "profile growth speed" could mean: '
        "(1) Follower growth rate per day (2) Following growth rate per day"
    )


def test_term_with_one_describable_candidate_is_not_open():
    open_terms = _open_ambiguous_terms(
        {"risk level"},
        {"risk level": [{"text": "Risk level"}, {"text": ""}]},
    )

    assert open_terms == {}
    assert _format_ambiguous_terms(open_terms) == "None"


def test_prompt_has_the_ambiguous_terms_section_and_rule():
    assert "{ambiguous_terms}" in _CLARIFY_PROMPT
    assert "12. For a term under" in _CLARIFY_PROMPT


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
