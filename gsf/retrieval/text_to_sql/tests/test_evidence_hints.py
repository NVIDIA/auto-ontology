# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gsf.retrieval.text_to_sql.evidence_hints import (
    build_evidence_hints_block,
)


def test_empty_evidence_returns_no_hints() -> None:
    assert build_evidence_hints_block("What happened at 0:01:54?", "") == ""


def test_explicit_evidence_and_clean_question_build_time_like_hint() -> None:
    block = build_evidence_hints_block(
        "What happened at 0:01:54?",
        "The requested time refers to events.duration LIKE 'M:SS%'.",
    )

    assert "## Evidence-derived rules" in block
    assert "events.duration LIKE '1:54%'" in block


def test_question_evidence_section_is_not_scanned() -> None:
    question = (
        "What happened at 0:01:54?\n\n"
        "Evidence: The requested time refers to events.duration LIKE 'M:SS%'."
    )

    assert build_evidence_hints_block(question, "") == ""


def test_explicit_refers_to_evidence_builds_mapping_hint() -> None:
    block = build_evidence_hints_block(
        "Which status is active?",
        "The active status refers to accounts.status_code.",
    )

    assert "`accounts.status_code`" in block
