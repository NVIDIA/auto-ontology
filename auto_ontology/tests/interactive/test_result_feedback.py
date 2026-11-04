# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock, patch

from auto_ontology.retrieval.interactive.coordinator import (
    _apply_debug_seed,
    _apply_follow_up_seed,
    _run_sql_generation,
)
from auto_ontology.retrieval.interactive.result_feedback import (
    attach_result_mismatch_diagnosis,
    format_result_sample,
    parse_result_feedback,
    wrong_result_hint,
)
from auto_ontology.retrieval.interactive.state import InteractiveSessionState


def _session(**path_state) -> InteractiveSessionState:
    session = InteractiveSessionState(
        session_id="s1",
        task_id="t1",
        db_name="alien",
        db_schema="TABLE aliens ...",
        external_kb="",
        original_question="How many aliens are active?",
        working_question="How many aliens are active?",
        data_retriever=MagicMock(),
        semantic_retriever=MagicMock(),
        connectors=[],
    )
    session._grounded_knowledge_for = session.working_question
    session.path_state = {
        "_resume_from": "reconstruct_sql",
        "error": wrong_result_hint(),
        "sql_code": "SELECT COUNT(*) AS n FROM aliens",
        "sql_response_from_db": [
            json.dumps([{"n": 0}]),
        ],
        **path_state,
    }
    return session


def test_format_result_sample_reads_the_execution_envelope():
    long_name = "A" * 80
    payload = json.dumps(
        [
            {"id": 1, "name": None},
            {"id": 2, "name": long_name},
        ]
    )

    sample = format_result_sample([payload])

    assert sample.startswith("2 rows; columns: id, name")
    assert "name null 1/2" in sample
    assert "name=NULL" in sample
    assert "A" * 60 in sample
    assert sample.endswith("…")


def test_format_result_sample_reports_an_empty_result():
    assert format_result_sample([json.dumps([])]) == "0 rows."
    assert format_result_sample(None) == ""
    assert format_result_sample(["not json"]) == ""


def test_parse_result_feedback_reads_a_labelled_reply():
    verdict, diagnosis = parse_result_feedback(
        "VERDICT: MISMATCH\nDIAGNOSIS: The count has no status filter."
    )

    assert verdict == "MISMATCH"
    assert diagnosis == "The count has no status filter."


def test_parse_result_feedback_treats_plain_text_as_a_mismatch():
    verdict, diagnosis = parse_result_feedback("The filter dropped every active row.")

    assert verdict == "MISMATCH"
    assert "active row" in diagnosis


def test_parse_result_feedback_accepts_type_or_order():
    verdict, diagnosis = parse_result_feedback(
        "**VERDICT:** TYPE_OR_ORDER\n**DIAGNOSIS:** Only the numeric type differs."
    )

    assert verdict == "TYPE_OR_ORDER"
    assert diagnosis == "Only the numeric type differs."


def test_attach_puts_a_mismatch_ahead_of_the_checklist():
    session = _session()

    with patch(
        "auto_ontology.retrieval.interactive.result_feedback.diagnose_result_mismatch",
        return_value=(
            "VERDICT: MISMATCH\n"
            "DIAGNOSIS: The count includes inactive aliens; the question asks "
            "for active ones."
        ),
    ):
        attach_result_mismatch_diagnosis(session, "status = 'active'")

    error = session.path_state["error"]
    assert error.startswith("RESULT MISMATCH DIAGNOSIS: The count includes inactive")
    assert "n=0" in error
    assert "status = 'active'" not in error  # evidence is for the model call
    assert "COMPOUND FILTER COMPLETENESS" in error
    assert "preserve the exact same logical result rows" not in error
    assert session.path_state["error_known_fixable"] is True


def test_attach_keeps_values_when_only_the_type_looks_wrong():
    session = _session(
        result_numeric_kinds=["float"],
        result_numeric_kinds_sql="SELECT COUNT(*) AS n FROM aliens",
    )
    seen = {}

    def _capture(**kwargs):
        seen.update(kwargs)
        return "VERDICT: TYPE_OR_ORDER\nDIAGNOSIS: The single count is already right."

    with patch(
        "auto_ontology.retrieval.interactive.result_feedback.diagnose_result_mismatch",
        side_effect=_capture,
    ):
        attach_result_mismatch_diagnosis(session, "")

    error = session.path_state["error"]
    assert "single count is already right" in error
    assert "Keep the same filters" in error
    assert "preserve the exact same logical result rows" in error
    assert seen["numeric_kinds"] == "float"
    assert session.path_state["error_known_fixable"] is True


def test_attach_leaves_the_checklist_when_diagnosis_fails():
    session = _session()

    with patch(
        "auto_ontology.retrieval.interactive.result_feedback.diagnose_result_mismatch",
        side_effect=RuntimeError("endpoint down"),
    ):
        attach_result_mismatch_diagnosis(session, "")

    assert session.path_state["error"] == wrong_result_hint()
    assert "error_known_fixable" not in session.path_state


def test_debug_seed_clears_a_stale_fixable_flag():
    session = _session()
    session.path_state["error_known_fixable"] = True

    _apply_debug_seed(session, debug_error=None)

    assert "error_known_fixable" not in session.path_state
    assert "SQL produced incorrect results" in session.path_state["error"]


def test_follow_up_seed_clears_a_stale_fixable_flag():
    session = _session()
    session.path_state["error_known_fixable"] = True

    _apply_follow_up_seed(session, follow_up_question="Now show them by year.")

    assert "error_known_fixable" not in session.path_state


def test_sql_generation_sends_the_diagnosis_into_reconstruction():
    session = _session()
    session.working_question = "How many aliens are active? Count only status = active."
    session._grounded_knowledge_for = session.working_question
    captured = {}

    def _fake_response(payload):
        captured.update(payload)
        return {
            "sql_code": "SELECT COUNT(*) FROM aliens WHERE active",
            "path_state": {},
        }

    with (
        patch(
            "auto_ontology.retrieval.interactive.result_feedback.diagnose_result_mismatch",
            return_value=(
                "VERDICT: MISMATCH\nDIAGNOSIS: Missing the active-status filter."
            ),
        ),
        patch(
            "auto_ontology.retrieval.text_to_sql.main.get_agent_response_with_state",
            side_effect=_fake_response,
        ),
    ):
        _run_sql_generation(session)

    error = captured["path_state"]["error"]
    assert "Missing the active-status filter" in error
    assert "n=0" in error
    assert captured["path_state"]["error_known_fixable"] is True
    assert captured["evidence"]  # question expansion still rides along


def test_sql_generation_skips_diagnosis_for_an_execution_error():
    session = _session(
        error='column "active" does not exist',
    )

    with (
        patch(
            "auto_ontology.retrieval.interactive.result_feedback.diagnose_result_mismatch",
        ) as diagnose,
        patch(
            "auto_ontology.retrieval.text_to_sql.main.get_agent_response_with_state",
            return_value={"sql_code": "SELECT 1", "path_state": {}},
        ),
    ):
        _run_sql_generation(session)

    diagnose.assert_not_called()
