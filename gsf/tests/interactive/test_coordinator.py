from unittest.mock import MagicMock, patch

import pytest

from gsf.retrieval.interactive.coordinator import (
    _apply_debug_seed,
    _apply_follow_up_seed,
    apply_user_answer,
    step,
)
from gsf.retrieval.interactive.types import SubmitSQLAction, TurnType
from gsf.retrieval.interactive.state import InteractiveSessionState


def _make_session(**kwargs):
    defaults = dict(
        session_id="s1",
        task_id="t1",
        db_name="alien",
        db_schema="TABLE aliens ...",
        external_kg="[]",
        original_question="test question",
        working_question="test question",
    )
    defaults.update(kwargs)
    return InteractiveSessionState(**defaults)


def test_step_requires_turn_type():
    """The coordinator no longer classifies turns itself — callers must
    supply turn_type from their own protocol state."""
    sess = _make_session()
    with pytest.raises(TypeError):
        step(sess)  # missing required keyword-only argument


def test_step_rejects_none_turn_type():
    sess = _make_session()
    with pytest.raises(ValueError):
        step(sess, turn_type=None)


def test_debug_seed_sets_resume_from():
    sess = _make_session()
    _apply_debug_seed(sess, debug_error=None)
    assert sess.path_state["_resume_from"] == "reconstruct_sql"
    assert sess.path_state["sql_attempts"] == 0
    assert sess.path_state["error_analysis_done"] is False


def test_debug_seed_sets_error():
    sess = _make_session()
    _apply_debug_seed(sess, debug_error="column X does not exist")
    assert sess.path_state["error"] == "column X does not exist"


def test_debug_seed_no_error_injects_wrong_result_hint():
    sess = _make_session()
    _apply_debug_seed(sess, debug_error=None)
    assert "SQL produced incorrect results" in sess.path_state["error"]


def test_step_routes_exec_error_through_debug_path():
    """End-to-end (no live DB/orchestrator): a caller that already extracted
    the DB error from its own submit response (see coordinator.step's
    turn_type/debug_error params) drives it through the real
    coordinator.step() and gets the real error seeded, not the generic
    wrong-result hint."""
    sess = _make_session()
    sess.path_state["sql_code"] = (
        "SELECT bad_column FROM aliens"  # prior failed attempt
    )

    actual_error = 'column "bad_column" does not exist'

    with patch(
        "gsf.retrieval.interactive.coordinator._run_sql_generation",
        return_value="SELECT column FROM aliens",
    ) as mock_gen:
        action = step(sess, turn_type=TurnType.DEBUG, debug_error=actual_error)

    mock_gen.assert_called_once()
    assert isinstance(action, SubmitSQLAction)
    assert action.sql == "SELECT column FROM aliens"
    assert sess.path_state["error"] == actual_error
    assert sess.path_state["_resume_from"] == "reconstruct_sql"


def test_follow_up_seed_sets_working_question():
    sess = _make_session()
    _apply_follow_up_seed(sess, follow_up_question="Show totals for 2023.")
    assert sess.working_question == "Show totals for 2023."


def test_follow_up_seed_falls_back_to_existing_question_when_empty():
    sess = _make_session(working_question="prior question")
    _apply_follow_up_seed(sess, follow_up_question="")
    assert sess.working_question == "prior question"


def test_follow_up_seed_clears_sql_keys():
    sess = _make_session()
    sess.path_state["sql_code"] = "SELECT 1"
    sess.path_state["normalized_question"] = "old"
    _apply_follow_up_seed(sess, follow_up_question="Show totals.")
    assert "sql_code" not in sess.path_state
    assert "normalized_question" not in sess.path_state


def test_step_initial_sets_question():
    sess = _make_session(original_question="", working_question="")
    with (
        patch(
            "gsf.retrieval.interactive.coordinator._run_sql_generation",
            return_value="SELECT 1",
        ),
        patch(
            "gsf.retrieval.interactive.coordinator.should_clarify",
            return_value=(False, None),
        ),
    ):
        step(sess, turn_type=TurnType.INITIAL, initial_question="How many aliens?")
    assert sess.working_question == "How many aliens?"
    assert sess.original_question == "How many aliens?"


def test_apply_user_answer_merges_question():
    sess = _make_session()
    sess._pending_question = "Which year?"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "How many aliens were observed in 2023?"
    with (
        patch("gsf.retrieval.interactive.coordinator._get_llm", return_value=mock_llm),
        # merge_clarification is called with _get_fast_llm(), not _get_llm() —
        # both must be mocked or the real (unmocked) LLM client gets used instead.
        patch(
            "gsf.retrieval.interactive.coordinator._get_fast_llm", return_value=mock_llm
        ),
    ):
        apply_user_answer(sess, "2023")
    assert sess.working_question == "How many aliens were observed in 2023?"
    assert len(sess.clarify_history) == 1
    assert sess.clarify_history[0] == {"q": "Which year?", "a": "2023"}
    assert sess._pending_question is None


def test_apply_user_answer_no_pending_still_merges():
    """Even with no pending question, apply_user_answer should not crash."""
    sess = _make_session()
    sess._pending_question = None
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "test question"
    with patch("gsf.retrieval.interactive.coordinator._get_llm", return_value=mock_llm):
        apply_user_answer(sess, "some answer")  # should not raise
