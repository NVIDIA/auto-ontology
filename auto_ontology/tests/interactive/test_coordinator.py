# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from auto_ontology.retrieval.interactive.coordinator import (
    _apply_debug_seed,
    _apply_follow_up_seed,
    _format_user_clarifications,
    _run_sql_generation,
    apply_user_answer,
    create_session,
    step,
)
from auto_ontology.retrieval.interactive.types import SubmitSQLAction, TurnType
from auto_ontology.retrieval.interactive.state import InteractiveSessionState


def _make_session(**kwargs):
    defaults = dict(
        session_id="s1",
        task_id="t1",
        db_name="alien",
        db_schema="TABLE aliens ...",
        external_kb="[]",
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
        "auto_ontology.retrieval.interactive.coordinator._run_sql_generation",
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
            "auto_ontology.retrieval.interactive.coordinator._run_sql_generation",
            return_value="SELECT 1",
        ),
        patch(
            "auto_ontology.retrieval.interactive.coordinator.should_clarify",
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
        patch(
            "auto_ontology.retrieval.interactive.coordinator._get_llm",
            return_value=mock_llm,
        ),
        # merge_clarification is called with _get_fast_llm(), not _get_llm() —
        # both must be mocked or the real (unmocked) LLM client gets used instead.
        patch(
            "auto_ontology.retrieval.interactive.coordinator._get_fast_llm",
            return_value=mock_llm,
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
    with patch(
        "auto_ontology.retrieval.interactive.coordinator._get_llm",
        return_value=mock_llm,
    ):
        apply_user_answer(sess, "some answer")  # should not raise


def test_format_user_clarifications_concatenates_usable_answers():
    sess = _make_session()
    sess.clarify_history = [
        {"q": "Which year?", "a": "2023"},
        {"q": "Which status?", "a": "Use completed, ignoring case."},
    ]

    evidence = _format_user_clarifications(sess)

    assert "## User Clarifications (verbatim — authoritative)" in evidence
    assert "Clarification 1:\nQuestion: Which year?\nAnswer: 2023" in evidence
    assert (
        "Clarification 2:\n"
        "Question: Which status?\n"
        "Answer: Use completed, ignoring case."
    ) in evidence


def test_format_user_clarifications_excludes_unusable_answers():
    sess = _make_session()
    sess.clarify_history = [
        {"q": "Which formula?", "a": "Sorry, this question is out of scope."},
        {"q": "Which year?", "a": "2024"},
        {"q": "Which region?", "a": ""},
    ]

    evidence = _format_user_clarifications(sess)

    assert "out of scope" not in evidence
    assert "Which region?" not in evidence
    assert "Question: Which year?\nAnswer: 2024" in evidence


def test_sql_generation_includes_raw_clarifications_in_evidence():
    sess = _make_session(
        original_question="original question",
        working_question="expanded merged question",
        data_retriever=MagicMock(),
        semantic_retriever=MagicMock(),
        connectors=[],
    )
    sess._grounded_knowledge_for = sess.working_question
    sess.clarify_history = [
        {"q": "Which formula?", "a": "Divide cost by revenue."},
        {"q": "Anything else?", "a": "Sorry, this question is out of scope."},
    ]

    with (
        patch(
            "auto_ontology.retrieval.interactive.coordinator.generate_evidence",
            return_value="generated evidence",
        ),
        patch(
            "auto_ontology.retrieval.text_to_sql.main.get_agent_response_with_state",
            return_value={"sql_code": "SELECT 1"},
        ) as mock_generate,
    ):
        sql = _run_sql_generation(sess)

    payload = mock_generate.call_args.args[0]
    assert sql == "SELECT 1"
    assert payload["question"] == "original question"
    assert payload["enriched_question"] == "expanded merged question"
    assert "## Expanded Question" in payload["evidence"]
    assert (
        "Question: Which formula?\nAnswer: Divide cost by revenue."
        in payload["evidence"]
    )
    assert "out of scope" not in payload["evidence"]
    assert "generated evidence" in payload["evidence"]


# ── numeric type retry on a wrong-answer debug turn ─────────────────────────


class _Connector:
    dialect = "postgres"
    database_name = "alien"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.executed: list[str] = []

    def execute(self, sql: str, **_):
        self.executed.append(sql)
        if self.fail:
            raise RuntimeError("boom")
        import pandas as pd

        return pd.DataFrame({"v": [1.5]})


def _debug_session(
    connector,
    *,
    first_kind="decimal",
    kinds=("float",),
    first_values_digest="same-values",
    retry_values_digest="same-values",
):
    sess = _make_session(connectors=[connector], numeric_type_alignment=True)
    sess.first_submit_numeric_kind = first_kind
    sess.first_submit_values_digest = first_values_digest
    sess.path_state["result_values_digest"] = retry_values_digest
    sess.path_state["relevant_tables"] = []
    sql = "SELECT AVG(x) AS v FROM aliens"
    sess.path_state["result_numeric_kinds"] = list(kinds)
    sess.path_state["result_numeric_kinds_sql"] = sql
    return sess, sql


def _step_debug(sess, sql, **kwargs):
    with (
        patch(
            "auto_ontology.retrieval.interactive.coordinator._run_sql_generation",
            return_value=sql,
        ),
        patch(
            "auto_ontology.retrieval.interactive.coordinator.resolve_connector_from_tables",
            side_effect=lambda tables, connectors: connectors[0],
        ),
    ):
        return step(sess, turn_type=TurnType.DEBUG, **kwargs)


def test_wrong_answer_debug_resubmits_the_opposite_numeric_type():
    connector = _Connector()
    sess, sql = _debug_session(connector, first_kind="float", kinds=("float",))

    action = _step_debug(sess, sql)

    assert action.sql == "SELECT CAST(AVG(x) AS DECIMAL) AS v FROM aliens"
    assert connector.executed == [action.sql]  # verified before it is submitted


def test_decimal_first_submit_retries_as_double_precision():
    connector = _Connector()
    sess, sql = _debug_session(connector, first_kind="decimal", kinds=("decimal",))

    action = _step_debug(sess, sql)

    assert action.sql == "SELECT CAST(AVG(x) AS DOUBLE PRECISION) AS v FROM aliens"


def test_debug_that_changed_the_answer_keeps_its_own_types():
    connector = _Connector()
    sess, sql = _debug_session(
        connector,
        first_kind="float",
        first_values_digest="old-values",
        retry_values_digest="new-values",
    )

    action = _step_debug(sess, sql)

    assert action.sql == sql
    assert connector.executed == []


def test_no_flip_without_a_first_submit_values_digest():
    connector = _Connector()
    sess, sql = _debug_session(connector, first_kind="float", first_values_digest=None)

    assert _step_debug(sess, sql).sql == sql
    assert connector.executed == []


def test_execution_error_debug_never_flips_types():
    connector = _Connector()
    sess, sql = _debug_session(connector, first_kind="float")

    action = _step_debug(sess, sql, debug_error='column "x" does not exist')

    assert action.sql == sql
    assert connector.executed == []


def test_flip_that_does_not_run_falls_back_to_the_generated_sql():
    connector = _Connector(fail=True)
    sess, sql = _debug_session(connector, first_kind="float")

    assert _step_debug(sess, sql).sql == sql


def test_stale_kinds_for_other_sql_are_not_used():
    connector = _Connector()
    sess, _ = _debug_session(connector, first_kind="float")

    action = _step_debug(sess, "SELECT SUM(y) AS v FROM aliens")

    assert action.sql == "SELECT SUM(y) AS v FROM aliens"
    assert connector.executed == []


def test_no_flip_when_the_first_submit_had_no_fractional_numbers():
    connector = _Connector()
    sess, sql = _debug_session(connector, first_kind=None)

    assert _step_debug(sess, sql).sql == sql


def test_alignment_is_off_unless_the_session_asks_for_it():
    connector = _Connector()
    sess, sql = _debug_session(connector, first_kind="float")
    sess.numeric_type_alignment = False

    assert _step_debug(sess, sql).sql == sql
    assert connector.executed == []


def _new_session(**kwargs):
    return create_session(
        session_id="s1",
        task_id="t1",
        db_name="alien",
        db_schema="",
        external_kb="[]",
        question="q",
        data_retriever=None,
        semantic_retriever=None,
        connectors=[],
        **kwargs,
    )


def test_create_session_leaves_alignment_off_by_default():
    assert _new_session().numeric_type_alignment is False


def test_create_session_can_opt_in_to_alignment():
    assert _new_session(numeric_type_alignment=True).numeric_type_alignment is True


def test_first_submit_records_which_numeric_type_it_returned():
    sess = _make_session(numeric_type_alignment=True)
    sql = "SELECT AVG(x) FROM aliens"
    sess.path_state["result_numeric_kinds"] = ["float", None]
    sess.path_state["result_numeric_kinds_sql"] = sql
    sess.path_state["result_values_digest"] = "digest"

    with patch(
        "auto_ontology.retrieval.interactive.coordinator._run_sql_generation",
        return_value=sql,
    ):
        step(sess, turn_type=TurnType.FOLLOW_UP, follow_up_question="and now?")

    assert sess.first_submit_numeric_kind == "float"
    assert sess.first_submit_values_digest == "digest"
