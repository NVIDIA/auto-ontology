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
from auto_ontology.retrieval.interactive.types import (
    InteractivePhase,
    SubmitSQLAction,
    TurnType,
)
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


def test_mixed_debug_keeps_decimal_columns_and_casts_floats_to_numeric() -> None:
    from decimal import Decimal

    import pandas as pd

    from auto_ontology.retrieval.text_to_sql.numeric_types import result_values_digest

    original = pd.DataFrame({"a": [Decimal("1.50")], "b": [2.25]})
    casted = pd.DataFrame({"a": [Decimal("1.50")], "b": [Decimal("2.25")]})
    digest = result_values_digest(original)
    assert digest == result_values_digest(casted)
    connector = _TableConnector(casted)
    sql = "SELECT a, b FROM aliens"
    sess = _table_session(connector, sql, ["decimal", "float"], digest=digest)
    sess.first_submit_numeric_kind = "decimal"
    sess.first_submit_values_digest = digest

    action = _step_debug(sess, sql)

    assert action.sql == "SELECT a, CAST(b AS DECIMAL) FROM aliens"
    assert "DOUBLE PRECISION" not in action.sql.upper()
    assert connector.executed == [action.sql]


def test_mixed_debug_drops_the_float_cast_when_values_change() -> None:
    from decimal import Decimal

    import pandas as pd

    from auto_ontology.retrieval.text_to_sql.numeric_types import result_values_digest

    original = pd.DataFrame({"a": [Decimal("1.50")], "b": [2.25]})
    connector = _TableConnector(
        pd.DataFrame({"a": [Decimal("9.99")], "b": [Decimal("2.25")]})
    )
    sql = "SELECT a, b FROM aliens"
    digest = result_values_digest(original)
    sess = _table_session(connector, sql, ["decimal", "float"], digest=digest)
    sess.first_submit_numeric_kind = "decimal"
    sess.first_submit_values_digest = digest

    action = _step_debug(sess, sql)

    assert action.sql == sql
    assert connector.executed  # the cast was probed, then rejected


def test_desc_limit_puts_nulls_last_before_submit() -> None:
    connector = _Connector()
    sess = _make_session(connectors=[connector], numeric_type_alignment=True)
    sess.path_state["relevant_tables"] = []
    sql = "SELECT id, score FROM aliens ORDER BY score DESC LIMIT 5"

    action = _step_follow_up(sess, sql)

    assert (
        action.sql
        == "SELECT id, score FROM aliens ORDER BY score DESC NULLS LAST LIMIT 5"
    )
    assert connector.executed == [action.sql]


def test_desc_limit_nulls_last_is_off_without_alignment() -> None:
    connector = _Connector()
    sess = _make_session(connectors=[connector])
    sess.path_state["relevant_tables"] = []
    sql = "SELECT id, score FROM aliens ORDER BY score DESC LIMIT 5"

    action = _step_follow_up(sess, sql)

    assert action.sql == sql
    assert connector.executed == []


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


def test_debug_without_a_recorded_type_retries_decimal_as_double():
    connector = _Connector()
    sess, sql = _debug_session(
        connector,
        first_kind=None,
        kinds=("decimal",),
        first_values_digest=None,
    )

    action = _step_debug(sess, sql)

    assert action.sql == "SELECT CAST(AVG(x) AS DOUBLE PRECISION) AS v FROM aliens"
    assert connector.executed == [action.sql]


def test_integer_first_submit_keeps_a_new_decimal_debug_answer():
    connector = _Connector()
    sess, sql = _debug_session(
        connector,
        first_kind=None,
        kinds=("decimal",),
        first_values_digest="integer-rows",
    )

    action = _step_debug(sess, sql)

    assert action.sql == sql
    assert connector.executed == []


def test_untyped_decimal_retry_falls_back_when_it_does_not_run():
    connector = _Connector(fail=True)
    sess, sql = _debug_session(
        connector,
        first_kind=None,
        kinds=("decimal",),
        first_values_digest=None,
    )

    assert _step_debug(sess, sql).sql == sql


def test_phase1_debug_sorts_by_the_leading_aggregate():
    connector = _Connector()
    sess = _make_session(connectors=[connector], numeric_type_alignment=True)
    sess.path_state["relevant_tables"] = []
    sql = (
        "SELECT c.series_id, COUNT(DISTINCT c.content_key) AS show_count, "
        "COUNT(DISTINCT r.tiertype) AS distribution_setup_count "
        "FROM core c GROUP BY c.series_id "
        "ORDER BY COUNT(DISTINCT r.tiertype) DESC, c.series_id"
    )

    action = _step_debug(sess, sql)

    order_by = action.sql.upper().split("ORDER BY", 1)[1]
    assert order_by.strip().startswith("SHOW_COUNT DESC")
    assert "SERIES_ID" not in order_by
    assert connector.executed  # the rewritten order was executed before submit


def test_phase2_debug_keeps_a_later_aggregate_order():
    connector = _Connector()
    sess = _make_session(connectors=[connector], numeric_type_alignment=True)
    sess.phase = InteractivePhase.ROUND2_CLARIFY
    sess.path_state["relevant_tables"] = []
    sql = (
        "SELECT c.series_id, COUNT(DISTINCT c.content_key) AS show_count, "
        "COUNT(DISTINCT r.tiertype) AS distribution_setup_count "
        "FROM core c GROUP BY c.series_id "
        "ORDER BY COUNT(DISTINCT r.tiertype) DESC, c.series_id"
    )

    action = _step_debug(sess, sql)

    assert action.sql == sql
    assert connector.executed == []


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


def test_first_submit_aligns_mixed_decimal_and_float_columns():
    from decimal import Decimal
    import pandas as pd

    class _MixedConnector:
        dialect = "postgres"
        database_name = "alien"

        def __init__(self) -> None:
            self.executed: list[str] = []

        def execute(self, sql: str, **_):
            self.executed.append(sql)
            return pd.DataFrame({"a": [Decimal("1.50")], "b": [Decimal("2.50")]})

    connector = _MixedConnector()
    sess = _make_session(connectors=[connector], numeric_type_alignment=True)
    sql = "SELECT a, b FROM aliens"
    sess.path_state["result_numeric_kinds"] = ["decimal", "float"]
    sess.path_state["result_numeric_kinds_sql"] = sql
    # Not this result's digest, so the table swap is rejected and the float
    # column is still aligned to numeric.
    sess.path_state["result_values_digest"] = "digest"
    sess.path_state["relevant_tables"] = []

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
        action = step(sess, turn_type=TurnType.FOLLOW_UP, follow_up_question="and now?")

    assert action.sql == "SELECT a, CAST(b AS DECIMAL) FROM aliens"
    assert sess.first_submit_numeric_kind == "decimal"
    assert sess.path_state["result_numeric_kinds"] == ["decimal", "decimal"]


def test_first_submit_kind_follows_the_stabilized_integer_sum():
    from decimal import Decimal

    import pandas as pd

    class _SumConnector:
        dialect = "postgres"
        database_name = "alien"

        def execute(self, sql: str, **_):
            if "INT" in sql.upper():
                return pd.DataFrame({"total": [12]})
            return pd.DataFrame({"total": [Decimal("12.00")]})

    sess = _make_session(connectors=[_SumConnector()], numeric_type_alignment=True)
    sql = "SELECT SUM(CAST(amount AS NUMERIC)) AS total FROM sales"
    sess.path_state["result_numeric_kinds"] = ["decimal"]
    sess.path_state["result_numeric_kinds_sql"] = sql
    sess.path_state["result_values_digest"] = "pre-stabilize"
    sess.path_state["relevant_tables"] = []

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
        action = step(sess, turn_type=TurnType.FOLLOW_UP, follow_up_question="and now?")

    assert "INT" in action.sql.upper()
    assert sess.first_submit_numeric_kind is None
    assert sess.first_submit_values_digest == "pre-stabilize"


def test_fractional_sum_keeps_the_decimal_cast():
    from decimal import Decimal

    import pandas as pd

    class _SumConnector:
        dialect = "postgres"
        database_name = "alien"

        def execute(self, sql: str, **_):
            if "INT" in sql.upper():
                return pd.DataFrame({"total": [4]})
            return pd.DataFrame({"total": [Decimal("3.20")]})

    sess = _make_session(connectors=[_SumConnector()], numeric_type_alignment=True)
    sql = "SELECT SUM(CAST(amount AS DECIMAL(10, 2))) AS total FROM sales"
    sess.path_state["result_numeric_kinds"] = ["decimal"]
    sess.path_state["result_numeric_kinds_sql"] = sql
    sess.path_state["result_values_digest"] = "pre-stabilize"
    sess.path_state["relevant_tables"] = []

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
        action = step(sess, turn_type=TurnType.FOLLOW_UP, follow_up_question="and now?")

    assert action.sql == sql
    assert sess.first_submit_numeric_kind == "decimal"
    assert sess.first_submit_values_digest == "pre-stabilize"


def test_integer_sum_that_does_not_run_keeps_the_decimal_cast():
    from decimal import Decimal

    import pandas as pd

    class _SumConnector:
        dialect = "postgres"
        database_name = "alien"

        def execute(self, sql: str, **_):
            if "INT" in sql.upper():
                raise RuntimeError("invalid input syntax for type integer")
            return pd.DataFrame({"total": [Decimal("2.5")]})

    sess = _make_session(connectors=[_SumConnector()], numeric_type_alignment=True)
    sql = "SELECT SUM(CAST(episodes AS NUMERIC)) AS total FROM shows"
    sess.path_state["result_numeric_kinds"] = ["decimal"]
    sess.path_state["result_numeric_kinds_sql"] = sql
    sess.path_state["result_values_digest"] = "pre-stabilize"
    sess.path_state["relevant_tables"] = []

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
        action = step(sess, turn_type=TurnType.FOLLOW_UP, follow_up_question="and now?")

    assert action.sql == sql
    assert sess.first_submit_numeric_kind == "decimal"


def test_first_submit_kind_follows_a_stabilized_numeric_average():
    from decimal import Decimal

    import pandas as pd

    class _PeriodConnector:
        dialect = "postgres"
        database_name = "alien"

        def execute(self, sql: str, **_):
            return pd.DataFrame(
                {
                    "audit_month": ["2024-01"],
                    "avg_afs": [Decimal("1.50")],
                    "severity_rank": [1],
                }
            )

    sess = _make_session(connectors=[_PeriodConnector()], numeric_type_alignment=True)
    sql = (
        "SELECT audit_month, CAST(avg_afs AS DOUBLE PRECISION), severity_rank "
        "FROM ranked ORDER BY severity_rank, audit_month"
    )
    sess.path_state["result_numeric_kinds"] = ["float"]
    sess.path_state["result_numeric_kinds_sql"] = sql
    sess.path_state["result_values_digest"] = "pre-stabilize"
    sess.path_state["relevant_tables"] = []

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
        action = step(sess, turn_type=TurnType.FOLLOW_UP, follow_up_question="and now?")

    assert "DOUBLE PRECISION" not in action.sql.upper()
    assert sess.first_submit_numeric_kind == "decimal"
    assert sess.first_submit_values_digest == "pre-stabilize"


def _step_follow_up(sess: InteractiveSessionState, sql: str) -> SubmitSQLAction:
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
        action = step(sess, turn_type=TurnType.FOLLOW_UP, follow_up_question="and now?")
    assert isinstance(action, SubmitSQLAction)
    return action


class _TableConnector:
    dialect = "postgres"
    database_name = "alien"

    def __init__(self, frame: object, *, fail: bool = False) -> None:
        self.frame = frame
        self.fail = fail
        self.executed: list[str] = []

    def execute(self, sql: str, **_: object) -> object:
        self.executed.append(sql)
        if self.fail:
            raise RuntimeError("boom")
        return self.frame


def _table_session(
    connector: _TableConnector,
    sql: str,
    kinds: list,
    *,
    digest: str | None,
    rows: int = 1,
) -> InteractiveSessionState:
    sess = _make_session(connectors=[connector], numeric_type_alignment=True)
    sess.path_state["result_numeric_kinds"] = kinds
    sess.path_state["result_numeric_kinds_sql"] = sql
    sess.path_state["result_values_digest"] = digest
    sess.path_state["result_row_count"] = rows
    sess.path_state["relevant_tables"] = []
    return sess


def test_table_first_submit_swaps_decimal_and_float_columns() -> None:
    from decimal import Decimal

    import pandas as pd

    from auto_ontology.retrieval.text_to_sql.numeric_types import result_values_digest

    original = pd.DataFrame({"a": [Decimal("1.50")], "b": [1.25], "name": ["n"]})
    swapped = pd.DataFrame({"a": [1.5], "b": [Decimal("1.25")], "name": ["n"]})
    connector = _TableConnector(swapped)
    sql = "SELECT a, b, name FROM aliens"
    sess = _table_session(
        connector,
        sql,
        ["decimal", "float", None],
        digest=result_values_digest(original),
    )

    action = _step_follow_up(sess, sql)

    assert (
        action.sql
        == "SELECT CAST(a AS DOUBLE PRECISION), CAST(b AS DECIMAL), name FROM aliens"
    )
    assert connector.executed == [action.sql]
    assert sess.first_submit_numeric_kind == "decimal"
    assert sess.path_state["result_numeric_kinds"] == ["float", "decimal", None]


def test_single_column_table_swaps_decimal_to_double() -> None:
    from decimal import Decimal

    import pandas as pd

    from auto_ontology.retrieval.text_to_sql.numeric_types import result_values_digest

    original = pd.DataFrame({"v": [Decimal("1.50"), Decimal("2.25")]})
    connector = _TableConnector(pd.DataFrame({"v": [1.5, 2.25]}))
    sql = "SELECT v FROM aliens"
    sess = _table_session(
        connector,
        sql,
        ["decimal"],
        digest=result_values_digest(original),
        rows=2,
    )

    action = _step_follow_up(sess, sql)

    assert action.sql == "SELECT CAST(v AS DOUBLE PRECISION) FROM aliens"
    assert sess.first_submit_numeric_kind == "float"


def test_scalar_first_submit_does_not_swap_numeric_type() -> None:
    from decimal import Decimal

    import pandas as pd

    from auto_ontology.retrieval.text_to_sql.numeric_types import result_values_digest

    frame = pd.DataFrame({"v": [Decimal("1.50")]})
    connector = _TableConnector(frame)
    sql = "SELECT AVG(x) AS v FROM aliens"
    sess = _table_session(
        connector, sql, ["decimal"], digest=result_values_digest(frame), rows=1
    )

    action = _step_follow_up(sess, sql)

    assert action.sql == sql
    assert connector.executed == []
    assert sess.first_submit_numeric_kind == "decimal"


def test_table_without_fractional_numbers_is_not_swapped() -> None:
    import pandas as pd

    connector = _TableConnector(pd.DataFrame({"id": [1], "name": ["n"]}))
    sql = "SELECT id, name FROM aliens"
    sess = _table_session(connector, sql, [None, None], digest="rows")

    action = _step_follow_up(sess, sql)

    assert action.sql == sql
    assert connector.executed == []


def test_table_swap_that_changes_values_keeps_the_generated_sql() -> None:
    from decimal import Decimal

    import pandas as pd

    from auto_ontology.retrieval.text_to_sql.numeric_types import result_values_digest

    original = pd.DataFrame({"v": [Decimal("1.50"), Decimal("2.25")]})
    connector = _TableConnector(pd.DataFrame({"v": [Decimal("9.99"), Decimal("2.25")]}))
    sql = "SELECT v FROM aliens"
    sess = _table_session(
        connector,
        sql,
        ["decimal"],
        digest=result_values_digest(original),
        rows=2,
    )

    action = _step_follow_up(sess, sql)

    assert action.sql == sql
    assert sess.first_submit_numeric_kind == "decimal"


def test_table_swap_that_does_not_run_keeps_the_generated_sql() -> None:
    from decimal import Decimal

    import pandas as pd

    from auto_ontology.retrieval.text_to_sql.numeric_types import result_values_digest

    frame = pd.DataFrame({"a": [Decimal("1.50")], "b": [1.25]})
    connector = _TableConnector(frame, fail=True)
    sql = "SELECT a, b FROM aliens"
    sess = _table_session(
        connector,
        sql,
        ["decimal", "float"],
        digest=result_values_digest(frame),
    )

    action = _step_follow_up(sess, sql)

    assert action.sql == sql
    assert sess.first_submit_numeric_kind == "decimal"
