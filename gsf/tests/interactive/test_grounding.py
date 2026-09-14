import sys
import types
from unittest.mock import MagicMock, patch

from gsf.retrieval.interactive.grounding import ground_external_knowledge


def _make_llm(response: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value.content = response
    return llm


# ── ground_external_knowledge ─────────────────────────────────────────────────


def test_ground_empty_kg_returns_empty_without_llm_call():
    llm = MagicMock()
    result = ground_external_knowledge("How many panels?", "", llm)
    assert result == ""
    llm.invoke.assert_not_called()


def test_ground_llm_returns_none_gives_empty():
    llm = _make_llm("NONE")
    result = ground_external_knowledge(
        "How many panels?", "- PPR: some definition", llm
    )
    assert result == ""


def test_ground_llm_returns_none_case_insensitive():
    llm = _make_llm("none")
    result = ground_external_knowledge(
        "How many panels?", "- PPR: some definition", llm
    )
    assert result == ""


def test_ground_returns_llm_output_when_relevant():
    llm = _make_llm(
        "- PPR: Panel Performance Ratio = MeasuredPower / RatedPower * 100%"
    )
    result = ground_external_knowledge(
        "What is the average PPR?", "- PPR: some definition", llm
    )
    assert "PPR" in result
    assert result != ""


def test_ground_passes_question_and_kg_to_prompt():
    llm = _make_llm("- some fact")
    ground_external_knowledge(
        "What is churn?", "- churn rate: % of lost customers", llm
    )
    # safe_invoke_text wraps the prompt as [HumanMessage(content=prompt)], not a raw string.
    messages = llm.invoke.call_args[0][0]
    prompt_text = messages[0].content
    assert "What is churn?" in prompt_text
    assert "churn rate" in prompt_text


# ── wiring: _run_sql_generation injects grounded_kb into custom_prompts ───────
#
# coordinator._run_sql_generation does a lazy `from gsf.retrieval.text_to_sql.main import ...`
# which triggers module-level LLM client initialisation (requires REASONING_API_KEY).
# Inject a fake module into sys.modules before the import runs to avoid this.


def _make_fake_t2s_main(fake_fn):
    mod = types.ModuleType("gsf.retrieval.text_to_sql.main")
    mod.get_agent_response_with_state = fake_fn
    return mod


def _make_fake_t2s_state():
    mod = types.ModuleType("gsf.retrieval.text_to_sql.state")
    # TextToSQLPayload is just a TypedDict — a plain dict suffices for tests
    mod.TextToSQLPayload = dict
    return mod


def test_run_sql_generation_sets_custom_prompts():
    from gsf.retrieval.interactive.coordinator import _run_sql_generation
    from gsf.retrieval.interactive.state import InteractiveSessionState

    sess = InteractiveSessionState(
        session_id="s1",
        task_id="t1",
        db_name="solar",
        db_schema="TABLE panels ...",
        external_kb="- PPR: Panel Performance Ratio",
        original_question="What is the average PPR?",
        working_question="What is the average PPR?",
    )

    captured_payload = {}

    def fake_get_agent_response(payload):
        captured_payload.update(payload)
        return {"sql_code": "SELECT AVG(ppr) FROM panels", "path_state": {}}

    mock_llm = _make_llm(
        "- PPR: Panel Performance Ratio = MeasuredPower / RatedPower * 100%"
    )

    with (
        patch(
            "gsf.retrieval.interactive.coordinator._get_fast_llm",
            return_value=mock_llm,
        ),
        # generate_evidence() calls safe_invoke_text_nr(), a live LLM call —
        # stub it so this test stays hermetic instead of hitting the network.
        patch(
            "gsf.retrieval.interactive.coordinator.generate_evidence",
            return_value="",
        ),
    ):
        sys.modules["gsf.retrieval.text_to_sql.main"] = _make_fake_t2s_main(
            fake_get_agent_response
        )
        sys.modules["gsf.retrieval.text_to_sql.state"] = _make_fake_t2s_state()
        try:
            _run_sql_generation(sess)
        finally:
            sys.modules.pop("gsf.retrieval.text_to_sql.main", None)
            sys.modules.pop("gsf.retrieval.text_to_sql.state", None)

    assert captured_payload.get("custom_prompts") != ""
    assert "PPR" in captured_payload["custom_prompts"]


def test_run_sql_generation_empty_kg_gives_empty_custom_prompts():
    from gsf.retrieval.interactive.coordinator import _run_sql_generation
    from gsf.retrieval.interactive.state import InteractiveSessionState

    sess = InteractiveSessionState(
        session_id="s1",
        task_id="t1",
        db_name="solar",
        db_schema="TABLE panels ...",
        external_kb="",
        original_question="How many panels?",
        working_question="How many panels?",
    )

    captured_payload = {}

    def fake_get_agent_response(payload):
        captured_payload.update(payload)
        return {"sql_code": "SELECT COUNT(*) FROM panels", "path_state": {}}

    mock_llm = MagicMock()

    with patch(
        "gsf.retrieval.interactive.coordinator._get_fast_llm", return_value=mock_llm
    ):
        sys.modules["gsf.retrieval.text_to_sql.main"] = _make_fake_t2s_main(
            fake_get_agent_response
        )
        sys.modules["gsf.retrieval.text_to_sql.state"] = _make_fake_t2s_state()
        try:
            _run_sql_generation(sess)
        finally:
            sys.modules.pop("gsf.retrieval.text_to_sql.main", None)
            sys.modules.pop("gsf.retrieval.text_to_sql.state", None)

    assert captured_payload.get("custom_prompts") == ""
    mock_llm.invoke.assert_not_called()
