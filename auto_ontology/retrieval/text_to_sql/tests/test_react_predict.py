# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KumoRFM through the ReAct agent: the ``predict`` tool and the modes that offer it."""

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from auto_ontology.retrieval import kumo
from auto_ontology.retrieval.kumo import rag
from auto_ontology.retrieval.text_to_sql import main, pipeline
from auto_ontology.retrieval.text_to_sql.react import agent as agent_module
from auto_ontology.retrieval.text_to_sql.react import loop
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
from auto_ontology.retrieval.text_to_sql.react.loop import EpisodeResult
from auto_ontology.retrieval.text_to_sql.react.mode import EpisodeMode, episode_mode
from auto_ontology.retrieval.text_to_sql.react.predict import Prediction, predict
from auto_ontology.retrieval.text_to_sql.tests.react_fakes import ScriptedModel, call

PREDICTED = {
    "response": "Customer 7 is likely to churn.",
    "sql_code": "PREDICT COUNT(orders.*, 0, 30, days) = 0 FOR customers.id = 7",
    "sql_columns": ["ENTITY", "True_PROB"],
    "custom_analyses_used": [],
    "sql_response_from_db": [{"ENTITY": 7, "True_PROB": 0.8}],
}

CUSTOMERS = {"id": "t1", "name": "customers", "database_name": "shop"}


class _Built:
    """Stands in for a KumoRFM ``PredictionContext``."""


@pytest.fixture
def kumo_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {"build": None, "run": None, "result": _Built()}

    def build(connectors, tables, join_paths=None, examples=None):
        seen["build"] = {"tables": tables, "join_paths": join_paths}
        return seen["result"]

    def run(question, llm, context):
        seen["run"] = question
        return PREDICTED

    monkeypatch.setattr(kumo, "PredictionContext", _Built)
    monkeypatch.setattr(kumo, "build_prediction_context", build)
    monkeypatch.setattr(kumo, "run_prediction", run)
    monkeypatch.setattr(rag, "fetch_pql_examples", lambda _retriever, _q: [])
    return seen


def _ctx(**kwargs: Any) -> ToolContext:
    return ToolContext(
        retriever=None,
        connectors=[SimpleNamespace(database_name="shop", dialect="sqlite")],
        question="Which customers will churn next month?",
        **kwargs,
    )


@pytest.mark.parametrize(
    ("override", "kumo_url", "mode"),
    [
        (True, "http://nim", EpisodeMode.PREDICT),
        (False, "http://nim", EpisodeMode.SQL),
        (None, "http://nim", EpisodeMode.AUTO),
        (None, "", EpisodeMode.SQL),
    ],
)
def test_the_request_picks_the_mode(
    monkeypatch: pytest.MonkeyPatch,
    override: bool | None,
    kumo_url: str,
    mode: EpisodeMode,
) -> None:
    monkeypatch.setenv("KUMO_RFM_API_URL", kumo_url)

    assert episode_mode(override) is mode


def test_forcing_a_prediction_without_kumo_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KUMO_RFM_API_URL", raising=False)

    with pytest.raises(ValueError, match="KUMO_RFM_API_URL"):
        main._build_state(
            {
                "question": "q",
                "connectors": [object()],
                "data_retriever": object(),
                "prediction": True,
            }
        )


def test_predict_builds_over_the_tables_in_view(kumo_calls: dict[str, Any]) -> None:
    ctx = _ctx(seen_tables={"t1": CUSTOMERS})
    ctx.join_paths.append({"path": [{"source_table": "orders"}]})

    observation, prediction = predict(ctx)

    assert observation == "Prediction complete."
    assert prediction == Prediction(PREDICTED)
    assert kumo_calls["build"]["tables"] == [CUSTOMERS]
    assert kumo_calls["build"]["join_paths"] == ctx.join_paths
    assert kumo_calls["run"] == "Which customers will churn next month?"


def test_a_refused_build_is_an_observation(kumo_calls: dict[str, Any]) -> None:
    kumo_calls["result"] = {"response": "No relevant tables were available."}
    ctx = _ctx(seen_tables={"t1": CUSTOMERS})

    observation, prediction = predict(ctx)

    assert prediction is None
    assert "No relevant tables were available." in observation
    assert ctx.prediction_refusal == "No relevant tables were available."


def test_predict_needs_tables_first(kumo_calls: dict[str, Any]) -> None:
    observation, prediction = predict(_ctx())

    assert prediction is None
    assert observation.startswith("Error: no tables")
    assert kumo_calls["build"] is None


def test_predict_calls_are_bounded(
    kumo_calls: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEXT_TO_SQL_REACT_MAX_PREDICT_CALLS", "1")
    kumo_calls["result"] = {"response": "refused"}
    ctx = _ctx(seen_tables={"t1": CUSTOMERS})

    predict(ctx)
    observation, _ = predict(ctx)

    assert observation == "No predict calls left in this episode."


def _reply(*calls: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=list(calls))


def test_a_prediction_ends_the_episode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loop, "retrieve", lambda _ctx, query, kinds=None: "TABLE x")
    monkeypatch.setattr(
        loop, "predict", lambda _ctx: ("Prediction complete.", Prediction(PREDICTED))
    )
    model = ScriptedModel(
        [_reply(call("retrieve", "1", query="customers")), _reply(call("predict", "2"))]
    )

    result = loop.run_episode(model, _ctx(), "q", mode=EpisodeMode.PREDICT)

    assert result.reason == "predicted"
    assert result.prediction == Prediction(PREDICTED)
    assert result.submission is None
    assert model.bound_tool_names[0] == ["retrieve", "predict"]


def test_sql_mode_does_not_offer_predict(monkeypatch: pytest.MonkeyPatch) -> None:
    model = ScriptedModel([_reply(call("predict", "1"))])

    result = loop.run_episode(model, _ctx(), "q", mode=EpisodeMode.SQL)

    assert "predict" not in model.bound_tool_names[0]
    observation = [t for t in result.trajectory if t["role"] == "tool"][0]["content"]
    assert observation.startswith("Unknown tool 'predict'")


def test_auto_mode_offers_both_answers() -> None:
    model = ScriptedModel([])

    loop.run_episode(model, _ctx(), "q", mode=EpisodeMode.AUTO)

    assert model.bound_tool_names[0] == ["retrieve", "run_sql", "submit", "predict"]


def _state(**extra: Any) -> dict:
    return {
        "initial_question": "Which customers will churn next month?",
        "evidence": "",
        "llm": SimpleNamespace(),
        "messages": [],
        "connectors": [SimpleNamespace(database_name="shop", dialect="sqlite")],
        "semantic_retriever": object(),
        "path_state": {"target_db": "shop"},
        **extra,
    }


def test_the_prediction_is_the_final_response(monkeypatch: pytest.MonkeyPatch) -> None:
    episode = EpisodeResult(None, "predicted", 2, [], prediction=Prediction(PREDICTED))
    monkeypatch.setattr(agent_module, "run_episode", lambda *a, **k: episode)

    result = agent_module.ReactSQLAgent().execute(_state(prediction_override=True))

    assert result["decision"] == "predicted"
    assert result["path_state"]["final_response"] == PREDICTED
    assert result["messages"][-1].content == PREDICTED["response"]
    assert pipeline.route_react(result) is None


def test_a_refusal_explains_an_unanswered_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def episode(_llm, ctx, **_kwargs):
        ctx.prediction_refusal = "This question covers too many customers."
        return EpisodeResult(None, "budget_exhausted", 2, [])

    monkeypatch.setattr(agent_module, "run_episode", episode)

    result = agent_module.ReactSQLAgent().execute(_state(prediction_override=True))

    assert result["decision"] == "unconstructable"
    assert result["path_state"]["unconstructable_explanation"] == (
        "This question covers too many customers."
    )


def test_a_prediction_ends_the_pipeline_without_formatting() -> None:
    calls: list[str] = []

    def step(name: str, output: dict) -> Any:
        def run(_state: Any) -> dict:
            calls.append(name)
            return output

        return run

    steps = {
        "react_sql_agent": step("react_sql_agent", {"decision": "predicted"}),
        "format_and_respond": step("format_and_respond", {}),
        "unconstructable_sql_response": step("unconstructable_sql_response", {}),
    }

    list(pipeline.run_steps({"path_state": {}, "decision": ""}, steps))

    assert calls == ["react_sql_agent"]
