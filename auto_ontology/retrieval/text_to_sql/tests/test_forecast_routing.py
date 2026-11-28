# SPDX-FileCopyrightText: Copyright (c) 2026, xiaoyesoso
# SPDX-License-Identifier: Apache-2.0

"""Frozen AUTO-mode forecast routing cases with a synthetic warehouse and predictor.

Set AUTO_ONTOLOGY_LIVE_FORECAST_ROUTING=1 and run this file with ``-k live``
under each reasoning configuration to be qualified. The live model gets the
ordinary AUTO tools and the same questions as the scripted CI cases.
"""

import json
import os
from collections.abc import Callable

import pytest
from langchain_core.messages import AIMessage

from auto_ontology.retrieval.text_to_sql.react import loop
from auto_ontology.retrieval.text_to_sql.react import run_sql as run_sql_module
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
from auto_ontology.retrieval.text_to_sql.react.mode import EpisodeMode, episode_mode
from auto_ontology.retrieval.text_to_sql.react.predict import Prediction
from auto_ontology.retrieval.text_to_sql.tests.react_fakes import (
    FakeCatalog,
    ScriptedModel,
    SQLiteConnector,
    call,
)

CATALOG = {
    "churn_forecasts": [
        "producer",
        "customer_id",
        "window_start",
        "window_end",
        "churn_score",
    ],
    "customers": ["customer_id"],
    "orders": ["customer_id", "ordered_on"],
}

DATA = """
CREATE TABLE churn_forecasts (
    producer TEXT, customer_id INTEGER, window_start TEXT, window_end TEXT,
    churn_score REAL,
    PRIMARY KEY (producer, customer_id, window_start, window_end)
);
INSERT INTO churn_forecasts VALUES
    ('retention-v2', 7, '2099-04-01', '2099-05-01', 0.73),
    ('retention-v1', 7, '2099-04-01', '2099-05-01', 0.11),
    ('retention-v2', 8, '2099-04-01', '2099-05-01', 0.22),
    ('retention-v2', 7, '2099-04-02', '2099-05-01', 0.33),
    ('retention-v2', 7, '2099-04-01', '2099-05-02', 0.44);
CREATE TABLE customers (customer_id INTEGER PRIMARY KEY);
CREATE TABLE orders (customer_id INTEGER REFERENCES customers, ordered_on TEXT);
"""

SQL_EVIDENCE = """
TABLE churn_forecasts: already-computed churn forecasts. One row per producer,
customer and exact forecast window. Dates are stored as YYYY-MM-DD.
- producer TEXT: identifier of the model that produced the result
- customer_id INTEGER: identifier of the forecast entity
- window_start TEXT: inclusive start of the forecast window
- window_end TEXT: exclusive end of the forecast window
- churn_score REAL: forecast probability of no orders within the window
TABLE customers: customer entities.
- customer_id INTEGER: primary key
TABLE orders: historical orders.
- customer_id INTEGER: references customers.customer_id
- ordered_on TEXT: date of the order
"""

DEFINITION = (
    "churn_score is the forecast probability that the customer places no orders "
    "within the forecast window, on a 0 to 1 scale."
)

STORED_SQL = """
SELECT customer_id, churn_score FROM churn_forecasts
WHERE producer = 'retention-v2' AND customer_id = 7
  AND window_start = '2099-04-01' AND window_end = '2099-05-01'
""".strip()

PREDICTION = Prediction(
    {
        "response": "The new churn forecast for customer 7 is 0.81.",
        "sql_columns": ["ENTITY", "True_PROB"],
        "sql_response_from_db": [{"ENTITY": 7, "True_PROB": 0.81}],
    }
)


@pytest.fixture
def forecast_context(monkeypatch: pytest.MonkeyPatch) -> ToolContext:
    catalog = FakeCatalog(CATALOG)
    monkeypatch.setattr(run_sql_module, "catalog_for", lambda *_: catalog)
    monkeypatch.setenv("KUMO_RFM_API_URL", "http://synthetic.invalid")
    monkeypatch.setenv("TEXT_TO_SQL_REACT_SUBMIT_TOOL", "true")

    def retrieve(ctx: ToolContext, query: str, kinds=None) -> str:
        for table in CATALOG:
            ctx.show_table(None, table)
        return SQL_EVIDENCE

    def predict(ctx: ToolContext) -> tuple[str, Prediction | None]:
        if not {("", "customers"), ("", "orders")} <= ctx.shown_tables:
            return "Retrieve the customer and its order history first.", None
        return "Prediction complete.", PREDICTION

    monkeypatch.setattr(loop, "retrieve", retrieve)
    monkeypatch.setattr(loop, "information", lambda *_: f"TERM {DEFINITION}")
    monkeypatch.setattr(loop, "predict", predict)
    return ToolContext(connectors=[SQLiteConnector(DATA)], information_answers=True)


@pytest.fixture(
    params=[
        "scripted",
        pytest.param(
            "live",
            marks=pytest.mark.skipif(
                os.getenv("AUTO_ONTOLOGY_LIVE_FORECAST_ROUTING") != "1",
                reason="requires an explicitly enabled live reasoning configuration",
            ),
        ),
    ]
)
def run_forecast(
    request: pytest.FixtureRequest, forecast_context: ToolContext
) -> Callable[..., loop.EpisodeResult]:
    def run(question: str, *calls: dict) -> loop.EpisodeResult:
        if request.param == "live":
            from auto_ontology.utils.llm_invoke import get_llm_client

            model = get_llm_client()
        else:
            model = ScriptedModel(
                [AIMessage(content="", tool_calls=[c]) for c in calls]
            )
        mode = episode_mode(None)
        assert mode is EpisodeMode.AUTO
        forecast_context.question = question
        return loop.run_episode(
            model, forecast_context, question, dialect="sqlite", mode=mode
        )

    return run


def _observations(result: loop.EpisodeResult, name: str) -> list[str]:
    return [
        entry["content"]
        for entry in result.trajectory
        if entry["role"] == "tool" and entry["name"] == name
    ]


def test_stored_forecast_runs_sql_with_exact_keys(run_forecast: Callable) -> None:
    result = run_forecast(
        "What customer_id and churn_score did retention-v2 already save for "
        "customer 7 for the exact April 1 to May 1, 2099 forecast window?",
        call("retrieve", "1", query="stored churn forecasts"),
        call("run_sql", "2", sql=STORED_SQL),
        call("submit", "3", sql=STORED_SQL, response="Customer 7 scored 0.73."),
    )

    assert not _observations(result, "predict")
    # Omitting any key admits a decoy with a distinct value. Check both the
    # SQL tool observation and the actual rows from the final query.
    assert "1 rows\ncustomer_id | churn_score\n7 | 0.73" in _observations(
        result, "run_sql"
    )
    assert result.reason == "submitted"
    assert result.submission is not None and result.submission.db_response is not None
    rows = json.loads(result.submission.db_response.result[0])
    assert rows == [{"customer_id": 7, "churn_score": 0.73}]
    assert list(rows[0]) == ["customer_id", "churn_score"]

    # Keeping only the decoys also catches a missing key masked by LIMIT 1.
    sql = result.submission.db_response.statement
    assert sql is not None
    decoys = SQLiteConnector(
        DATA + "DELETE FROM churn_forecasts WHERE churn_score = 0.73;"
    )
    assert decoys.execute(sql).empty


def test_forecast_definition_uses_catalog_and_prose(
    run_forecast: Callable, forecast_context: ToolContext
) -> None:
    result = run_forecast(
        "What does churn_score mean in the churn forecast catalog?",
        call("retrieve", "1", query="churn_score", purpose="information"),
        call("submit", "2", response=DEFINITION),
    )

    assert forecast_context.information_retrieved
    assert not forecast_context.connectors[0].executed
    assert not _observations(result, "run_sql")
    assert not _observations(result, "predict")
    assert result.reason == "submitted"
    assert result.submission is not None
    assert not result.submission.sql
    assert result.submission.db_response is None
    answer = result.submission.response.lower()
    assert any(word in answer for word in ("probability", "likelihood", "chance"))
    assert "order" in answer and any(word in answer for word in ("window", "period"))


def test_new_forecast_calls_predict(
    run_forecast: Callable, forecast_context: ToolContext
) -> None:
    result = run_forecast(
        "How likely is customer 7 to place no orders over the next 30 days? "
        "Please compute a fresh forecast using today's order history.",
        call("retrieve", "1", query="customer and order history"),
        call("predict", "2"),
    )

    assert _observations(result, "predict") == ["Prediction complete."]
    assert not _observations(result, "run_sql")
    assert not forecast_context.connectors[0].executed
    assert result.reason == "predicted"
    assert result.submission is None
    assert result.prediction == PREDICTION
