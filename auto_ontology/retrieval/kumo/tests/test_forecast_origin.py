# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dated predictions through the PQL runner, without a hosted service."""

from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from auto_ontology.retrieval.kumo import pql_gen, predictor
from auto_ontology.retrieval.kumo.telemetry import GraphIdentity

_ORIGIN = pd.Timestamp("2024-03-29")
_CONTEXT = pd.Timestamp("2024-02-29")
_PQL = "PREDICT SUM(orders.amount, 0, 1, months) FOR EACH customers.id"


@pytest.fixture
def model():
    model = Mock()
    model.validate_pql.return_value = SimpleNamespace(
        target_ast=SimpleNamespace(
            date_offset_range=SimpleNamespace(end_date_offset=pd.DateOffset(months=1))
        )
    )

    def predict(query, *, indices, **kwargs):
        rows = pd.DataFrame(
            {
                "ENTITY": indices,
                "ANCHOR_TIMESTAMP": pd.Timestamp(kwargs["anchor_time"], tz="UTC"),
                "SCORE": [12] * len(indices),
            }
        )
        if kwargs.get("explain"):
            return SimpleNamespace(prediction=rows, summary="An explanation")
        return rows

    model.predict.side_effect = predict
    return model


def _generate(model, connector, **kwargs):
    return pql_gen.generate_pql(
        "Forecast the order total",
        llm=Mock(invoke=Mock(return_value=SimpleNamespace(content=_PQL))),
        kumo_model=model,
        connector=connector,
        graph_ddl='"customers"("id" ID)\n"orders"("amount" numerical)',
        available_entity_ids={"customers": [1, 2, 3]},
        time_columns=kwargs.pop("time_columns", {"orders": "placed_at"}),
        **kwargs,
    )


@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize(
    ("today", "data_max"), [("2025-01-01", "2024-12-31"), ("2026-11-28", "2026-07-31")]
)
def test_explicit_origin_ignores_wall_clock_and_data_max(
    model, monkeypatch, explain, today, data_max
):
    monkeypatch.setattr(pd.Timestamp, "today", lambda: pd.Timestamp(today))
    connector = Mock(
        execute=Mock(return_value=pd.DataFrame({"m": [pd.Timestamp(data_max)]}))
    )

    result = _generate(
        model,
        connector,
        explain=explain,
        explain_entity="2",
        anchor_time=_ORIGIN,
        context_anchor_time=_CONTEXT,
    )

    assert result.success, result.error
    assert result.rows
    assert all(
        row["ANCHOR_TIMESTAMP"] == _ORIGIN.tz_localize("UTC") for row in result.rows
    )
    assert model.predict.call_args.kwargs["anchor_time"] == _ORIGIN
    assert model.predict.call_args.kwargs["context_anchor_time"] == _CONTEXT
    assert all("MAX(" not in call.args[0] for call in connector.execute.call_args_list)
    if explain:
        assert result.explanation == "An explanation"
        assert model.predict.call_args.kwargs["indices"] == [2]


def test_undated_prediction_still_uses_default_origin(model, monkeypatch):
    expected = pd.Timestamp("2024-01-01")
    fallback = Mock(return_value=expected)
    monkeypatch.setattr(pql_gen, "_forecast_anchor", fallback)

    result = _generate(model, Mock())

    assert result.success, result.error
    fallback.assert_called_once()
    assert model.predict.call_args.kwargs["anchor_time"] == expected
    assert "context_anchor_time" not in model.predict.call_args.kwargs


def test_each_batch_and_repair_attempt_retains_the_requested_origins(
    model, monkeypatch
):
    monkeypatch.setattr(pql_gen, "_PREDICT_BATCH_SIZE", 2)
    predict = model.predict.side_effect
    calls = 0

    def fail_one_batch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("Replan this PQL")
        return predict(*args, **kwargs)

    model.predict.side_effect = fail_one_batch
    result = _generate(model, Mock(), anchor_time=_ORIGIN, context_anchor_time=_CONTEXT)

    assert result.success, result.error
    assert result.attempts == 2
    assert [call.kwargs["indices"] for call in model.predict.call_args_list] == [
        [1, 2],
        [3],
        [1, 2],
        [3],
    ]
    assert all(
        call.kwargs["anchor_time"] == _ORIGIN for call in model.predict.call_args_list
    )
    assert all(
        call.kwargs["context_anchor_time"] == _CONTEXT
        for call in model.predict.call_args_list
    )


@pytest.mark.parametrize(
    ("origin", "context", "time_columns", "error"),
    [
        ("2024-03-28", "2024-02-29", {"orders": "placed_at"}, "target window"),
        ("2024-03-29", "2024-03-30", {"orders": "placed_at"}, "later"),
        ("2024-03-29", None, {"orders": None}, "timestamped"),
    ],
)
def test_incompatible_origins_are_refused_before_inference(
    model, origin, context, time_columns, error
):
    result = _generate(
        model,
        Mock(),
        anchor_time=pd.Timestamp(origin),
        context_anchor_time=pd.Timestamp(context) if context else None,
        time_columns=time_columns,
    )

    assert not result.success
    assert error in result.error
    model.predict.assert_not_called()


@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize("anchors", [None, ["2024-03-29", "2024-03-30"], []])
def test_unverified_rows_are_not_exposed(model, explain, anchors):
    if anchors is None:
        rows = pd.DataFrame({"ENTITY": [1]})
    else:
        rows = pd.DataFrame({"ANCHOR_TIMESTAMP": anchors})
    model.predict.side_effect = None
    model.predict.return_value = SimpleNamespace(prediction=rows) if explain else rows

    result = _generate(
        model, Mock(), explain=explain, explain_entity="2", anchor_time=_ORIGIN
    )

    assert not result.success
    assert "requested forecast origin" in result.error
    assert not result.rows
    model.predict.assert_called_once()


def test_native_runner_formats_the_verified_origin(model):
    context = predictor.PredictionContext(
        kumo_model=model,
        connector=Mock(),
        graph_ddl='"customers"("id" ID)\n"orders"("amount" numerical)',
        graph_edges=[],
        graph_col_stypes={},
        time_columns={"orders": "placed_at"},
        table_names={},
        entity_ids={"customers": [1, 2, 3]},
        examples=[],
        column_reference="",
        identity=GraphIdentity(),
    )
    llm = Mock(invoke=Mock(return_value=SimpleNamespace(content=_PQL)))

    response = predictor.run_prediction(
        "Forecast the order total",
        llm,
        context,
        anchor_time=_ORIGIN,
        context_anchor_time=_CONTEXT,
    )

    rows = response["sql_response_from_db"]
    assert rows
    assert {row["ANCHOR_TIMESTAMP"] for row in rows} == {"2024-03-29T00:00:00+00:00"}
    assert model.predict.call_args.kwargs["context_anchor_time"] == _CONTEXT
    prompt = llm.invoke.call_args.args[0][0].content
    assert "sets the forecast origin to 2024-03-29" in prompt
    assert "sets the context cutoff to 2024-02-29" in prompt


def test_full_population_helper_forwards_dated_options(model, monkeypatch):
    monkeypatch.setattr(
        pql_gen, "_resolve_indices", lambda *args, **kwargs: ([1, 2], 2)
    )
    frame = pql_gen.predict_all(
        _PQL,
        kumo_model=model,
        connector=Mock(),
        time_columns={"orders": "placed_at"},
        anchor_time=_ORIGIN,
        context_anchor_time=_CONTEXT,
    )

    assert len(frame) == 2
    assert model.predict.call_args.kwargs["anchor_time"] == _ORIGIN
    assert model.predict.call_args.kwargs["context_anchor_time"] == _CONTEXT
