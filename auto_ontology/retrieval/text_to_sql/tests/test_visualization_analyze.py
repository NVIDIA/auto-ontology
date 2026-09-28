# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visualization orchestration preserves detail results instead of charting one facet."""

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel
from pytest import MonkeyPatch

from auto_ontology.retrieval.text_to_sql.visualization import analyze
from auto_ontology.retrieval.text_to_sql.visualization.prompts import (
    PlotRecommendation,
    VisualizationDecisionModel,
    VisualizationRecommendation,
)


def test_detail_listing_uses_complete_table(monkeypatch: MonkeyPatch) -> None:
    question = "Give me specifications of all AWS clusters we collect telemetry from"
    sql = "SELECT name, location, gpu_technology, gpu_count FROM cluster_passport"
    response = (
        '[{"name":"cluster-a","location":"us-west","gpu_technology":"H100",'
        '"gpu_count":64}]'
    )
    calls: list[type[BaseModel]] = []

    def invoke(
        _llm: object,
        messages: list[BaseMessage],
        schema: type[BaseModel],
    ) -> BaseModel:
        calls.append(schema)
        prompt = next(
            message.content for message in messages if isinstance(message, HumanMessage)
        )
        assert question in prompt
        assert sql in prompt
        assert "gpu_technology" in prompt
        assert "gpu_count" in prompt
        return VisualizationDecisionModel(render_as="table")

    monkeypatch.setattr(analyze, "invoke_with_structured_output", invoke)

    charts = analyze.analyze_and_visualize(object(), question, sql, response)

    assert charts is None
    assert calls == [VisualizationDecisionModel]


def test_missing_render_decision_skips_chart_generation(
    monkeypatch: MonkeyPatch,
) -> None:
    calls: list[type[BaseModel]] = []

    def invoke(
        _llm: object,
        _messages: list[BaseMessage],
        schema: type[BaseModel],
    ) -> None:
        calls.append(schema)
        return None

    monkeypatch.setattr(analyze, "invoke_with_structured_output", invoke)

    charts = analyze.analyze_and_visualize(
        object(),
        "Give me specifications of all AWS clusters",
        "SELECT name, location, gpu_count FROM cluster_passport",
        '[{"name":"a","location":"us-west","gpu_count":64}]',
    )

    assert charts is None
    assert calls == [VisualizationDecisionModel]


def test_comparison_continues_to_chart_generation(monkeypatch: MonkeyPatch) -> None:
    responses = iter(
        [
            VisualizationDecisionModel(render_as="chart"),
            VisualizationRecommendation(
                plots=[
                    PlotRecommendation(
                        plot_type="bar",
                        title="GPU Count by Cluster",
                        x="name",
                        y="gpu_count",
                    )
                ]
            ),
        ]
    )
    schemas: list[type[BaseModel]] = []

    def invoke(
        _llm: object,
        _messages: list[BaseMessage],
        schema: type[BaseModel],
    ) -> BaseModel:
        schemas.append(schema)
        return next(responses)

    monkeypatch.setattr(analyze, "invoke_with_structured_output", invoke)

    charts = analyze.analyze_and_visualize(
        object(),
        "Compare GPU counts across AWS clusters",
        "SELECT name, gpu_count FROM cluster_passport",
        '[{"name":"a","gpu_count":64},{"name":"b","gpu_count":128}]',
    )

    assert charts is not None
    assert charts[0]["x"]["key"] == "name"
    assert charts[0]["series"][0]["key"] == "gpu_count"
    assert schemas == [VisualizationDecisionModel, VisualizationRecommendation]
