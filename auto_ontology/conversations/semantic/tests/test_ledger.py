# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-stage ledger."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pytest

from auto_ontology.conversations.semantic.pipeline import _Ledger


@dataclass
class _Graph:
    nodes: list[str] = field(default_factory=list)
    edges: list[str] = field(default_factory=list)


class TestNoting:
    def test_the_first_stage_counts_everything_it_made(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        graph = _Graph(nodes=["a", "b"], edges=["x"])
        with caplog.at_level(logging.INFO):
            _Ledger("7").note("extract", graph)

        assert "nodes   +2" in caplog.text
        assert "edges   +1" in caplog.text

    def test_a_later_stage_counts_only_its_own_change(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        graph = _Graph(nodes=["a", "b"], edges=["x"])
        ledger = _Ledger("7")
        ledger.note("extract", graph)
        graph.nodes.append("c")
        with caplog.at_level(logging.INFO):
            ledger.note("consolidate", graph)

        assert "consolidate" in caplog.text
        assert "nodes   +1" in caplog.text

    def test_a_stage_that_removes_shows_a_negative(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        graph = _Graph(nodes=["a"], edges=["x", "y"])
        ledger = _Ledger("7")
        ledger.note("extract", graph)
        graph.edges.pop()
        with caplog.at_level(logging.INFO):
            ledger.note("fix reversed", graph)

        assert "edges   -1" in caplog.text

    def test_a_no_op_stage_is_visibly_a_no_op(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        graph = _Graph(nodes=["a"])
        ledger = _Ledger("7")
        ledger.note("extract", graph)
        with caplog.at_level(logging.INFO):
            ledger.note("disambiguate", graph)

        assert "nodes   +0" in caplog.text

    def test_the_conversation_is_named_on_every_line(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO):
            _Ledger("42").note("extract", _Graph())

        assert "[42]" in caplog.text

    def test_what_a_stage_found_is_reported_beside_its_counts(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO):
            _Ledger("7").note("extract", _Graph(), found="54 window(s)")

        assert "54 window(s)" in caplog.text


class TestRecording:
    def test_a_stage_with_no_draft_graph_still_lands(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO):
            _Ledger("7").record("derive", edges=18, found="18 supersedes")

        assert "edges  +18" in caplog.text
        assert "18 supersedes" in caplog.text

    def test_recording_does_not_disturb_the_running_totals(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        graph = _Graph(nodes=["a"])
        ledger = _Ledger("7")
        ledger.note("extract", graph)
        ledger.record("derive", edges=18)
        graph.nodes.append("b")
        with caplog.at_level(logging.INFO):
            ledger.note("later", graph)

        assert "nodes   +1" in caplog.text


class TestSummarising:
    def test_every_stage_appears_in_the_table(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        ledger = _Ledger("7")
        ledger.note("extract", _Graph(nodes=["a"]))
        ledger.record("derive", edges=2)
        with caplog.at_level(logging.INFO):
            ledger.summarise()

        assert "stage" in caplog.text
        assert "extract" in caplog.text
        assert "derive" in caplog.text

    def test_a_stage_that_found_nothing_shows_a_dash(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        ledger = _Ledger("7")
        ledger.note("extract", _Graph())
        with caplog.at_level(logging.INFO):
            ledger.summarise()

        assert "-" in caplog.text

    def test_summarising_an_empty_run_does_not_fail(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO):
            _Ledger("7").summarise()

        assert "stage by stage" in caplog.text
