# SPDX-FileCopyrightText: Copyright (c) 2026, souljoy
# SPDX-License-Identifier: Apache-2.0

import json
import sqlite3
import sys
from contextlib import closing
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from dev_tools import qa_benchmark as benchmark

CONFIG = benchmark.RunConfig(
    revision="test-commit",
    model="test-model",
    provider="scripted",
    prompt_id="test-prompts-v1",
    thinking=False,
    concurrency=1,
    repeats=1,
    react_tool_budget=24,
    react_sql_budget=10,
    model_max_attempts=3,
)


@pytest.mark.parametrize("case", benchmark.CASES, ids=lambda case: case.id)
def test_frozen_expectations(case: benchmark.Case) -> None:
    if case.reference_sql is None:
        assert case.expected.status in {"clarify", "unanswerable"}
        return
    with closing(sqlite3.connect(":memory:")) as database:
        database.executescript(benchmark.FIXTURE_SQL)
        assert database.execute(case.reference_sql).fetchall() == [
            (case.expected.value,)
        ]
    assert case.expected.status == "answered"


@pytest.mark.parametrize(
    "text",
    [
        "refused",
        "[]",
        '{"status":"unanswerable"}',
        '{"status":"unanswerable","value":0}',
        '{"status":"clarify","value":false}',
        '{"status":"answered","value":true}',
        '{"status":"answered","value":"3"}',
        '{"status":"answered","value":-1}',
        '{"status":"answered","value":3,"reason":"extra"}',
    ],
)
def test_invalid_answer_is_unscored(text: str) -> None:
    assert benchmark.parse_outcome(text) is None


def _samples() -> list[benchmark.Sample]:
    samples = []
    for index, case in enumerate(benchmark.CASES):
        # These are scripted response and metric samples, including failures.
        react = benchmark.parse_outcome(json.dumps(asdict(case.expected)))
        control = benchmark.parse_outcome(
            '{"status":"unanswerable","value":null}' if index % 2 == 0 else ""
        )
        samples.extend(
            (
                benchmark.Sample(
                    case.id, 1, "react", react, 10 * (index + 1), 3, 2, 1, 1, 100, 20
                ),
                benchmark.Sample(
                    case.id,
                    1,
                    "llm_only",
                    control,
                    5,
                    1,
                    0,
                    0,
                    0,
                    80 if index % 2 == 0 else None,
                    None,
                ),
            )
        )
    return samples


def test_report_scores_by_slice_and_arm() -> None:
    report = benchmark.paired_report(CONFIG, _samples())
    assert report == benchmark.paired_report(CONFIG, reversed(_samples()))
    assert report["config"] == asdict(CONFIG)
    assert len(report["suite_sha256"]) == 64
    assert len(report["samples"]) == 20
    single_table = report["by_slice"]["single_table"]
    assert single_table["react"] == {
        "count": 2,
        "correct": 2,
        "refusals": 0,
        "p50_ms": 15,
        "p95_ms": 20,
        "model_calls": {"sum": 6, "samples": 2},
        "tool_calls": {"sum": 4, "samples": 2},
        "sql_runs": {"sum": 2, "samples": 2},
        "retries": {"sum": 2, "samples": 2},
        "input_tokens": {"sum": 200, "samples": 2},
        "output_tokens": {"sum": 40, "samples": 2},
    }
    assert single_table["llm_only"]["input_tokens"] == {"sum": 80, "samples": 1}
    assert single_table["llm_only"]["output_tokens"] == {"sum": None, "samples": 0}
    assert report["by_slice"]["unanswerable"]["llm_only"]["correct"] == 1
    assert report["by_slice"]["ambiguous"]["llm_only"]["correct"] == 0
    assert report["by_slice"]["ambiguous"]["react"]["correct"] == 2


def test_report_rejects_incomplete_or_duplicate_pairs() -> None:
    samples = _samples()
    with pytest.raises(ValueError, match="missing paired samples"):
        benchmark.paired_report(CONFIG, samples[1:])
    with pytest.raises(ValueError, match="unexpected or duplicate sample"):
        benchmark.paired_report(CONFIG, [*samples, samples[0]])
    with pytest.raises(ValueError, match="missing paired samples"):
        benchmark.paired_report(replace(CONFIG, repeats=2), samples)
    with pytest.raises(ValueError, match="LLM-only samples cannot"):
        benchmark.paired_report(CONFIG, [replace(samples[1], tool_calls=1)])
    with pytest.raises(ValueError, match="invalid latency"):
        benchmark.paired_report(CONFIG, [replace(samples[0], latency_ms=float("nan"))])


def test_cli_prints_report_for_captured_responses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    rows = [asdict(sample) for sample in _samples()]
    for row in rows:
        row["outcome"] = json.dumps(row["outcome"]) if row["outcome"] else None
    manifest = tmp_path / "samples.json"
    manifest.write_text(
        json.dumps({"config": asdict(CONFIG), "samples": rows}), encoding="utf-8"
    )
    monkeypatch.setattr(sys, "argv", ["qa_benchmark", str(manifest)])

    benchmark.main()

    assert json.loads(capsys.readouterr().out) == benchmark.paired_report(
        CONFIG, _samples()
    )
