# SPDX-FileCopyrightText: Copyright (c) 2026, souljoy
# SPDX-License-Identifier: Apache-2.0

"""Frozen Q&A cases and a paired report for comparing two answering arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, Literal

Slice = Literal["single_table", "multi_table", "ambiguous", "follow_up", "unanswerable"]
Status = Literal["answered", "clarify", "unanswerable"]
Arm = Literal["react", "llm_only"]

FIXTURE_SQL = """
CREATE TABLE teams (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
CREATE TABLE tickets (
    id INTEGER PRIMARY KEY,
    team_id INTEGER NOT NULL REFERENCES teams(id),
    status TEXT NOT NULL CHECK (status IN ('open', 'closed')),
    priority TEXT NOT NULL CHECK (priority IN ('high', 'normal'))
);
INSERT INTO teams VALUES (1, 'Support'), (2, 'Billing'), (3, 'Operations');
INSERT INTO tickets VALUES
    (10, 1, 'open', 'high'), (11, 1, 'open', 'normal'),
    (12, 1, 'closed', 'high'), (13, 2, 'open', 'high'),
    (14, 2, 'closed', 'normal'), (15, 2, 'closed', 'high'),
    (16, 3, 'closed', 'high');
"""


@dataclass(frozen=True)
class Outcome:
    status: Status
    value: int | None = None


@dataclass(frozen=True)
class Case:
    id: str
    slice: Slice
    question: str
    expected: Outcome
    reference_sql: str | None = None


CASES = (
    Case(
        "all-tickets",
        "single_table",
        "How many tickets are there?",
        Outcome("answered", 7),
        "SELECT COUNT(*) FROM tickets",
    ),
    Case(
        "all-open",
        "single_table",
        "How many tickets are open?",
        Outcome("answered", 3),
        "SELECT COUNT(*) FROM tickets WHERE status = 'open'",
    ),
    Case(
        "support-open",
        "multi_table",
        "How many open tickets does Support have?",
        Outcome("answered", 2),
        "SELECT COUNT(*) FROM tickets t JOIN teams g ON t.team_id = g.id "
        "WHERE g.name = 'Support' AND t.status = 'open'",
    ),
    Case(
        "teams-without-open",
        "multi_table",
        "How many teams have no open tickets?",
        Outcome("answered", 1),
        "SELECT COUNT(*) FROM teams g WHERE NOT EXISTS "
        "(SELECT 1 FROM tickets t WHERE t.team_id = g.id AND t.status = 'open')",
    ),
    Case("best-team", "ambiguous", "Which is the best team?", Outcome("clarify")),
    Case(
        "unspecified-team",
        "ambiguous",
        "How many tickets does the team have?",
        Outcome("clarify"),
    ),
    Case(
        "other-team",
        "follow_up",
        "Previous user: How many open tickets does Support have?\n"
        "Previous assistant: 2.\nCurrent user: What about Billing?",
        Outcome("answered", 1),
        "SELECT COUNT(*) FROM tickets t JOIN teams g ON t.team_id = g.id "
        "WHERE g.name = 'Billing' AND t.status = 'open'",
    ),
    Case(
        "other-status",
        "follow_up",
        "Previous user: How many open tickets are high priority?\n"
        "Previous assistant: 2.\nCurrent user: How many closed ones are there?",
        Outcome("answered", 3),
        "SELECT COUNT(*) FROM tickets WHERE status = 'closed' AND priority = 'high'",
    ),
    Case(
        "response-time",
        "unanswerable",
        "What was the average first response time in minutes?",
        Outcome("unanswerable"),
    ),
    Case(
        "weekend-tickets",
        "unanswerable",
        "How many tickets were created last weekend?",
        Outcome("unanswerable"),
    ),
)

ANSWER_FORMAT = (
    'Return a JSON object with exactly "status" and "value". Use status "answered" '
    'and an integer value for a count. Use status "clarify" and null if important '
    'information is missing from the question. Use status "unanswerable" and null '
    "if the fixture lacks the data needed to answer. Use only the provided fixture."
)


def fixture_evidence() -> str:
    """Render the same complete data snapshot for either arm from the SQL fixture."""
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript(FIXTURE_SQL)
        tables = {}
        for table in ("teams", "tickets"):
            cursor = connection.execute(f"SELECT * FROM {table} ORDER BY id")
            tables[table] = {
                "columns": [column[0] for column in cursor.description],
                "rows": cursor.fetchall(),
            }
    return "tickets.team_id refers to teams.id.\n" + json.dumps(tables, sort_keys=True)


def parse_outcome(text: str | None) -> Outcome | None:
    """Decode a structured answer. Invalid or missing responses receive no credit."""
    if text is None:
        return None
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"status", "value"}:
        return None
    status, value = payload["status"], payload["value"]
    if status == "answered" and type(value) is int and value >= 0:
        return Outcome(status, value)
    if status in ("clarify", "unanswerable") and value is None:
        return Outcome(status)
    return None


@dataclass(frozen=True)
class RunConfig:
    revision: str
    model: str
    provider: str
    prompt_id: str
    thinking: bool
    concurrency: int
    repeats: int
    react_tool_budget: int
    react_sql_budget: int
    model_max_attempts: int


@dataclass(frozen=True)
class Sample:
    """One arm result. Model calls include retries and failed invocation attempts."""

    case_id: str
    repeat: int
    arm: Arm
    outcome: Outcome | None
    latency_ms: float
    model_calls: int
    tool_calls: int
    sql_runs: int
    retries: int
    input_tokens: int | None
    output_tokens: int | None


def _summarize(samples: list[dict]) -> dict:
    latencies = sorted(sample["latency_ms"] for sample in samples)
    summary = {
        "count": len(samples),
        "correct": sum(sample["correct"] for sample in samples),
        "refusals": sum(
            sample["outcome"] == {"status": "unanswerable", "value": None}
            for sample in samples
        ),
        "p50_ms": median(latencies),
        "p95_ms": latencies[math.ceil(len(latencies) * 0.95) - 1],
    }
    for metric in (
        "model_calls",
        "tool_calls",
        "sql_runs",
        "retries",
        "input_tokens",
        "output_tokens",
    ):
        values = [sample[metric] for sample in samples if sample[metric] is not None]
        summary[metric] = {
            "sum": sum(values) if values else None,
            "samples": len(values),
        }
    return summary


def paired_report(config: RunConfig, samples: Iterable[Sample]) -> dict:
    """Score and aggregate a complete set of pairs with a JSON-serializable result."""
    if min(config.concurrency, config.repeats, config.model_max_attempts) < 1:
        raise ValueError(
            "concurrency, repeats, and model_max_attempts must be positive"
        )
    cases = {case.id: case for case in CASES}
    arms = ("react", "llm_only")
    expected = {
        (case_id, repeat, arm)
        for case_id in cases
        for repeat in range(1, config.repeats + 1)
        for arm in arms
    }
    rows = {}
    by_slice = defaultdict(list)
    for sample in samples:
        key = (sample.case_id, sample.repeat, sample.arm)
        if key not in expected or key in rows:
            raise ValueError(f"unexpected or duplicate sample: {key}")
        if not math.isfinite(sample.latency_ms) or sample.latency_ms < 0:
            raise ValueError(f"invalid latency for {key}")
        counts = (
            sample.model_calls,
            sample.tool_calls,
            sample.sql_runs,
            sample.retries,
        )
        tokens = (sample.input_tokens, sample.output_tokens)
        if (
            any(type(value) is not int or value < 0 for value in counts)
            or any(
                value is not None and (type(value) is not int or value < 0)
                for value in tokens
            )
            or sample.retries > sample.model_calls
        ):
            raise ValueError(f"invalid call or token counts for {key}")
        if sample.arm == "llm_only" and (sample.tool_calls or sample.sql_runs):
            raise ValueError(f"LLM-only samples cannot execute tools or SQL: {key}")
        case = cases[sample.case_id]
        row = {
            **asdict(sample),
            "slice": case.slice,
            "correct": sample.outcome == case.expected,
        }
        rows[key] = row
        by_slice[(case.slice, sample.arm)].append(row)
    if missing := expected - rows.keys():
        raise ValueError(f"missing paired samples: {sorted(missing)}")
    suite = {
        "fixture": FIXTURE_SQL,
        "evidence": fixture_evidence(),
        "cases": [asdict(case) for case in CASES],
        "answer_format": ANSWER_FORMAT,
    }
    return {
        "report_version": 1,
        "suite_sha256": hashlib.sha256(
            json.dumps(suite, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "config": asdict(config),
        "samples": [rows[key] for key in sorted(rows)],
        "by_slice": {
            slice_name: {arm: _summarize(by_slice[(slice_name, arm)]) for arm in arms}
            for slice_name in sorted({case.slice for case in CASES})
        },
    }


def main() -> None:
    """Read captured samples from a JSON manifest and print the scored report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest", type=Path, help="JSON object with config and samples"
    )
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        config = RunConfig(**manifest["config"])
        samples = (
            Sample(**{**row, "outcome": parse_outcome(row["outcome"])})
            for row in manifest["samples"]
        )
        report = paired_report(config, samples)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
