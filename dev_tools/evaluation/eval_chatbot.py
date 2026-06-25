# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the text-to-SQL agent against a chatbot evaluation JSON file.

For every entry in the JSON array (each containing ``question_id``,
``question``, ``SQL`` (expected), and ``answer_raw`` (expected user-facing
result)), this script:

1. Calls ``get_agent_response`` with the question (same path as
   ``ingest_postgres.run_retrieve``).
2. Scores the agent's SQL against the expected SQL by **executing both**
   queries against the live Postgres connector and comparing the resulting
   row sets.  Failure to execute either side yields score 0.
3. Scores the agent's answer text against ``answer_raw`` via difflib
   similarity and a normalised substring check.
4. Writes one row per question to a CSV.  Any per-question exception is
   logged, recorded in the ``error`` column, and scored 0 — execution
   continues with the next question.

Usage::

    uv run python -m dev_tools.evaluation.eval_chatbot \
        [--input PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import logging
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from nemo_retriever.tabular_data.sql_database import SQLDatabase

from gsf.retrieval.text_to_sql.main import get_agent_response
from gsf.retrieval.text_to_sql.state import TextToSQLPayload

from gsf.env import load_env
from gsf.connectors import get_connectors
from gsf.utils import get_data_objects_retriever, get_semantic_objects_retriever

load_env()

# The text-to-SQL agent stores executed-DB rows under this key on its result dict.
_DB_RESULT_KEY = "sql_response_from_db"

logger = logging.getLogger("eval_chatbot")

_NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
if not _NVIDIA_API_KEY:
    raise EnvironmentError(
        "NVIDIA_API_KEY is not set. "
        "Export it before running:\n\n"
        "    export NVIDIA_API_KEY='nvapi-...'\n\n"
        "Get your key at https://build.nvidia.com"
    )

# Match the chat server's wiring (gsf/server/chat/helpers.py): same retriever
# and pgvector store. Anything else here and scoring stops being apples-to-apples
# with production.

_DEFAULT_INPUT = Path(__file__).parent / "chatbot_evaluation.json"


_DEFAULT_OUTPUT = Path(__file__).parent / "chatbot_evaluation_scores.csv"


# -----------------------------------------------------------------------------
# Scoring helpers
# -----------------------------------------------------------------------------


def _normalize_text(s: str) -> str:
    """Lowercase, collapse whitespace, drop trailing semicolons."""
    if s is None:
        return ""
    s = str(s).strip().rstrip(";")
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def _sql_text_similarity(expected: str, actual: str) -> float:
    a = _normalize_text(expected)
    b = _normalize_text(actual)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _df_values_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Compare two DataFrames by row-multiset of values, ignoring column names/order."""
    try:
        if a.shape != b.shape:
            return False
        a_rows = sorted(tuple(_canonical(v) for v in row) for row in a.values.tolist())
        b_rows = sorted(tuple(_canonical(v) for v in row) for row in b.values.tolist())
        return a_rows == b_rows
    except Exception:
        return False


def _canonical(value: Any) -> Any:
    """Make a value hashable and comparable across small numeric/string drift."""
    if value is None:
        return None
    if isinstance(value, float):
        # Round to mitigate float jitter from aggregations
        return round(value, 4)
    return str(value).strip().lower()


def _execute_sql(
    connector: SQLDatabase, sql: str
) -> Tuple[Optional[pd.DataFrame], str]:
    if not sql or not sql.strip():
        return None, "empty SQL"
    try:
        df = connector.execute(sql)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        return df, ""
    except Exception as exc:  # pragma: no cover - tooling script
        return None, f"{type(exc).__name__}: {exc}"


def _score_sql(connector: SQLDatabase, expected: str, actual: str) -> Dict[str, Any]:
    text_sim = _sql_text_similarity(expected, actual)
    expected_df, expected_err = _execute_sql(connector, expected)
    actual_df, actual_err = _execute_sql(connector, actual)
    exec_match = 0
    if expected_df is not None and actual_df is not None:
        exec_match = 1 if _df_values_equal(expected_df, actual_df) else 0
    return {
        "sql_text_similarity": round(text_sim, 4),
        "sql_exec_match": exec_match,
        "expected_sql_error": expected_err,
        "returned_sql_error": actual_err,
        "expected_sql_result": _stringify_db_result(expected_df)
        if expected_df is not None
        else "",
    }


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_numbers(text: str) -> List[float]:
    if not text:
        return []
    out = []
    for tok in _NUM_RE.findall(str(text)):
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def _parse_markdown_table(md: str) -> Optional[pd.DataFrame]:
    """Parse a simple markdown table into a DataFrame, or None on failure."""
    if not md:
        return None
    lines = [ln.strip() for ln in md.strip().splitlines() if ln.strip()]
    # Need at least header + separator + one data row
    if len(lines) < 3:
        return None
    data_lines = [ln for ln in lines if not re.match(r"^\|[\s:_-]+\|$", ln)]
    if len(data_lines) < 2:
        return None
    header = [c.strip() for c in data_lines[0].strip("|").split("|")]
    rows = []
    for line in data_lines[1:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == len(header):
            rows.append(cells)
    if not rows:
        return None
    return pd.DataFrame(rows, columns=pd.Index(header))


def _db_result_to_df(value: str) -> Optional[pd.DataFrame]:
    """Try to parse the stringified DB result into a DataFrame."""
    if not value:
        return None
    text = str(value).strip()
    # Unwrap outer list wrapper like ['[{"count":712}]']
    if text.startswith("[") and text.endswith("]"):
        try:
            outer = json.loads(text)
            if (
                isinstance(outer, list)
                and len(outer) == 1
                and isinstance(outer[0], str)
            ):
                text = outer[0]
        except (json.JSONDecodeError, TypeError):
            pass
    # Try JSON array of objects
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return pd.DataFrame(parsed)
        if isinstance(parsed, dict):
            return pd.DataFrame([parsed])
    except (json.JSONDecodeError, TypeError):
        pass
    # Try CSV
    try:
        from io import StringIO

        df = pd.read_csv(StringIO(text))
        if not df.empty:
            return df
    except Exception:
        pass
    return None


_SCORE_STR_LIMIT = 8000


def _score_answer(expected_raw: str, returned_db_str: str) -> Dict[str, Any]:
    """Score the agent's answer against the expected ``answer_raw`` markdown table.

    Strategy (in priority order):
    1. Parse both sides into DataFrames and compare row-multisets (structural match).
    2. Compare the multiset of numeric values (survives formatting differences).
    3. Fall back to fuzzy text similarity.
    """
    expected_df = _parse_markdown_table(expected_raw)
    if expected_df is None:
        expected_df = _db_result_to_df(expected_raw)
    actual_df = _db_result_to_df(returned_db_str)

    structural_match = 0
    if expected_df is not None and actual_df is not None:
        structural_match = 1 if _df_values_equal(expected_df, actual_df) else 0

    # Truncate large strings before expensive text operations
    haystack = str(returned_db_str or "")[:_SCORE_STR_LIMIT]
    expected_capped = str(expected_raw or "")[:_SCORE_STR_LIMIT]

    expected_nums = sorted(round(n, 4) for n in _extract_numbers(expected_capped))
    actual_nums = sorted(round(n, 4) for n in _extract_numbers(haystack))
    if not expected_nums and not actual_nums:
        nums_match = 1
    else:
        nums_match = 1 if expected_nums and expected_nums == actual_nums else 0

    sim = (
        difflib.SequenceMatcher(
            None, _normalize_text(expected_capped), _normalize_text(haystack)
        ).ratio()
        if expected_capped and haystack
        else 0.0
    )
    if structural_match:
        sim = 1.0
    elif nums_match:
        sim = max(sim, 1.0)

    return {
        "answer_text_similarity": round(sim, 4),
        "answer_numbers_match": nums_match,
    }


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def _load_questions(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(
            f"Evaluation file not found: {path}\n"
            f"Pass --input <path> or create the default file at "
            f"{_DEFAULT_INPUT}."
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array of questions, got {type(data).__name__}"
        )
    return data


_STRINGIFY_ROW_LIMIT = 200


def _stringify_db_result(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, pd.DataFrame):
        return value.head(_STRINGIFY_ROW_LIMIT).to_csv(index=False)
    # Some agent paths return executed rows as a plain ``list[dict]`` — serialise
    # as JSON so the downstream parser hits the JSON branch (rather than
    # ``str(...)`` which uses single quotes and breaks json.loads).
    if isinstance(value, list) and value and isinstance(value[0], dict):
        try:
            return json.dumps(value[:_STRINGIFY_ROW_LIMIT], default=str)
        except (TypeError, ValueError):
            return str(value[:_STRINGIFY_ROW_LIMIT])
    return str(value)[:_SCORE_STR_LIMIT]


CSV_FIELDS = [
    "row_index",
    "question_id",
    "difficulty",
    "question",
    "expected_sql",
    "returned_sql",
    "sql_text_similarity",
    "sql_exec_match",
    "expected_sql_error",
    "returned_sql_error",
    "expected_sql_result",
    "expected_answer_raw",
    "returned_answer",
    "answer_text_similarity",
    "answer_numbers_match",
    "runtime_seconds",
    "error",
]


def _print_agent_result(
    qid: Any, question: str, agent_result: Dict[str, Any] | None, expected_sql: str = ""
) -> None:
    """Pretty-print the agent result to stdout for quick visual inspection."""
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  Question {qid}: {question}")
    print(sep)
    if expected_sql:
        print("\n  [expected_sql]")
        for line in expected_sql.splitlines():
            print(f"    {line}")
    if not agent_result:
        print("  (no result)")
        print(sep)
        return
    for key in ("sql_code", "response", "sql_response_from_db"):
        val = agent_result.get(key)
        if val is None:
            continue
        print(f"\n  [{key}]")
        for line in str(val).splitlines():
            print(f"    {line}")
    remaining = {
        k: v
        for k, v in agent_result.items()
        if k not in ("sql_code", "response", "sql_response_from_db")
    }
    if remaining:
        print("\n  [other keys]")
        for k, v in remaining.items():
            print(f"    {k}: {v}")
    print(sep)


def evaluate(
    input_path: Path,
    output_path: Path,
    start_index: int = 0,
    end_index: int | None = None,
) -> None:
    all_questions = _load_questions(input_path)
    questions = all_questions[start_index:end_index]
    logger.info(
        "Running questions %d–%d (%d of %d total) from %s",
        start_index,
        start_index + len(questions) - 1,
        len(questions),
        len(all_questions),
        input_path,
    )   

    connectors = get_connectors()
    data_retriever = get_data_objects_retriever()
    semantic_retriever = get_semantic_objects_retriever()

    resuming = start_index > 0 and output_path.exists()
    mode = "a" if resuming else "w"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not resuming:
            writer.writeheader()

        for idx, item in enumerate(questions, start=start_index):
            qid = item.get("question_id", idx)
            question = item.get("question", "")
            expected_sql = item.get("SQL", "")
            expected_answer = item.get("answer_raw", "")
            difficulty = item.get("difficulty", "")
            logger.info("[%d/%d] q%s: %s", idx + 1, len(questions), qid, question)

            row: Dict[str, Any] = {
                "row_index": idx,
                "question_id": qid,
                "difficulty": difficulty,
                "question": question,
                "expected_sql": expected_sql,
                "returned_sql": "",
                "sql_text_similarity": 0.0,
                "sql_exec_match": 0,
                "expected_sql_error": "",
                "returned_sql_error": "",
                "expected_answer_raw": expected_answer,
                "returned_answer": "",
                "answer_text_similarity": 0.0,
                "answer_numbers_match": 0,
                "runtime_seconds": "",
                "error": "",
            }

            t0 = time.perf_counter()
            try:
                payload: TextToSQLPayload = {
                    "question": question,
                    "data_retriever": data_retriever,
                    "semantic_retriever": semantic_retriever,
                    "connectors": connectors,
                    "path_state": {},
                    "custom_prompts": "",
                    "acronyms": [],
                }
                agent_result = get_agent_response(payload)
                _print_agent_result(qid, question, agent_result, expected_sql)
                returned_sql = (agent_result or {}).get("sql_code", "") or ""
                returned_db = (agent_result or {}).get(_DB_RESULT_KEY)
                returned_db_str = _stringify_db_result(returned_db)

                row["returned_sql"] = returned_sql
                row["returned_answer"] = returned_db_str

                row.update(_score_sql(connectors[0], expected_sql, returned_sql))
                row.update(_score_answer(expected_answer, returned_db_str))
            except Exception as exc:
                logger.exception("Question %s failed", qid)
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["error"] += (
                    " | " + traceback.format_exc().replace("\n", " | ")[:1000]
                )
            finally:
                row["runtime_seconds"] = round(time.perf_counter() - t0, 2)
                writer.writerow(row)
                f.flush()

    logger.info("Wrote scores to %s", output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help=f"Input JSON path (default: {_DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--consistency",
        action="store_true",
        default=False,
        help="Run consistency evaluation (repeat N times and report SQL/answer stability).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of runs for consistency evaluation (default: 10).",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        default=False,
        help="Run a single hardcoded query (edit SINGLE_QUERY in the script).",
    )
    return parser.parse_args()


def _write_consistency_csv(
    csv_path: Path,
    questions: list,
    results: Dict[int, list],
    completed_runs: int,
    start_index: int,
) -> None:
    """Write/overwrite the consistency CSV with all data collected so far."""
    fieldnames = ["question_id", "question"]
    for r in range(1, completed_runs + 1):
        fieldnames.extend([f"sql_run_{r}", f"answer_run_{r}"])
    fieldnames.extend(["sql_consistency", "answer_consistency"])

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for q_idx, item in enumerate(questions):
            qid = item.get("question_id", start_index + q_idx)
            question = item.get("question", "")
            run_results = results[q_idx]

            row: Dict[str, Any] = {"question_id": qid, "question": question}

            first_sql: Dict[str, int] = {}
            first_answer: Dict[str, int] = {}

            for r_idx, r in enumerate(run_results):
                run_num = r_idx + 1
                sql_val = r["sql"]
                ans_val = r["answer"]

                if sql_val in first_sql:
                    row[f"sql_run_{run_num}"] = f"same as run {first_sql[sql_val]}"
                else:
                    first_sql[sql_val] = run_num
                    row[f"sql_run_{run_num}"] = sql_val

                if ans_val in first_answer:
                    row[f"answer_run_{run_num}"] = (
                        f"same as run {first_answer[ans_val]}"
                    )
                else:
                    first_answer[ans_val] = run_num
                    row[f"answer_run_{run_num}"] = ans_val

            sql_counts = {}
            answer_counts = {}
            for r in run_results:
                sql_counts[r["sql"]] = sql_counts.get(r["sql"], 0) + 1
                answer_counts[r["answer"]] = answer_counts.get(r["answer"], 0) + 1
            row["sql_consistency"] = (
                f"{max(sql_counts.values())}/{completed_runs}" if sql_counts else ""
            )
            row["answer_consistency"] = (
                f"{max(answer_counts.values())}/{completed_runs}"
                if answer_counts
                else ""
            )

            writer.writerow(row)


def evaluate_consistency(
    input_path: Path,
    output_path: Path,
    start_index: int = 0,
    end_index: int | None = None,
    runs: int = 10,
) -> None:
    """Run each question multiple times and report SQL/answer consistency."""
    all_questions = _load_questions(input_path)
    questions = all_questions[start_index:end_index]
    logger.info(
        "Consistency eval: %d questions, %d runs each, output=%s",
        len(questions),
        runs,
        output_path,
    )

    connectors = get_connectors()
    data_retriever = get_data_objects_retriever()
    semantic_retriever = get_semantic_objects_retriever()

    results: Dict[int, list] = {i: [] for i in range(len(questions))}

    for run_num in range(1, runs + 1):
        print(f"\n{'=' * 60}")
        print(f"  RUN {run_num}/{runs}")
        print(f"{'=' * 60}")

        for q_idx, item in enumerate(questions):
            qid = item.get("question_id", start_index + q_idx)
            question = item.get("question", "")
            logger.info("[Run %d] q%s: %s", run_num, qid, question)

            try:
                payload: TextToSQLPayload = {
                    "question": question,
                    "data_retriever": data_retriever,
                    "semantic_retriever": semantic_retriever,
                    "connectors": connectors,
                    "path_state": {},
                    "custom_prompts": "",
                    "acronyms": [],
                }
                agent_result = get_agent_response(payload)
                returned_sql = _normalize_text(
                    (agent_result or {}).get("sql_code", "") or ""
                )
                returned_db = (agent_result or {}).get(_DB_RESULT_KEY)
                returned_db_str = _stringify_db_result(returned_db)
            except Exception as exc:
                logger.exception("Run %d, question %s failed", run_num, qid)
                returned_sql = f"ERROR: {exc}"
                returned_db_str = ""

            results[q_idx].append({"sql": returned_sql, "answer": returned_db_str})

        print(f"\n--- After run {run_num} ---")
        for q_idx, item in enumerate(questions):
            qid = item.get("question_id", start_index + q_idx)
            run_results = results[q_idx]
            sqls = [r["sql"] for r in run_results]
            answers = [r["answer"] for r in run_results]
            print(
                f"  q{qid}: {len(sqls)} runs -> "
                f"{len(set(sqls))} unique SQLs, "
                f"{len(set(answers))} unique answers"
            )

        _write_consistency_csv(output_path, questions, results, run_num, start_index)
        logger.info("Updated consistency CSV: %s (after run %d)", output_path, run_num)

    print(f"\n{'=' * 60}")
    print(f"  CONSISTENCY SUMMARY ({runs} runs)")
    print(f"{'=' * 60}")

    for q_idx, item in enumerate(questions):
        qid = item.get("question_id", start_index + q_idx)
        question = item.get("question", "")
        run_results = results[q_idx]

        sql_counts: Dict[str, int] = {}
        answer_counts: Dict[str, int] = {}
        sql_to_answer: Dict[str, str] = {}

        for r in run_results:
            sql_counts[r["sql"]] = sql_counts.get(r["sql"], 0) + 1
            answer_counts[r["answer"]] = answer_counts.get(r["answer"], 0) + 1
            sql_to_answer[r["sql"]] = r["answer"]

        print(f"\n  q{qid}: {question}")
        print(f"  {'─' * 50}")
        for sql, count in sorted(sql_counts.items(), key=lambda x: -x[1]):
            print(f"    SQL ({count}/{runs}): {sql[:500]}")
            print(f"    Answer: {sql_to_answer[sql][:500]}")
            print()

        most_common_sql = max(sql_counts.values())
        most_common_answer = max(answer_counts.values())
        print(f"    -> SQL consistency:    {most_common_sql}/{runs}")
        print(f"    -> Answer consistency: {most_common_answer}/{runs}")

    logger.info("Final consistency CSV: %s", output_path)


def run_single_query(question: str) -> None:
    """Run a single question through the agent and print the result."""
    connectors = get_connectors()
    data_retriever = get_data_objects_retriever()
    semantic_retriever = get_semantic_objects_retriever()

    payload: TextToSQLPayload = {
        "question": question,
        "data_retriever": data_retriever,
        "semantic_retriever": semantic_retriever,
        "connectors": connectors,
        "path_state": {},
        "custom_prompts": "",
        "acronyms": [],
    }
    t0 = time.perf_counter()
    agent_result = get_agent_response(payload)
    elapsed = round(time.perf_counter() - t0, 2)

    _print_agent_result("single", question, agent_result)
    print(f"\n  Runtime: {elapsed}s")


SINGLE_QUERY = "What is the most frequently used GPU MODS version?"

START_INDEX = 0
END_INDEX = None  # None = run to the end
CONSISTENCY_RUNS = 10
RUN_CONSISTENCY = False

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    if args.single:
        run_single_query(SINGLE_QUERY)
    elif RUN_CONSISTENCY or args.consistency:
        num_runs = args.runs if args.consistency else CONSISTENCY_RUNS
        consistency_output = Path(__file__).parent / "chatbot_consistency_scores.csv"
        evaluate_consistency(
            input_path=args.input,
            output_path=consistency_output,
            start_index=START_INDEX,
            end_index=END_INDEX,
            runs=num_runs,
        )
    else:
        evaluate(
            input_path=args.input,
            output_path=args.output,
            start_index=START_INDEX,
            end_index=END_INDEX,
        )
