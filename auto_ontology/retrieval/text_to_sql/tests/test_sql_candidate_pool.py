# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the multi-candidate SQL pool in sql_from_semantic.py.

The property that matters is that slots diverge: same question, same schema,
different derivation. These tests assert on the instructions each slot is
handed rather than on generated SQL, since the LLM is mocked.
"""

from __future__ import annotations

import importlib
import random
from types import ModuleType, SimpleNamespace

import pytest
from langchain_core.messages import SystemMessage

from auto_ontology.utils import llm_invoke
from auto_ontology.retrieval.text_to_sql import candidate_strategies as strategies
from auto_ontology.retrieval.text_to_sql.agents import sql_from_semantic
from auto_ontology.retrieval.text_to_sql.agents.sql_from_semantic import (
    SQLFromCandidatesAgent,
)
from auto_ontology.retrieval.text_to_sql.models import (
    SQLDecompositionModel,
    SQLGenerationModel,
    SQLQueryPlanModel,
    SyntheticSQLExample,
    SyntheticSQLExamplesModel,
)

# A candidate slot builds its own sampling client through ``get_llm_client``,
# which refuses an empty ``REASONING_API_KEY`` before any request is made.
# Nothing here reaches the network — the invoke is always stubbed — so a
# placeholder is enough, and without it every test that runs a slot fails
# wherever the environment carries no key, CI included.
llm_invoke._API_KEY = "test-key"
llm_invoke._BASE_URL = "https://example.test/v1"
llm_invoke._MODEL_NAME = "openai/test-model"


def _state() -> dict:
    return {
        "llm": object(),
        "initial_question": "how many schools are charter schools?",
        "messages": [],
        "path_state": {
            "target_db": "california_schools",
            "primary_attribute": {
                "attr_name": "Charter",
                "col_name": "Charter",
                "table_name": "schools",
            },
            "attribute_join_paths": [],
            "relevant_tables": [
                {
                    "name": "schools",
                    "schema_name": "main",
                    "database_name": "california_schools",
                    "columns": [
                        {"name": "CDSCode", "data_type": "text", "description": ""},
                        {"name": "Charter", "data_type": "int", "description": ""},
                    ],
                },
                {
                    "name": "frpm",
                    "schema_name": "main",
                    "database_name": "california_schools",
                    "columns": [
                        {"name": "CDSCode", "data_type": "text", "description": ""}
                    ],
                },
            ],
        },
    }


def _ok_sql(sql: str = "SELECT COUNT(*) FROM schools") -> SQLGenerationModel:
    return SQLGenerationModel(
        thought="counting", sql_code=sql, response="Counts the schools."
    )


class _Recorder:
    """Stands in for the LLM, returning whatever each schema asks for."""

    def __init__(self, sql_per_call: list | None = None):
        self.calls: list[tuple[list, type]] = []
        self.sql_per_call = sql_per_call

    def __call__(self, _client, messages, schema):
        self.calls.append((messages, schema))
        if schema is SQLQueryPlanModel:
            return SQLQueryPlanModel(plan="1. read schools 2. count")
        if schema is SQLDecompositionModel:
            return SQLDecompositionModel(
                sub_questions=["which schools are charter", "how many"],
                composition="count the filtered rows",
            )
        if schema is SyntheticSQLExamplesModel:
            return SyntheticSQLExamplesModel(
                examples=[
                    SyntheticSQLExample(question="how many rows", sql="SELECT 1"),
                    SyntheticSQLExample(question="list codes", sql="SELECT 2"),
                ]
            )
        if self.sql_per_call is not None:
            return self.sql_per_call.pop(0)
        return _ok_sql()

    def strategy_texts(self) -> list[str]:
        """Every system message that carries a stage instruction."""
        found = []
        for messages, _schema in self.calls:
            for message in messages:
                if isinstance(message, SystemMessage) and (
                    "STAGE 1 OF 2" in message.content
                    or "STAGE 2 OF 2" in message.content
                    or "SCHEMA DIVERGENCE" in message.content
                ):
                    found.append(message.content)
        return found


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(sql_from_semantic, "safe_invoke_with_structured_output", rec)
    monkeypatch.setattr(
        sql_from_semantic, "_run_sql", lambda sql, connector: _Accepted()
    )
    return rec


class _Accepted:
    error = None
    result = ["[]"]


def test_single_candidate_is_the_untouched_path(recorder, monkeypatch):
    """An unset BIRD_NCAND must not add a strategy message anywhere."""
    monkeypatch.delenv("BIRD_NCAND", raising=False)

    out = SQLFromCandidatesAgent().execute(_state())

    assert out["decision"] == "constructable"
    assert len(recorder.calls) == 1
    assert recorder.strategy_texts() == []
    assert len(out["path_state"]["sql_candidates"]) == 1


def test_pool_runs_one_distinct_strategy_per_slot(recorder, monkeypatch):
    monkeypatch.setenv("BIRD_NCAND", "6")

    out = SQLFromCandidatesAgent().execute(_state())

    assert len(out["path_state"]["sql_candidates"]) == 6
    texts = " || ".join(recorder.strategy_texts())
    # Two-stage strategies show both of their stages; alt_table_set is one-shot.
    assert "QUERY PLAN ONLY" in texts
    assert "TRANSLATE THE PLAN TO SQL" in texts
    assert "DECOMPOSITION ONLY" in texts
    assert "ASSEMBLE THE FINAL SQL" in texts
    assert "SAME-SCHEMA DEMONSTRATIONS ONLY" in texts
    assert "SOLVE THE TARGET QUESTION" in texts
    assert "change the base table set" in texts


def test_a_slot_that_fails_drops_out_of_the_pool(monkeypatch):
    """One dead slot must cost one candidate, not the whole question."""
    monkeypatch.setenv("BIRD_NCAND", "3")
    calls = {"n": 0}

    def flaky(_client, messages, schema):
        if schema is SQLQueryPlanModel:
            return SQLQueryPlanModel(plan="p")
        if schema is SQLDecompositionModel:
            raise RuntimeError("gateway down")
        calls["n"] += 1
        return _ok_sql()

    monkeypatch.setattr(sql_from_semantic, "safe_invoke_with_structured_output", flaky)

    out = SQLFromCandidatesAgent().execute(_state())

    assert out["decision"] == "constructable"
    assert len(out["path_state"]["sql_candidates"]) == 2


def test_unusable_candidates_are_filtered_before_selection(monkeypatch):
    monkeypatch.setenv("BIRD_NCAND", "2")

    def half_empty(_client, messages, schema):
        if schema is SQLQueryPlanModel:
            return SQLQueryPlanModel(plan="p")
        return _ok_sql() if schema is SQLGenerationModel else None

    monkeypatch.setattr(
        sql_from_semantic, "safe_invoke_with_structured_output", half_empty
    )
    out = SQLFromCandidatesAgent().execute(_state())
    assert all(c.sql_code.strip() for c in out["path_state"]["sql_candidates"])


def test_every_slot_generating_nothing_is_unconstructable(monkeypatch):
    monkeypatch.setenv("BIRD_NCAND", "2")
    monkeypatch.setattr(
        sql_from_semantic,
        "safe_invoke_with_structured_output",
        lambda *a, **k: None,
    )

    out = SQLFromCandidatesAgent().execute(_state())

    assert out["decision"] == "unconstructable"


# --- strategy helpers -------------------------------------------------------


def test_slot_zero_stays_baseline_and_deterministic():
    assert strategies.strategy_for_slot(0, 6) == "baseline"
    assert strategies.slot_samples(0, 6) is False
    assert strategies.slot_samples(1, 6) is True
    # A single-candidate run never samples, whatever the slot index.
    assert strategies.slot_samples(0, 1) is False


def test_pin_forces_every_slot_including_zero(monkeypatch):
    monkeypatch.setenv("BIRD_PIN_STRATEGY", "decomposition")
    assert [strategies.strategy_for_slot(i, 3) for i in range(3)] == [
        "decomposition"
    ] * 3
    # Slot 0 must sample too, or one draw of N would be deterministic.
    assert strategies.slot_samples(0, 3) is True


def test_unknown_pin_falls_back_to_round_robin(monkeypatch):
    monkeypatch.setenv("BIRD_PIN_STRATEGY", "not_a_strategy")
    assert strategies.strategy_for_slot(1, 6) == "query_plan"


def test_slot_plan_overrides_the_round_robin(monkeypatch):
    monkeypatch.setenv("BIRD_SLOT_PLAN", "baseline,synthetic_examples")
    assert [strategies.strategy_for_slot(i, 4) for i in range(4)] == [
        "baseline",
        "synthetic_examples",
    ] * 2


def test_unknown_slot_plan_is_ignored(monkeypatch):
    monkeypatch.setenv("BIRD_SLOT_PLAN", "baseline,bogus")
    assert strategies.strategy_for_slot(1, 6) == "query_plan"


def test_shuffle_leaves_the_caller_list_untouched():
    tables = [{"name": "a", "columns": [{"name": "x"}, {"name": "y"}]}]
    out = strategies.shuffled_tables(tables, random.Random(0))
    out[0]["name"] = "mutated"
    out[0]["columns"].append({"name": "z"})
    assert tables[0]["name"] == "a"
    assert len(tables[0]["columns"]) == 2


def test_example_rotation_is_complementary():
    examples = list(range(8))
    assert strategies.rotate_examples(examples, 0) == examples
    first = strategies.rotate_examples(examples, 1)
    second = strategies.rotate_examples(examples, 2)
    assert first != second
    assert not set(first) & set(second)


def test_schema_directives_are_opt_in(monkeypatch):
    monkeypatch.delenv("BIRD_SCHEMA_SLOTS", raising=False)
    assert strategies.schema_directive(1, None) == ""

    monkeypatch.setenv("BIRD_SCHEMA_SLOTS", "1")
    # Slot 0 stays free so the reading can only be added, never removed.
    assert strategies.schema_directive(0, None) == ""
    assert "prefer the joined reading" in strategies.schema_directive(1, None)


def test_alternative_binding_needs_named_ambiguity(monkeypatch):
    monkeypatch.setenv("BIRD_SCHEMA_SLOTS", "1")
    # Slot 2 is the alternative-binding role; with nothing ambiguous to point
    # at it must fall back rather than invite an arbitrary swap.
    assert "prefer the joined reading" in strategies.schema_directive(2, [])

    directive = strategies.schema_directive(
        2, [{"entity": "charter", "columns": ["a", "b"]}]
    )
    assert "second-choice binding" in directive
    assert '"charter"' in directive


# --- export to callers ------------------------------------------------------


@pytest.fixture(name="main")
def main_fixture(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the agent entry point without a configured LLM.

    ``main`` builds its reasoning client at import time, so importing it at
    module scope would fail collection on a machine without credentials.
    """
    monkeypatch.setattr(llm_invoke, "get_llm_client", lambda **_kwargs: None)
    return importlib.import_module("auto_ontology.retrieval.text_to_sql.main")


def test_pool_is_exported_as_plain_sql_strings(main: ModuleType) -> None:
    answer = main._extract_answer(
        {
            "path_state": {
                "final_response": {"response": "done"},
                "sql_candidates": [
                    SimpleNamespace(sql_code="SELECT 1"),
                    SimpleNamespace(sql_code="  SELECT 2  "),
                ],
            }
        }
    )

    assert answer["sql_candidates"] == ["SELECT 1", "SELECT 2"]


def test_blank_candidates_are_dropped_from_the_export(main: ModuleType) -> None:
    answer = main._extract_answer(
        {
            "path_state": {
                "final_response": {"response": "done"},
                "sql_candidates": [
                    SimpleNamespace(sql_code="SELECT 1"),
                    SimpleNamespace(sql_code="   "),
                    SimpleNamespace(sql_code=None),
                ],
            }
        }
    )

    assert answer["sql_candidates"] == ["SELECT 1"]


def test_paths_without_a_generator_carry_no_candidates_key(main: ModuleType) -> None:
    """The information and routing paths must keep their exact answer shape."""
    answer = main._extract_answer(
        {"path_state": {"final_response": {"response": "routed"}}}
    )

    assert answer == {"response": "routed"}
