# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("REASONING_API_KEY", "test-key")
os.environ.setdefault("REASONING_ENDPOINT", "https://example.test/v1")
os.environ.setdefault("REASONING_MODEL", "openai/test-model")

from gsf.utils import llm_invoke  # noqa: E402

llm_invoke._API_KEY = "test-key"
llm_invoke._BASE_URL = "https://example.test/v1"
llm_invoke._MODEL_NAME = "openai/test-model"

from gsf.retrieval.text_to_sql import main  # noqa: E402


class _FailAfterGenerationApp:
    def stream(self, state, config):
        yield {
            "validate_intent": {
                "path_state": {
                    "sql_generation_result": SimpleNamespace(
                        sql_code="SELECT 1",
                        response="ok",
                        thought="generated before downstream failure",
                    )
                }
            }
        }
        raise KeyError("intent_valid")


def test_stream_error_preserves_partial_generated_sql(monkeypatch):
    monkeypatch.setattr(main, "_build_state", lambda payload: {"path_state": {}})
    monkeypatch.setattr(main, "app", _FailAfterGenerationApp())

    events = list(main.stream_agent_response({"question": "q"}))

    assert events[-1] == {
        "type": "error",
        "message": "Agent failed after validate_intent: 'intent_valid'",
        "node": "validate_intent",
        "error_type": "KeyError",
        "partial_answer": {
            "sql_code": "SELECT 1",
            "response": "ok",
            "thought": "generated before downstream failure",
        },
    }


def test_nonstreaming_error_exposes_partial_answer(monkeypatch):
    monkeypatch.setattr(main, "_build_state", lambda payload: {"path_state": {}})
    monkeypatch.setattr(main, "app", _FailAfterGenerationApp())

    with pytest.raises(main.AgentRunError) as caught:
        main.get_agent_response({"question": "q"})

    assert caught.value.node == "validate_intent"
    assert caught.value.partial_answer["sql_code"] == "SELECT 1"
