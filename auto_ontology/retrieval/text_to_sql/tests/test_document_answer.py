# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from auto_ontology.retrieval.text_to_sql import text_to_sql_graph
from auto_ontology.retrieval.text_to_sql.agents import document_answer
from auto_ontology.retrieval.text_to_sql.agents.document_answer import (
    AnswerFromDocumentsAgent,
    DecideDocumentRelevanceAgent,
    DocumentRelevanceDecision,
    RetrieveDocumentCandidatesAgent,
)
from auto_ontology.retrieval.text_to_sql.text_to_sql_graph import (
    route_document_candidates,
    route_document_relevance,
    route_unstructured_flow,
)
from unstructured_ao.search import ChunkOntology, Passage


def _state(path_state: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "initial_question": "What is the leave policy?",
        "messages": [],
        "path_state": path_state or {},
        "llm": object(),
    }


def test_disabled_unstructured_flow_routes_directly_to_structured_data(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        text_to_sql_graph, "is_unstructured_flow_enabled", lambda: False
    )

    assert route_unstructured_flow(_state()) == "structured"


@pytest.mark.parametrize(
    "mode",
    [{"calculation_only": True}, {"prediction_override": True}],
    ids=["calculation", "prediction"],
)
def test_an_explicit_mode_bypasses_enabled_documents(
    monkeypatch: Any, mode: dict[str, bool]
) -> None:
    monkeypatch.setattr(text_to_sql_graph, "is_unstructured_flow_enabled", lambda: True)

    assert route_unstructured_flow(_state()) == "documents"
    assert route_unstructured_flow({**_state(), **mode}) == "structured"


def test_retrieve_candidates_keeps_stable_source_ids(monkeypatch: Any) -> None:
    monkeypatch.setattr(document_answer, "_documents_available", lambda: True)
    monkeypatch.setattr(
        document_answer.search,
        "chunk_search",
        lambda _question, k: [
            Passage(
                text="Employees receive twenty days.",
                attribute_id="",
                label="",
                source="vector",
                score=0.1,
                document="policy.pdf",
                page=4,
                document_id="doc-1",
                chunk_id="chunk-1",
            )
        ],
    )
    annotated: list[list[str | None]] = []

    def chunk_ontology(chunk_ids: list[str | None]) -> ChunkOntology:
        annotated.append(chunk_ids)
        return ChunkOntology(labels=[["Annual leave"]], note="")

    monkeypatch.setattr(document_answer.search, "chunk_ontology", chunk_ontology)

    result = RetrieveDocumentCandidatesAgent().execute(_state())

    assert annotated == [["chunk-1"]]

    candidate = result["path_state"]["document_candidates"][0]
    assert candidate["document_id"] == "doc-1"
    assert candidate["chunk_id"] == "chunk-1"
    assert candidate["label"] == "Annual leave"


def test_retrieve_failure_falls_through_to_structured_flow(monkeypatch: Any) -> None:
    monkeypatch.setattr(document_answer, "_documents_available", lambda: True)

    def fail(_question: str, *, k: int) -> list[Passage]:
        raise RuntimeError("vector table missing")

    monkeypatch.setattr(document_answer.search, "chunk_search", fail)
    result = RetrieveDocumentCandidatesAgent().execute(_state())

    assert result["path_state"]["document_candidates"] == []
    assert result["path_state"]["use_documents"] is False
    assert route_document_candidates(result) == "structured"
    assert route_document_relevance(result) == "structured"


def test_empty_corpus_skips_embedding_and_relevance(monkeypatch: Any) -> None:
    monkeypatch.setattr(document_answer, "_documents_available", lambda: False)

    def unexpected_search(*_args: Any, **_kwargs: Any) -> list[Passage]:
        raise AssertionError("chunk search must not run without documents")

    monkeypatch.setattr(document_answer.search, "chunk_search", unexpected_search)

    result = RetrieveDocumentCandidatesAgent().execute(_state())

    assert result["path_state"]["document_candidates"] == []
    assert route_document_candidates(result) == "structured"


def test_relevance_selects_at_most_three_valid_unique_indexes(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        document_answer,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: DocumentRelevanceDecision(
            use_documents=True,
            selected_chunk_indexes=[0, 0, 99, 1, 2, 3],
            reason="The excerpts directly answer the question.",
        ),
    )
    candidates = [{"text": f"chunk {index}"} for index in range(5)]

    result = DecideDocumentRelevanceAgent().execute(
        _state({"document_candidates": candidates})
    )

    assert result["path_state"]["use_documents"] is True
    assert result["path_state"]["selected_document_chunks"] == [0, 1, 2]
    assert route_document_candidates(result) == "judge"
    assert route_document_relevance(result) == "documents"


def test_relevance_failure_falls_through_to_structured_flow(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        document_answer,
        "invoke_with_structured_output",
        lambda *_args, **_kwargs: None,
    )

    result = DecideDocumentRelevanceAgent().execute(
        _state({"document_candidates": [{"text": "possibly related"}]})
    )

    assert result["path_state"]["use_documents"] is False
    assert route_document_relevance(result) == "structured"


def test_document_answer_returns_cited_terminal_response(monkeypatch: Any) -> None:
    calls: list[tuple[object, str]] = []

    def answer_with_chat_llm(llm: object, prompt: str) -> str:
        calls.append((llm, prompt))
        return "Employees receive twenty days."

    monkeypatch.setattr(
        document_answer,
        "safe_invoke_text",
        answer_with_chat_llm,
    )
    candidate = {
        "text": "Employees receive twenty days.",
        "label": "Annual leave",
        "document_id": "doc-1",
        "chunk_id": "chunk-1",
        "document": "policy.pdf",
        "page": 4,
    }

    result = AnswerFromDocumentsAgent().execute(
        _state(
            {
                "document_candidates": [candidate],
                "selected_document_chunks": [0],
                "document_relation_note": "",
            }
        )
    )

    answer = result["path_state"]["final_response"]
    assert calls[0][0] is not None
    assert "What is the leave policy?" in calls[0][1]
    assert "Employees receive twenty days." in calls[0][1]
    assert answer["answer_source"] == "documents"
    assert answer["sql_code"] == ""
    assert answer["sql_response_from_db"] is None
    assert answer["sources"] == [
        {
            "document_id": "doc-1",
            "chunk_id": "chunk-1",
            "document": "policy.pdf",
            "page": 4,
        }
    ]
