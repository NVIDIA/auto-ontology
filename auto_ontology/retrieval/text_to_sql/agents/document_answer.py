# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Document retrieval, sufficiency routing, and grounded answering nodes."""

from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.messages import AIMessage, SystemMessage
from pydantic import Field

from auto_ontology.dal.session import get_engine
from auto_ontology.retrieval.text_to_sql.base import (
    AgentExecutionError,
    BaseAgent,
    record_thought,
)
from auto_ontology.retrieval.text_to_sql.state import (
    AgentState,
    get_question_for_processing,
)
from auto_ontology.utils.llm_invoke import (
    StrictLLMOutputModel,
    invoke_with_structured_output,
    safe_invoke_text,
)
from unstructured_ao import answer as document_answer
from unstructured_ao import dal as document_dal
from unstructured_ao import search

_CANDIDATE_COUNT = 10
_MAX_SELECTED = 3
_JUDGE_EXCERPT_CHARS = 2500


class DocumentRelevanceDecision(StrictLLMOutputModel):
    """Whether retrieved document evidence can fully answer the question."""

    use_documents: bool
    selected_chunk_indexes: list[int] = Field(default_factory=list)
    reason: str


def _candidate_payload(passage: search.Passage, label: str) -> dict[str, Any]:
    return {
        "text": passage.text,
        "label": label,
        "document_id": passage.document_id,
        "chunk_id": passage.chunk_id,
        "document": passage.document,
        "page": passage.page,
        "score": passage.score,
    }


def _documents_available() -> bool:
    with get_engine().connect() as conn:
        return document_dal.has_retrievable_documents(conn)


class RetrieveDocumentCandidatesAgent(BaseAgent):
    """Retrieve document chunks and annotate them with ontology concepts."""

    def __init__(self) -> None:
        super().__init__("retrieve_document_candidates")

    def execute(self, state: AgentState) -> Dict[str, Any]:
        path_state = dict(state.get("path_state") or {})
        question = get_question_for_processing(state)

        try:
            if _documents_available():
                passages = search.chunk_search(question, k=_CANDIDATE_COUNT)
                found = search.chunk_ontology(
                    [passage.chunk_id for passage in passages]
                )
            else:
                passages = []
                found = search.ChunkOntology()
        except RuntimeError as exc:
            self.logger.info(
                "Document corpus unavailable; continuing to structured flow: %s",
                exc,
            )
            passages = []
            found = search.ChunkOntology()
            path_state["document_retrieval_error"] = (f"{type(exc).__name__}: {exc}")[
                :500
            ]
        except Exception as exc:  # noqa: BLE001 — unavailable corpus falls through to SQL
            self.logger.warning(
                "Document retrieval unavailable; continuing to structured flow: %s",
                exc,
                exc_info=True,
            )
            passages = []
            found = search.ChunkOntology()
            path_state["document_retrieval_error"] = (f"{type(exc).__name__}: {exc}")[
                :500
            ]

        labels = ["; ".join(names) for names in found.labels]
        if len(labels) < len(passages):
            labels.extend([""] * (len(passages) - len(labels)))
        candidates = [
            _candidate_payload(passage, labels[index])
            for index, passage in enumerate(passages)
        ]

        path_state.update(
            {
                "document_candidates": candidates,
                "document_relation_note": found.note,
                "use_documents": False,
                "selected_document_chunks": [],
            }
        )
        record_thought(
            path_state,
            self.agent_name,
            (
                f"Retrieved {len(candidates)} candidate document excerpt"
                f"{'' if len(candidates) == 1 else 's'}."
            ),
        )
        return {"path_state": path_state}


def _relevance_prompt(question: str, candidates: list[dict[str, Any]]) -> str:
    excerpts = [
        {
            "index": index,
            "document": candidate.get("document"),
            "page": candidate.get("page"),
            "label": candidate.get("label"),
            "text": str(candidate.get("text") or "")[:_JUDGE_EXCERPT_CHARS],
        }
        for index, candidate in enumerate(candidates)
    ]
    return f"""Decide whether uploaded-document excerpts contain enough evidence to
answer the user's question accurately and completely.

Topical similarity is not enough. Set use_documents=false when the excerpts are
only generally related, omit a requested fact, or would require outside knowledge.
When sufficient, select at most {_MAX_SELECTED} excerpt indexes that directly support
the answer. Never select an index that is not present. Do not answer the question.

Question:
{question}

Excerpts:
{json.dumps(excerpts, ensure_ascii=False)}
"""


class DecideDocumentRelevanceAgent(BaseAgent):
    """Select sufficient document evidence or fall through to structured data."""

    def __init__(self) -> None:
        super().__init__("decide_document_relevance")

    def execute(self, state: AgentState) -> Dict[str, Any]:
        path_state = dict(state.get("path_state") or {})
        candidates = list(path_state.get("document_candidates") or [])
        if not candidates:
            path_state.update(
                {
                    "use_documents": False,
                    "selected_document_chunks": [],
                    "document_route_reason": "No document excerpts were retrieved.",
                }
            )
            record_thought(
                path_state, self.agent_name, "No document evidence was available."
            )
            return {"path_state": path_state}

        decision = invoke_with_structured_output(
            state["llm"],
            [
                SystemMessage(
                    content=_relevance_prompt(
                        get_question_for_processing(state), candidates
                    )
                )
            ],
            DocumentRelevanceDecision,
        )
        if decision is None:
            path_state.update(
                {
                    "use_documents": False,
                    "selected_document_chunks": [],
                    "document_route_reason": "Document relevance could not be determined.",
                }
            )
            path_state.pop("document_candidates", None)
            path_state.pop("document_relation_note", None)
            record_thought(
                path_state,
                self.agent_name,
                "Document sufficiency could not be established; using structured data.",
            )
            return {"path_state": path_state}

        selected: list[int] = []
        for index in decision.selected_chunk_indexes:
            if 0 <= index < len(candidates) and index not in selected:
                selected.append(index)
            if len(selected) == _MAX_SELECTED:
                break
        use_documents = bool(decision.use_documents and selected)
        path_state.update(
            {
                "use_documents": use_documents,
                "selected_document_chunks": selected if use_documents else [],
                "document_route_reason": decision.reason,
            }
        )
        if not use_documents:
            path_state.pop("document_candidates", None)
            path_state.pop("document_relation_note", None)
        record_thought(path_state, self.agent_name, decision.reason)
        return {"path_state": path_state}


class AnswerFromDocumentsAgent(BaseAgent):
    """Generate the terminal response from the selected document excerpts."""

    def __init__(self) -> None:
        super().__init__("answer_from_documents")

    def execute(self, state: AgentState) -> Dict[str, Any]:
        path_state = dict(state.get("path_state") or {})
        candidates = list(path_state.get("document_candidates") or [])
        indexes = list(path_state.get("selected_document_chunks") or [])
        selected = [
            candidates[index]
            for index in indexes
            if isinstance(index, int) and 0 <= index < len(candidates)
        ]
        if not selected:
            raise AgentExecutionError("Document answer selected without evidence.")

        question = get_question_for_processing(state)
        context = document_answer.build_context(
            [str(candidate.get("text") or "") for candidate in selected],
            labels=[str(candidate.get("label") or "") for candidate in selected],
            note=str(path_state.get("document_relation_note") or "") or None,
        )
        prompt = (
            f"{document_answer.SYSTEM_PROMPT}\n\n"
            f"{document_answer.USER_TEMPLATE.format(context=context, query=question)}"
        )
        try:
            text = safe_invoke_text(state["llm"], prompt).strip()
        except Exception as exc:
            raise AgentExecutionError(
                f"Document answer generation failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not text:
            raise AgentExecutionError("Document answer generation failed: empty answer")

        sources = [
            {
                "document_id": candidate.get("document_id"),
                "chunk_id": candidate.get("chunk_id"),
                "document": candidate.get("document"),
                "page": candidate.get("page"),
            }
            for candidate in selected
        ]
        final_response = {
            "response": text,
            "sql_code": "",
            "sql_response_from_db": None,
            "answer_source": "documents",
            "sources": sources,
        }
        path_state.update(
            {
                "formatted_response": text,
                "document_sources": sources,
                "final_response": final_response,
            }
        )
        path_state.pop("document_candidates", None)
        path_state.pop("document_relation_note", None)
        path_state.pop("selected_document_chunks", None)
        return {
            "messages": list(state.get("messages") or []) + [AIMessage(content=text)],
            "path_state": path_state,
        }


__all__ = [
    "AnswerFromDocumentsAgent",
    "DecideDocumentRelevanceAgent",
    "DocumentRelevanceDecision",
    "RetrieveDocumentCandidatesAgent",
]
