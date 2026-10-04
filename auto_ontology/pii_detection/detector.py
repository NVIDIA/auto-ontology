# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rules-first PII detection with a configured LLM fallback."""

import json
import logging
import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator

from auto_ontology.pii_detection.models import ColumnInput, PiiDecision, PiiStatus
from auto_ontology.pii_detection.rules import evaluate_rules
from auto_ontology.utils.llm_invoke import (
    get_llm_client,
    invoke_with_structured_output,
)

logger = logging.getLogger(__name__)

PII_LLM_MAX_WORKERS = int(os.environ.get("PII_LLM_MAX_WORKERS", "6"))

SYSTEM_PROMPT = """\
You classify database columns for personally identifiable information (PII).
Use the column name, table name, and description together.
PII is information that identifies a person directly or can identify them when combined
with other data. Do not classify business, product, or aggregate information as PII.
If context is insufficient or you are unsure whether the column is PII, classify it as
"pii". This is a conservative policy: uncertainty must be treated as PII, not "review".
Always return status, category, confidence, and a short reason.
Never include hidden reasoning."""


class PiiBackend(Protocol):
    """A secondary classifier used when deterministic rules cannot decide."""

    def classify(self, column: ColumnInput) -> PiiDecision: ...


class _PiiResponse(BaseModel):
    """Strict structured output expected from the configured project LLM."""

    model_config = ConfigDict(extra="forbid")

    status: PiiStatus = Field(description="pii, not_pii, or review")
    category: str | None = Field(
        default=None, description="Short snake_case PII category, or null"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence from 0 to 1"
    )
    reason: str = Field(default="", description="One short explanation")

    @field_validator("reason")
    @classmethod
    def _strip_reason(cls, value: str) -> str:
        return value.strip()


class LlmPiiClassifier:
    """Classify uncertain columns with the standard project LLM utilities."""

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self._llm = llm
        self._llm_lock = Lock()

    def _get_llm(self) -> BaseChatModel:
        """Create the shared client once when concurrent classifications begin."""

        if self._llm is None:
            with self._llm_lock:
                if self._llm is None:
                    self._llm = get_llm_client(temperature=0.0, max_tokens=300)
        return self._llm

    def classify(self, column: ColumnInput) -> PiiDecision:
        """Classify one column using metadata only, never sampled values."""

        payload = {
            "column_name": column.column_name,
            "table_name": column.table_name,
            "description": column.description,
        }
        response = invoke_with_structured_output(
            self._get_llm(),
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(payload)),
            ],
            _PiiResponse,
        )
        if response is None:
            raise RuntimeError("PII classifier returned no valid response")

        category = response.category
        status = response.status
        confidence = response.confidence
        if status is PiiStatus.REVIEW:
            # Illumex's PII policy is deliberately conservative: an uncertain
            # classification is positive. Normalise here as well as prompting
            # for it so a model returning REVIEW cannot bypass that policy.
            status = PiiStatus.PII
            confidence = 1.0
        if status is PiiStatus.PII and not category:
            category = "other"

        return PiiDecision(
            status=status,
            category=category,
            confidence=confidence,
            reason=response.reason or f"LLM classified the column as {status.value}.",
            source="llm",
        )


class PiiDetector:
    """Classify columns deterministically before consulting an LLM backend."""

    def __init__(
        self,
        backend: PiiBackend | None = None,
        *,
        max_workers: int = PII_LLM_MAX_WORKERS,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self._backend = backend
        self._max_workers = max_workers

    def detect(self, column: ColumnInput) -> PiiDecision:
        """Classify one column without allowing backend failures to escape."""

        rule_decision = evaluate_rules(column)
        if rule_decision is not None:
            return rule_decision

        if self._backend is None:
            return PiiDecision(
                status=PiiStatus.REVIEW,
                category=None,
                confidence=0.0,
                reason="No deterministic rule matched and no classifier is configured.",
                source="fallback",
            )

        try:
            return self._backend.classify(column)
        except Exception as error:
            logger.exception("PII classifier failed for column %s", column.column_name)
            return PiiDecision(
                status=PiiStatus.REVIEW,
                category=None,
                confidence=0.0,
                reason=f"Classifier failed: {type(error).__name__}.",
                source="fallback",
            )

    def detect_many(
        self, columns: Iterable[ColumnInput]
    ) -> list[tuple[ColumnInput, PiiDecision]]:
        """Classify columns while deduplicating identical metadata."""

        requested = list(columns)
        unique = list(dict.fromkeys(requested))
        if not unique:
            return []
        if self._backend is None or self._max_workers == 1:
            decisions = [self.detect(column) for column in unique]
        else:
            workers = min(self._max_workers, len(unique))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                decisions = list(pool.map(self.detect, unique))

        cache = dict(zip(unique, decisions, strict=True))
        return [(column, cache[column]) for column in requested]
