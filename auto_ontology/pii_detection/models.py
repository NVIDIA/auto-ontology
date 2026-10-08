# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Value objects shared by PII detectors and persistence orchestration."""

from dataclasses import dataclass
from enum import StrEnum


class PiiStatus(StrEnum):
    """Classification outcome for one catalog column."""

    PII = "pii"
    NOT_PII = "not_pii"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class ColumnInput:
    """Metadata safe to provide to a PII classifier."""

    column_name: str
    table_name: str | None = None
    description: str | None = None


@dataclass(frozen=True, slots=True)
class PiiDecision:
    """One detector decision, including enough context for operational logs."""

    status: PiiStatus
    category: str | None
    confidence: float
    reason: str
    source: str

    @property
    def is_pii(self) -> bool:
        """Whether this decision classifies the column as PII."""

        return self.status is PiiStatus.PII

    def should_auto_tag(self, threshold: float = 0.9) -> bool:
        """Whether this decision is safe to turn into an automatic tag."""

        return self.is_pii and self.confidence >= threshold
