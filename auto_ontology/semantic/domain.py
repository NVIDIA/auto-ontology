# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Domain summary and system prompt loading for seed selection."""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_SUMMARY_DIR = Path(".semantic_summaries")
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class DomainSummary(BaseModel):
    """Compact domain context for seed and Term extraction."""

    domains: list[str] = Field(default_factory=list)
    core_entities: list[str] = Field(default_factory=list)
    business_rules: list[str] = Field(default_factory=list)


def load_domain_summary(database_name: str) -> DomainSummary | None:
    path = _SUMMARY_DIR / f"{database_name}.json"
    if not path.exists():
        return None
    try:
        return DomainSummary.model_validate_json(path.read_text())
    except Exception:
        logger.warning("Failed to load domain summary from %s", path)
        return None


def load_system_prompt() -> str:
    path = _PROMPTS_DIR / "system_prompt.txt"
    if path.exists():
        return path.read_text().strip()
    return (
        "You are compiling a business semantic layer over a relational database. "
        "Use clear, human-readable Term names with spaces between words and precise "
        "business language."
    )
