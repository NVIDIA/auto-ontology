# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output-type skip predicate for the interactive clarification pipeline.

Determines whether the output type (scalar vs table) is obvious from the question,
or ambiguous enough to warrant asking the user before clarification begins.

Controlled by the ASK_OUTPUT_TYPE env var:
  - Defaults to enabled when INTERACTIVE=true
  - Set ASK_OUTPUT_TYPE=false/true to disable/enable explicitly regardless of INTERACTIVE

Return values of should_skip_output_type_question():
  "scalar"  → output type is obvious; set scalar hint for SQL gen, skip the question
  "table"   → output type is obvious; skip the question silently (no hint)
  "ddl"     → DDL/DML question; skip the question silently (safety fallback)
  None      → output type is ambiguous; inject the clarification question
"""

from __future__ import annotations

import logging
import os
import re
from typing import Literal, Optional

logger = logging.getLogger(__name__)

# ── Feature flag ─────────────────────────────────────────────────────────────


def output_type_enabled() -> bool:
    """Return True when the output-type clarification question is active.

    ASK_OUTPUT_TYPE=false/0 disables explicitly (overrides INTERACTIVE).
    ASK_OUTPUT_TYPE=true/1 enables explicitly.
    Otherwise falls back to INTERACTIVE (defaults off when unset).
    """
    explicit = os.environ.get("ASK_OUTPUT_TYPE", "").lower()
    if explicit in ("false", "0"):
        return False
    if explicit in ("true", "1"):
        return True
    return os.environ.get("INTERACTIVE", "").lower() in ("true", "1")


# ── Constants ─────────────────────────────────────────────────────────────────

OUTPUT_TYPE_QUESTION = "What type of output are you looking for?"

SCALAR_HINT = (
    "OutputType: this question expects a single aggregate value (no GROUP BY). "
    "Prefer SELECT AVG(...) / SUM(...) / COUNT(*) without breakdown unless the "
    "question explicitly asks for per-item results."
)

# ── Regex patterns (v3, data-driven) ─────────────────────────────────────────

# Scalar override: forces scalar regardless of grouping guard.
# Triggers on "give the" or "output the" anywhere in the question.
_SCALAR_OVERRIDE = re.compile(
    r"\b(give the|output the)\b",
    re.IGNORECASE,
)

# Grouping guard: voids scalar signals when a breakdown dimension is present.
_GROUPING_GUARD = re.compile(
    r"\b("
    r"for each|in each|for every|at each|"
    r"group by|"
    r"each\s+\w+|"
    r"(per|by)\s+"
    r"(year|month|day|week|quarter|"
    r"category|group|region|plant|store|product|team|"
    r"department|class|type|segment)"
    r")\b",
    re.IGNORECASE,
)

# Scalar signals — high-precision, subject to grouping guard.
_SCALAR_SIGNAL = re.compile(
    r"("
    # "what is/was/are the <aggregate>"
    r"\bwhat (is|was|are) the (average|total|sum|count|maximum|minimum|max|min)\b|"
    r"\bwhat'?s the (average|total|sum|count|maximum|minimum|max|min)\b|"
    # "how many / how much"
    r"\bhow (many|much)\b|"
    # "find the <aggregate>"
    r"\bfind the (max|min|average|total|maximum|minimum|sum|count)\b|"
    # "give me the final/overall <number/figure/total/count/value>"
    r"\bgive me the (final|overall) (number|figure|total|count|value|sum|average)\b|"
    # "return the final/overall ..."
    r"\breturn the (final|overall)\b|"
    # "on average"
    r"\bon average\b|"
    # "overall <aggregate>"
    r"\boverall (average|total|sum|count|maximum|minimum|max|min)\b|"
    # "calculate the <aggregate/value/score>"
    r"\bcalculate the (aggregate|total|average|sum|count|max|min|maximum|minimum|value|score)\b"
    r")",
    re.IGNORECASE,
)

# Table signals — ≥90% precision, high-volume patterns.
_TABLE_SIGNAL = re.compile(
    r"("
    r"\b(rank|ranking)\b|"
    r"\b(categorize|segment into)\b|"
    r"\bI need a list\b|"
    r"\ball \w+ (that|which|who)\b|"
    r"\bwhich ones (are|have|do)\b|"
    r"\b(display|show) (all|each|every)\b|"
    r"\bgroup by\b|"
    r"\b(sorted|ordered) by\b|"
    r"\blist (all|each|every|them)\b|"
    r"\bfor each\b|"
    r"\bbreakdown\b"
    r")",
    re.IGNORECASE,
)

# DDL/DML signals — safety fallback for management queries.
_DDL_SIGNAL = re.compile(
    r"\b(create|drop|alter|insert|update|delete|truncate|add column|modify column)\b",
    re.IGNORECASE,
)


# ── Public predicate ──────────────────────────────────────────────────────────


def should_skip_output_type_question(
    question: str,
) -> Optional[Literal["scalar", "table", "ddl"]]:
    """Return the detected output type, or None if the question should be asked.

    Logic (v3, data-driven):
      1. DDL/DML detected → "ddl" (safety fallback)
      2. Scalar override ("give the" / "output the") → "scalar" unconditionally
      3. Grouping guard present → void scalar signals
      4. Unambiguous scalar signal (no table) → "scalar"
      5. Unambiguous table signal (no scalar) → "table"
      6. Ambiguous or no signal → None (ask the user)
    """
    # 1. DDL/DML safety fallback
    if _DDL_SIGNAL.search(question):
        logger.debug("OutputType — DDL/DML detected, skipping question")
        return "ddl"

    # 2. Scalar override: unconditionally forces scalar
    if _SCALAR_OVERRIDE.search(question):
        logger.debug("OutputType — scalar override ('give the'/'output the') detected")
        return "scalar"

    has_scalar = bool(_SCALAR_SIGNAL.search(question))
    has_table = bool(_TABLE_SIGNAL.search(question))

    # 3. Grouping guard voids scalar signals (override already handled above)
    if has_scalar and _GROUPING_GUARD.search(question):
        logger.debug("OutputType — grouping guard voided scalar signal")
        has_scalar = False

    # 4-6. Classify
    if has_scalar and not has_table:
        return "scalar"
    if has_table and not has_scalar:
        return "table"

    # Ambiguous (both or neither)
    return None
