# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conditional-output hint injection for the interactive SQL generation pipeline.

Detects user questions that require a CASE/WHEN expression to produce labeled or
conditional output (rather than a plain filter or aggregation), and injects a
targeted, pattern-specific hint into the evidence block so the SQL generator
emits the correct structure.

Controlled by the INJECT_CONDITIONAL_OUTPUT_HINT env var:
  - Defaults to enabled when INTERACTIVE=true
  - Set INJECT_CONDITIONAL_OUTPUT_HINT=false/true to disable/enable explicitly
    regardless of INTERACTIVE

Three pattern families are recognised, each mapping to its own hint:

  Pattern FALLBACK — explicit conditional branching:
    otherwise · no action · if so

  Pattern STATUS — status / health check:
    check if/whether · determine if/whether · tell me if/whether ·
    show whether · see if/whether

  Pattern CLASSIFICATION — tiering, labeling, group comparison:
    label/tag/mark/flag … each/them/as · classify each/them ·
    bucket/tier/band … into · categorize/categorization ·
    translates to · converts to · and their classification/label/tier ·
    into (quadrant|tier|band|category|class) · versus · vs ·
    compare/comparison · grouped by · for each group · breakdown

Only one hint fires per query (priority: FALLBACK → STATUS → CLASSIFICATION).
False positives receive a specific, targeted hint rather than a generic menu,
so even wrong triggers produce a coherent (if unneeded) instruction.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Literal, Optional

logger = logging.getLogger(__name__)

# ── Feature flag ──────────────────────────────────────────────────────────────


def conditional_output_enabled() -> bool:
    """Return True when the conditional-output hint injection is active.

    INJECT_CONDITIONAL_OUTPUT_HINT=false/0 disables explicitly (overrides
    INTERACTIVE). INJECT_CONDITIONAL_OUTPUT_HINT=true/1 enables explicitly.
    Otherwise falls back to INTERACTIVE (defaults off when unset).
    """
    explicit = os.environ.get("INJECT_CONDITIONAL_OUTPUT_HINT", "").lower()
    if explicit in ("false", "0"):
        return False
    if explicit in ("true", "1"):
        return True
    return os.environ.get("INTERACTIVE", "").lower() in ("true", "1")


# ── Per-pattern hint text ─────────────────────────────────────────────────────

_HINT_FALLBACK = (
    "[Conditional Output Hint]\n"
    'The query contains explicit conditional branching ("otherwise", "if so", "no action").\n'
    "Consider encoding this as:\n"
    "→ CASE WHEN <condition> THEN (<subquery>)::TEXT ELSE '<fallback message>' END\n"
    "  The ELSE branch should be a descriptive string, not NULL or an empty result.\n"
    "  or if the question only asks for a plain yes/no or true/false fact,\n"
    "  with no distinct fallback message actually required, a boolean TRUE/FALSE is\n"
    "  more appropriate."
)

_HINT_STATUS = (
    "[Conditional Output Hint]\n"
    "The query asks to check a condition and return a result for both outcomes.\n"
    "Consider encoding this as:\n"
    "→ CASE WHEN <condition> THEN 'Positive status' ELSE 'Negative status' END\n"
    "  (not a WHERE filter — both branches must produce output, not just the matching rows)\n"
    "  or if no descriptive label is implied, a boolean TRUE/FALSE \n"
    "  or a combination: CASE WHEN <condition> THEN TRUE ELSE FALSE END."
)

_HINT_CLASSIFICATION = (
    "[Conditional Output Hint]\n"
    "The query asks to label, group, or compare rows into named categories.\n"
    "Consider encoding this as:\n"
    "→ CASE WHEN score > 80 THEN 'High' WHEN score > 50 THEN 'Medium' ELSE 'Low' END\n"
    "→ CASE WHEN <explicit condition> THEN 'Group A' ELSE 'Group B' END\n"
    "  The ELSE branch must name the default group — do not leave it NULL."
)

# ── Regex trigger patterns ────────────────────────────────────────────────────

# Pattern FALLBACK — explicit conditional branching ("otherwise", "no action", "if so")
_FALLBACK_SIGNAL = re.compile(
    r"\b(otherwise|no action|if so)\b",
    re.IGNORECASE,
)

# Pattern STATUS — status / health check ("check if/whether", "determine if/whether", …)
_STATUS_CHECK_SIGNAL = re.compile(
    r"\b(check|determine|tell me|show|see)\s+(if|whether)\b",
    re.IGNORECASE,
)

# Pattern CLASSIFICATION — tiering, labeling, group comparison
_CLASSIFICATION_SIGNAL = re.compile(
    r"("
    # labeling verbs applied to rows
    r"\b(label|tag|mark|flag)\s+(each|them|as)\b|"
    r"\b(classify|classification)\s+(each|them)\b|"
    # grouping / bucketing
    r"\b(bucket|tier|band)\s+.{0,30}\binto\b|"
    r"\b(categori[sz]e|categorization)\b|"
    # translation / conversion to a label
    r"\btranslates? to\b|"
    r"\bconverts? to\b|"
    # and-their-X patterns ("and their classification", "and their label", …)
    r"\band their (classification|label|tier|category|group)\b|"
    # into-a-bucket patterns
    r"\binto\s+(quadrant|tier|band|category|class)\b|"
    # comparison operators
    r"\bversus\b|"
    r"\bvs\.?\b|"
    r"\b(compare|comparison)\b|"
    # grouped-by / for-each-group / breakdown
    r"\bgrouped by\b|"
    r"\bfor each group\b|"
    r"\bbreakdown\b"
    r")",
    re.IGNORECASE,
)

# ── Public API ────────────────────────────────────────────────────────────────

_PatternKey = Literal["fallback", "status", "classification"]


def detect_conditional_output_pattern(question: str) -> Optional[_PatternKey]:
    """Return which pattern fired, or None if no trigger matched.

    Priority: FALLBACK → STATUS → CLASSIFICATION (first match wins).
    """
    if _FALLBACK_SIGNAL.search(question):
        logger.debug("ConditionalOutput — Pattern FALLBACK matched")
        return "fallback"
    if _STATUS_CHECK_SIGNAL.search(question):
        logger.debug("ConditionalOutput — Pattern STATUS matched")
        return "status"
    if _CLASSIFICATION_SIGNAL.search(question):
        logger.debug("ConditionalOutput — Pattern CLASSIFICATION matched")
        return "classification"
    return None


_HINT_BY_PATTERN: dict[_PatternKey, str] = {
    "fallback": _HINT_FALLBACK,
    "status": _HINT_STATUS,
    "classification": _HINT_CLASSIFICATION,
}


def get_conditional_output_hint(question: str) -> Optional[str]:
    """Return the targeted hint string for the matched pattern, or None.

    Returns None when no pattern matches or the feature is disabled.
    Call conditional_output_enabled() before this if you want to gate on the flag.
    """
    pattern = detect_conditional_output_pattern(question)
    if pattern is None:
        return None
    return _HINT_BY_PATTERN[pattern]
