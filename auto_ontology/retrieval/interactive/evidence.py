# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import re

from auto_ontology.utils.llm_invoke import safe_invoke_text_nr

from .state import InteractiveSessionState

logger = logging.getLogger(__name__)

_EVIDENCE_PROMPT = """\
Working question: {question}

Relevant external knowledge (one entry per term):
{grounded_kb}
{resolved_terms_section}
Extract any formulas, calculation rules, threshold/filter conditions or definitions from the relevant \
external knowledge that are directly needed to answer the working question above. For each such entry, output one \
line in SQL-friendly notation:
  TermName = <formula, threshold, filter condition or definition using column names and values, if present>
Only include conditions expressible with specific column names and values — skip \
natural-language qualifiers with no clear SQL translation. \
Never invent a column-like name (Title_Case/snake_case) for a term with no confirmed \
mapping. If no term in a formula has a confirmed mapping, omit the line entirely — \
EXCEPT when that entry defines a sub-metric referenced as an operand by another \
formula required by the working question (e.g. a sub-index inside a composite score), \
in which case keep the sub-formula line and use [UNRESOLVED: <term>] for its unmapped \
operands. If only some terms are confirmed, keep the formula structure and substitute \
[UNRESOLVED: <term>] — using the term's exact original wording from the question — \
for each unconfirmed operand, never a name that could pass as a real column. \
When a confirmed mapping's description or "Sample values:" lists JSON sub-keys \
(e.g. `initial_margin_pct`, `Low_52W`, `High_52W`, `Tx_Adh`, `Func_Impv`, \
`disposal.disposal_cost`), use those exact JSON sub-keys for operands that match \
them. Do NOT use a VDB-matched flat column mapping when it clearly conflicts with \
the term's required data type, unit, table scope, or role in the formula — \
specifically: (a) a boolean flag or a duration/hours column where the formula sums \
or subtracts monetary costs/fees, (b) a measurement-uncertainty/error column \
(e.g. "Upper Uncertainty") where the formula requires the actual physical quantity \
(e.g. mass), (c) a flat column whose meaning differs from a sibling sub-field when \
the other operands of the same formula all come from sub-keys of a single JSONB \
column (e.g. cost_breakdown or risk_profile), (d) a flat column whose only sample \
value is `0.0` when a JSONB column on the same table has a sub-key for that metric \
(e.g. `disposal.disposal_cost` in `Cost Breakdown`), or (e) a narrower attribute on \
a different table (such as `Medication Adherence` on `Treatment Basics`) when the \
formula explicitly defines a rate over `treatment outcomes` and `Treatment Outcome` \
has a JSONB column (`Treatment Progress Metrics`) with a matching sub-key (`Tx_Adh`). \
In any such conflict, use the matching JSONB sub-key if shown in the mappings, or \
write [UNRESOLVED: <term>] instead of forcing the mismatched flat column. \
Skip any entry not required by the working question. \
Entries may include a "# matched from: <terms>" annotation line listing the original \
natural-language phrases from the question that correspond to this KB entry — use these \
to connect KB entries to the working question even when the phrasing differs. \
If an entry is marked [DISAMBIGUATION], it means a KB formula and a direct schema column \
both matched the same term — include only whichever is correct given the question context. \
Preserve exact scaling factors from the external knowledge (such as `* 100` for percentages, \
proportions, or shares) even when the working question refers to the metric as a "proportion" \
or "share". \
If nothing applies, output: NONE"""


def build_grounded_terms_hint(session: InteractiveSessionState) -> str:
    """Surface already-resolved schema mappings for terms VDB-matched to the
    current question, so the evidence LLM grounds formula terms in real
    columns instead of guessing table/column names — and can correctly skip a
    term instead of inventing one when nothing is confirmed.

    Collisions (two terms landing on the same column) are resolved upstream,
    during clarify's _resolve_collisions — by the time _cached_resolved_hits
    reaches here every term is already mapped to a distinct column, or
    confirmed as legitimately sharing one (in which case a note is injected
    directly into evidence separately — see coordinator._run_sql_generation). So every
    hit here can be presented as a plain confirmed mapping.
    """
    hits = session._cached_resolved_hits or []
    if not hits:
        return ""

    # Keep the best (lowest-distance) hit per term.
    best: dict[str, tuple[str, float, str]] = {}
    for norm, hit_text, score, hit_id in hits:
        if norm not in best or score < best[norm][1]:
            best[norm] = (hit_text, score, hit_id)
    if not best:
        return ""

    ambiguous_map = getattr(session, "_last_ambiguous_hits", None) or {}
    lines: list[str] = []
    for norm, (hit_text, _score, hit_id) in best.items():
        lines.append(f'  "{norm}" → {hit_text}')
        alt_hits = ambiguous_map.get(norm) or []
        shown_alts = 0
        for alt in alt_hits:
            alt_id = str(alt.get("id") or "")
            alt_score = float(alt.get("score") or 1.0)
            alt_text = str(alt.get("text") or "").strip()
            if not alt_text or alt_id == hit_id or alt_score > 0.68:
                continue
            lines.append(f'    (close alternative for "{norm}": {alt_text})')
            shown_alts += 1
            if shown_alts >= 2:
                break
    return (
        "\nConfirmed schema mappings for terms VDB-matched to the question "
        "(terms not listed here have no confirmed mapping):\n" + "\n".join(lines) + "\n"
    )


def generate_evidence(
    question: str, grounded_kb: str, resolved_terms_section: str = ""
) -> str:
    """Convert grounded KB text into a short Evidence string for the SQL generator."""
    if not grounded_kb:
        return ""
    prompt = _EVIDENCE_PROMPT.format(
        question=question,
        grounded_kb=grounded_kb,
        resolved_terms_section=resolved_terms_section,
    )
    response = safe_invoke_text_nr(prompt).strip()
    logger.debug("SQL gen — Evidence raw response: %s", response)
    if not response or response.upper() == "NONE":
        return ""
    # Valid output is one short line per fact — a formula, a condition, or a
    # plain definition (not every evidence line is a "Term = expression"
    # formula; some are just a resolved definition with no operator at all).
    # Reject only on length: a genuine condensed evidence line is always
    # short, so a line approaching full KB-definition-paragraph length is
    # almost certainly restated prose rather than compressed evidence.
    _MAX_EVIDENCE_LINE_CHARS = 600
    _CONTINUATION = re.compile(r"^\s+(AND|OR)\b", re.IGNORECASE)
    # A scope-less "Term = COUNT(x)" line is valid, KB-derived evidence and must
    # pass — do not add a bare-aggregate exclusion here.
    # Join AND/OR continuation lines onto the preceding valid formula line before filtering,
    # so multi-condition expressions like "A = x AND y IN (...)" survive even if the LLM
    # wraps the second clause onto a new line.
    joined_lines: list[str] = []
    for line in response.splitlines():
        if _CONTINUATION.match(line) and joined_lines:
            joined_lines[-1] = joined_lines[-1].rstrip() + " " + line.strip()
        else:
            joined_lines.append(line)
    valid_lines = [
        ln
        for ln in joined_lines
        if ln.strip()
        and not ln.lstrip().startswith("#")
        and len(ln) <= _MAX_EVIDENCE_LINE_CHARS
    ]
    if not valid_lines:
        logger.warning(
            "SQL gen — Evidence generation returned prose, discarding: %s",
            response[:100],
        )
        return ""
    return "\n".join(valid_lines)
