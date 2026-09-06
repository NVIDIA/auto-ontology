from __future__ import annotations

import logging
import re

from gsf.utils.llm_invoke import safe_invoke_text_nr

from .state import InteractiveSessionState

logger = logging.getLogger(__name__)

_EVIDENCE_PROMPT = """\
Working question: {question}

Relevant external knowledge (one entry per term):
{grounded_kg}
{resolved_terms_section}
Extract any formulas, calculation rules, threshold/filter conditions or definitions from the relevant \
external knowledge that are directly needed to answer the working question above. For each such entry, output one \
line in SQL-friendly notation:
  TermName = <formula, threshold, filter condition or definition using column names and values, if present>
Only include conditions expressible with specific column names and values — skip \
natural-language qualifiers with no clear SQL translation. \
Never invent a column-like name (Title_Case/snake_case) for a term with no confirmed \
mapping. If no term in a formula has a confirmed mapping, omit the line entirely. If \
only some terms are confirmed, keep the formula structure and substitute \
[UNRESOLVED: <term>] — using the term's exact original wording from the question — \
for each unconfirmed operand, never a name that could pass as a real column. \
Skip any entry not required by the working question. \
Entries may include a "# matched from: <terms>" annotation line listing the original \
natural-language phrases from the question that correspond to this KB entry — use these \
to connect KB entries to the working question even when the phrasing differs. \
If an entry is marked [DISAMBIGUATION], it means a KB formula and a direct schema column \
both matched the same term — include only whichever is correct given the question context. \
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

    lines = [
        f'  "{norm}" → {hit_text}' for norm, (hit_text, _score, _hit_id) in best.items()
    ]
    return (
        "\nConfirmed schema mappings for terms VDB-matched to the question "
        "(terms not listed here have no confirmed mapping):\n" + "\n".join(lines) + "\n"
    )


def generate_evidence(
    question: str, grounded_kg: str, resolved_terms_section: str = ""
) -> str:
    """Convert grounded KB text into a short Evidence string for the SQL generator."""
    if not grounded_kg:
        return ""
    prompt = _EVIDENCE_PROMPT.format(
        question=question,
        grounded_kg=grounded_kg,
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
