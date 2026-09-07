"""
Live test to compare reasoning vs non-reasoning model for KB coverage classification.

Checks whether each model correctly identifies which KB entries cover a given entity,
and whether either model produces the known false positive (Equipment Problems for
"problematic events" in the alien_3 query — the ground-truth count is LIF > 0.5 only,
Equipment Problems is not relevant).

Run with:
    uv run pytest gsf/tests/interactive/test_kb_coverage.py -s -v
"""

import time

from gsf.retrieval.interactive.kb_coverage import (
    _slim_kb_for_coverage,
    _KB_COVERAGE_PROMPT,
)
from gsf.utils.llm_invoke import (
    get_llm_client,
    get_non_reasoning_llm_client,
    safe_invoke_text,
)

# ── KB fixture: alien_3 external knowledge (relevant entries only) ────────────

_ALIEN_KB = """\
- High Lunar Interference Events
  Description: Observations with significant lunar interference.
  Definition: Events where the calculated LIF is greater than 0.5, indicating strong lunar contamination in the data.

- Equipment Problems
  Description: Defines what counts as an abnormal condition for a telescope's subsystems.
  Definition: A telescope is considered to have an equipment problem whenever **any** of its key subsystem states are not in their nominal condition: • Equipment status is not "Operational" • Calibration status is not "Current" • Cooling-system status is not "Normal"

- Signal Degradation Scenario (SDS)
  Description: Describes conditions under which signal quality degrades significantly.
  Definition: An SDS occurs when the Signal-to-Noise Quality Indicator (SNQI) falls below 0.3 AND the Bandwidth-Frequency Ratio (BFR) exceeds 2.0.
"""

# Entities to classify — taken from the alien_3 clarification pipeline
_QUESTION = (
    "Analyze how lunar interference affects observations by showing the current moon phase, "
    "average interference level and the count of problematic events for each observatory, "
    "sorted by average interference."
)

_ENTITIES = [
    "problematic events",
    "lunar interference",
    "observatory",
    "current moon phase",
]

# Known false positive to watch: "problematic events" should NOT be covered by
# Equipment Problems for this query (ground truth uses LIF > 0.5 only).
FALSE_POSITIVE_TERM = "problematic events"
FALSE_POSITIVE_ENTRY = "equipment problems"


def _run_coverage(
    llm, entities: list[str], formatted_kb: str, question: str = ""
) -> tuple[dict[str, list[str]], float]:
    """Run coverage LLM and return (term → matched_entries, elapsed_seconds)."""
    entity_list = "\n".join(f"- {e}" for e in entities)
    slim_kb = _slim_kb_for_coverage(formatted_kb)
    prompt = _KB_COVERAGE_PROMPT.format(
        formatted_kb=slim_kb,
        entity_list=entity_list,
        question=question or "(not provided)",
    )

    t0 = time.monotonic()
    response = safe_invoke_text(llm, prompt).strip()
    elapsed = time.monotonic() - t0

    # Parse YES lines the same way _filter_covered_by_external_knowledge does
    import re

    term_to_entries: dict[str, list[str]] = {}
    for line in response.splitlines():
        if ": YES" not in line.upper():
            continue
        stripped = re.sub(r"^coverage\s*:\s*", "", line, count=1, flags=re.IGNORECASE)
        term = stripped.split(":")[0].strip().lstrip("- ").lower()
        entries = []
        if "|" in line:
            after_yes = line.split("|", 1)[1]
            for name in after_yes.split("|"):
                n = name.strip().lower()
                if n:
                    entries.append(n)
        term_to_entries[term] = entries

    return term_to_entries, elapsed


def test_coverage_comparison():
    reasoning_llm = get_llm_client()
    non_reasoning_llm = get_non_reasoning_llm_client()

    r_result, r_time = _run_coverage(reasoning_llm, _ENTITIES, _ALIEN_KB, _QUESTION)
    nr_result, nr_time = _run_coverage(
        non_reasoning_llm, _ENTITIES, _ALIEN_KB, _QUESTION
    )

    print(f"\n{'─' * 60}")
    print(f"{'Term':<30}  {'Reasoning':>20}  {'Non-reasoning':>20}")
    print(f"{'─' * 60}")
    for term in _ENTITIES:
        r_entries = r_result.get(term, [])
        nr_entries = nr_result.get(term, [])
        r_str = ", ".join(r_entries) if r_entries else "NO"
        nr_str = ", ".join(nr_entries) if nr_entries else "NO"
        fp_flag = ""
        if (
            FALSE_POSITIVE_ENTRY in r_entries or FALSE_POSITIVE_ENTRY in nr_entries
        ) and term == FALSE_POSITIVE_TERM:
            fp_flag = "  ← FALSE POSITIVE"
        print(f"  {term:<28}  {r_str:>20}  {nr_str:>20}{fp_flag}")
    print(f"{'─' * 60}")
    print(f"  Runtime — reasoning: {r_time:.1f}s   non-reasoning: {nr_time:.1f}s")

    r_fp = FALSE_POSITIVE_ENTRY in r_result.get(FALSE_POSITIVE_TERM, [])
    nr_fp = FALSE_POSITIVE_ENTRY in nr_result.get(FALSE_POSITIVE_TERM, [])
    print(f"\n  False positive ({FALSE_POSITIVE_TERM!r} → {FALSE_POSITIVE_ENTRY!r}):")
    print(f"    Reasoning:     {'YES (false positive)' if r_fp else 'no'}")
    print(f"    Non-reasoning: {'YES (false positive)' if nr_fp else 'no'}")

    assert isinstance(r_result, dict)
    assert isinstance(nr_result, dict)
