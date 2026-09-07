"""
Live test to compare reasoning vs non-reasoning model for formula completeness detection.

Run with:
    uv run pytest gsf/tests/interactive/test_completeness.py -s -v
"""

import pytest

from gsf.retrieval.interactive.completeness import detect_incomplete_formulas
from gsf.utils.llm_invoke import get_llm_client, get_non_reasoning_llm_client

CASES = [
    {
        "id": "archeology_1_vague_sqs",
        "description": "Vague SQS answer — base ratio and noise dampening formula never specified",
        "working_question": (
            "I'd like to see a quality assessment of scans across our archaeological sites. "
            "Show site code, site name, Scan Quality Score (SQS) for each site "
            "(where SQS is calculated based on scan resolution, point density, coverage percentage, "
            "and noise levels) and rank them in descending order (highest quality first)."
        ),
        "last_turn": {
            "q": "What is the exact formula for calculating the Scan Quality Score (SQS) from scan resolution, point density, coverage percentage, and noise levels?",
            "a": (
                "The Scan Quality Score (SQS) is calculated using a composite formula that combines "
                "multiple scan parameters. It takes the scan resolution (in millimeters) and point "
                "density to establish a base quality ratio, which is then raised to a power of 1.5 "
                "for emphasis. This is multiplied by the coverage percentage (normalized to a 0-1 scale) "
                "and a noise factor derived from the noise level in decibels. The noise component uses "
                "a squared dampening function to penalize higher noise values. The final average SQS "
                "for each site is rounded to 2 decimal places."
            ),
        },
        "relevant_kb": "None",
        "expected_incomplete": [
            "SQS",
            "noise",
        ],  # at least one of these should be flagged
    },
    {
        "id": "alien_8_ntm_modulation_from_kb",
        "description": "NTM KB entry has 'non-natural modulation' — should be flagged from KB even without user answer",
        "working_question": (
            "Scan our database for signals matching narrowband profiles. "
            "Show signal identifiers, central frequency, drift rate, BFR, and NTM classification."
        ),
        "last_turn": {
            "q": "Should the ranking be in descending order?",
            "a": "Yes, highest quality first.",
        },
        "relevant_kb": (
            "- NTM Classification System\n"
            "  Description: A tiered classification system for Narrowband Technological Markers.\n"
            "  Definition: Three-tier classification: 'Strong NTM' (BFR < 0.0001 AND FreqDriftHzs < 0.1 "
            "AND non-natural modulation), 'Moderate NTM' (BFR < 0.0005 AND FreqDriftHzs < 0.5 "
            "AND non-natural modulation), and 'Not NTM' (all other signals)."
        ),
        "expected_incomplete": ["modulation", "NTM"],
    },
    {
        "id": "prior_gap_resolved_by_exact_answer",
        "description": "BFR was a prior gap; user now gives exact formula — should be dropped from list",
        "working_question": (
            "Scan our database for signals matching narrowband profiles. "
            "Show signal identifiers, central frequency, drift rate, bandwidth ratio (BFR), "
            "and NTM classification."
        ),
        "prior_gaps": [
            ("BFR", "exact formula combining bandwidth and center frequency not given"),
            ("NTM classification", "classification tiers not defined"),
        ],
        "last_turn": {
            "q": "What is the exact formula for the bandwidth ratio (BFR)?",
            "a": "BFR = BandwidthHz / (FreqMhz * 1,000,000).",
        },
        "relevant_kb": "None",
        "expected_incomplete": ["NTM"],  # BFR resolved; NTM still open
        "should_not_flag": ["BFR", "bandwidth ratio"],
    },
    {
        "id": "alien_8_explicit_bfr",
        "description": "Explicit BFR formula given — BFR itself should NOT be flagged (other gaps OK)",
        "working_question": (
            "Scan our database for signals matching narrowband profiles. "
            "Show signal identifiers, central frequency, drift rate, bandwidth ratio (BFR), "
            "and NTM classification."
        ),
        "last_turn": {
            "q": "What is the exact formula for the bandwidth ratio (BFR)?",
            "a": "BFR = BandwidthHz / (FreqMhz * 1,000,000). This is the ratio of signal bandwidth to center frequency.",
        },
        "relevant_kb": "None",
        "expected_incomplete": [],  # not used — see should_not_flag below
        "should_not_flag": [
            "BFR",
            "bandwidth ratio",
        ],  # the answered question should be resolved
    },
]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_completeness_comparison(case):
    reasoning = get_llm_client()
    non_reasoning = get_non_reasoning_llm_client()
    prior = case.get("prior_gaps", [])

    r_gaps = detect_incomplete_formulas(
        case["working_question"],
        case["last_turn"],
        case["relevant_kb"],
        prior,
        reasoning,
    )
    nr_gaps = detect_incomplete_formulas(
        case["working_question"],
        case["last_turn"],
        case["relevant_kb"],
        prior,
        non_reasoning,
    )

    print(f"\n{'=' * 60}")
    print(f"Case: {case['id']}")
    print(f"Description: {case['description']}")
    print("\n--- Reasoning gaps ---")
    for term, missing in r_gaps:
        print(f"  INCOMPLETE: {term} | {missing}")
    if not r_gaps:
        print("  COMPLETE")
    print("\n--- Non-reasoning gaps ---")
    for term, missing in nr_gaps:
        print(f"  INCOMPLETE: {term} | {missing}")
    if not nr_gaps:
        print("  COMPLETE")

    expected = case["expected_incomplete"]
    if expected:
        # Only assert on reasoning model — non-reasoning is documented as insufficient for this task
        flagged_text = " ".join(t + " " + m for t, m in r_gaps).lower()
        assert any(kw.lower() in flagged_text for kw in expected), (
            f"[reasoning] expected one of {expected} to be flagged but got: {r_gaps}"
        )
        nr_flagged = " ".join(t + " " + m for t, m in nr_gaps).lower()
        nr_ok = any(kw.lower() in nr_flagged for kw in expected)
        print(
            f"  [non-reasoning] catches expected gaps: {nr_ok} (known limitation if False)"
        )

    should_not_flag = case.get("should_not_flag", [])
    for result, label in [(r_gaps, "reasoning"), (nr_gaps, "non-reasoning")]:
        flagged_terms = " ".join(t for t, _ in result).lower()
        for kw in should_not_flag:
            assert kw.lower() not in flagged_terms, (
                f"[{label}] {kw!r} should be resolved but was flagged as incomplete: {result}"
            )
