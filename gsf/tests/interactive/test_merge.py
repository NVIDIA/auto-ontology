"""
Live test to compare reasoning vs non-reasoning model for clarification merging.

Run with:
    uv run pytest gsf/tests/interactive/test_merge.py -s -v

Each case checks whether the merge LLM invents a formula from a vague answer
(bad) vs preserving the user's description in their own words (good).
"""

import pytest

from gsf.retrieval.interactive.merge import merge_clarification
from gsf.utils.llm_invoke import get_llm_client, get_non_reasoning_llm_client

CASES = [
    {
        "id": "archeology_1_vague_formula",
        "description": (
            "Vague SQS description — exact base ratio and noise factor never given. "
            "Merged question should preserve ambiguity so clarify can re-ask. "
            "EXPECTED FAILURE: neither model currently signals the formula is incomplete."
        ),
        "current_question": (
            "I'd like to see a quality assessment of scans across our archaeological sites. "
            "Show site code, site name, Scan Quality Score (SQS) for each site "
            "(where SQS is calculated based on scan resolution, point density, coverage percentage, "
            "and noise levels) and rank them in descending order (highest quality first)."
        ),
        "new_q": (
            "What is the exact formula for calculating the Scan Quality Score (SQS) "
            "from scan resolution, point density, coverage percentage, and noise levels?"
        ),
        "new_a": (
            "The Scan Quality Score (SQS) is calculated using a composite formula that combines "
            "multiple scan parameters. It takes the scan resolution (in millimeters) and point "
            "density to establish a base quality ratio, which is then raised to a power of 1.5 "
            "for emphasis. This is multiplied by the coverage percentage (normalized to a 0-1 scale) "
            "and a noise factor derived from the noise level in decibels. The noise component uses "
            "a squared dampening function to penalize higher noise values. The final average SQS "
            "for each site is rounded to 2 decimal places for reporting purposes."
        ),
        "relevant_kg": "None",
        "should_not_contain": [
            "pointdense / scanresolmm",
            "PointDense / ScanResolMm",
            "1 / NoiseDb",
            "1/NoiseDb",
        ],
        # Neither model currently signals incompleteness — both just embed the vague description.
        # A formula-completeness detector is needed to flag these for re-asking.
        "preserves_ambiguity": False,  # set True if model explicitly marks formula as unresolved
    },
    {
        "id": "alien_8_explicit_bfr",
        "description": "Explicit BFR formula — SHOULD be embedded verbatim",
        "current_question": (
            "Could you scan our database for potential signals matching narrowband profiles? "
            "I need the signal identifiers, central frequency, drift rate, bandwidth ratio "
            "and the classification of NTM categories based on signal stability."
        ),
        "new_q": "What is the exact formula for computing the bandwidth ratio (BFR) from the database columns?",
        "new_a": (
            "The bandwidth ratio (BFR) is calculated by dividing the bandwidth in Hz by the "
            "center frequency in MHz converted to Hz. Specifically, BFR = BandwidthHz / (FreqMhz * 1,000,000)."
        ),
        "relevant_kg": "None",
        "should_contain": ["BandwidthHz", "FreqMhz", "1,000,000", "1000000"],
    },
]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_merge_comparison(case):
    reasoning = get_llm_client()
    non_reasoning = get_non_reasoning_llm_client()

    turn = {"q": case["new_q"], "a": case["new_a"]}

    r_result = merge_clarification(
        case["current_question"], turn, reasoning, case["relevant_kg"]
    )
    nr_result = merge_clarification(
        case["current_question"], turn, non_reasoning, case["relevant_kg"]
    )

    print(f"\n{'=' * 60}")
    print(f"Case: {case['id']}")
    print(f"Description: {case['description']}")
    print(f"\n--- Reasoning ---\n{r_result}")
    print(f"\n--- Non-reasoning ---\n{nr_result}")

    _AMBIGUITY_SIGNALS = [
        "exact formula",
        "not specified",
        "unresolved",
        "unclear",
        "not provided",
        "needs clarification",
    ]

    for result, label in [(r_result, "reasoning"), (nr_result, "non-reasoning")]:
        if "should_not_contain" in case:
            for fragment in case["should_not_contain"]:
                assert fragment not in result, (
                    f"[{label}] invented formula fragment {fragment!r} found in merged question"
                )
        if "should_contain" in case:
            assert any(f in result for f in case["should_contain"]), (
                f"[{label}] expected formula not found in merged question. "
                f"Expected one of: {case['should_contain']}"
            )
        if "preserves_ambiguity" in case:
            signals = [s for s in _AMBIGUITY_SIGNALS if s.lower() in result.lower()]
            preserved = bool(signals)
            print(
                f"  [{label}] ambiguity preserved: {preserved} (signals found: {signals or 'none'})"
            )
            if case["preserves_ambiguity"]:
                assert preserved, (
                    f"[{label}] expected merged question to signal unresolved formula"
                )
