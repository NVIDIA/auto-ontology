# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from auto_ontology.retrieval.text_to_sql.prompts import format_authoritative_evidence


def test_empty_evidence_returns_no_prompt_block() -> None:
    assert format_authoritative_evidence("") == ""


def test_evidence_is_preserved_verbatim_without_derived_rules() -> None:
    evidence = (
        "The requested time refers to events.duration LIKE 'M:SS%'.\n"
        "Use the literal 0:01:54 from the question."
    )

    block = format_authoritative_evidence(evidence)

    assert block.endswith(evidence)
    assert "events.duration LIKE '1:54%'" not in block
    assert "Explicit mappings in this evidence are binding" in block
    assert "Do not broaden an explicit predicate" in block


def test_formula_instructions_require_exact_calculation_without_shortcut() -> None:
    evidence = "Lifetime loss = annual_degradation * capacity * revenue_factor * 15."

    block = format_authoritative_evidence(evidence)

    assert block.endswith(evidence)
    assert "Preserve evidence-specified operators" in block
    assert "formula terms, and scaling factors" in block
    assert "Confirm formulas retain their operators" in block


def test_generation_can_require_evidence_first_schema_mapping() -> None:
    evidence = "Eligible rate = eligible_count / enrollment."

    block = format_authoritative_evidence(evidence, evidence_first_sql=True)

    assert "Start SQL construction from the explicit evidence requirements" in block
    assert "Build their projections, formulas, predicates" in block
    assert "only then add requirements from the question" in block
    assert "Verify every named identifier against the Available Schema" in block
    assert block.endswith(evidence)
