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
    assert "MUST follow every instruction" in block
    assert "stated column, table, filter, and operator" in block


def test_formula_instructions_require_exact_calculation_without_shortcut() -> None:
    evidence = "Lifetime loss = annual_degradation * capacity * revenue_factor * 15."

    block = format_authoritative_evidence(evidence)

    assert block.endswith(evidence)
    assert "calculate that exact formula" in block
    assert "Do not replace it with a shortcut or precomputed field" in block
    assert "Preserve the evidence's stated order" in block
    assert "never reorder them" in block


def test_generation_can_require_evidence_first_schema_mapping() -> None:
    evidence = "Eligible rate = eligible_count / enrollment."

    block = format_authoritative_evidence(evidence, evidence_first_sql=True)

    assert "Start SQL construction from the evidence" in block
    assert "Decompose it into individual rules" in block
    assert "whose name and description in the Available Schema best match" in block
    assert "only then add the remaining requirements from the question" in block
    assert block.endswith(evidence)
