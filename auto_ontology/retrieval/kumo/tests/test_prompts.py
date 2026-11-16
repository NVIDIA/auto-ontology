# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from auto_ontology.retrieval.kumo.prompts import build_pql_prompt


def test_the_organization_context_reaches_the_pql_prompt() -> None:
    prompt = build_pql_prompt(
        graph_ddl="customers(id)",
        question="Will MRR drop?",
        organization="GLOSSARY\n- MRR: monthly recurring revenue",
    )

    section = "## Organization context\nGLOSSARY\n- MRR: monthly recurring revenue"
    assert section in prompt
    assert prompt.index(section) < prompt.index("## Question")


def test_no_organization_context_adds_no_section() -> None:
    prompt = build_pql_prompt(graph_ddl="customers(id)", question="q", organization=" ")

    assert "Organization context" not in prompt
