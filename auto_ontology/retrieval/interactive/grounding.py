from __future__ import annotations

import logging

from auto_ontology.utils.llm_invoke import safe_invoke_text

logger = logging.getLogger(__name__)

_GROUNDING_PROMPT = """\
You are preparing context for a SQL generation step.

User question: {question}

Domain definitions:
{formatted_kb}

List only the definitions, formulas, or rules above that are directly needed \
to answer this question. Format as concise bullet points. \
If none apply, output: NONE"""


def ground_external_knowledge(question: str, formatted_kb: str, llm) -> str:
    """Return only the knowledge items relevant to *question*, or '' if none."""
    if not formatted_kb:
        return ""
    prompt = _GROUNDING_PROMPT.format(question=question, formatted_kb=formatted_kb)
    response = safe_invoke_text(llm, prompt).strip()
    logger.info("Grounding — response: %s", response[:200])
    if response.upper() == "NONE":
        return ""
    return response
