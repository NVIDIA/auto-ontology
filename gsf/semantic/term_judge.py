"""Lightweight Judge LLM for Term deduplication during compilation."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from gsf.utils.llm_invoke import get_llm_client, invoke_with_structured_output

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM = """\
You decide whether a proposed business Term is the same real-world concept \
as an existing Term already in the semantic layer.

Merge when they clearly refer to the same entity — even if names differ \
(e.g. "Client" and "Customer", "Product" and "Item", "Employee" and "Staff").

Keep separate when they are related but distinct concepts \
(e.g. "Order" and "Order Line", "Customer" and "Customer Category", \
"Invoice" and "Invoice Line").

Archive or history variants of the same entity SHOULD be merged \
(e.g. "Customer Archive" and "Customer" are the same concept)."""


class TermJudgeVerdict(BaseModel):
    """LLM output: merge decision for a proposed Term."""

    model_config = ConfigDict(extra="forbid")

    merge_into: str | None = Field(
        ...,
        description=(
            "Name of the existing Term to merge into, "
            "or null if the proposed Term should be kept separate."
        ),
    )
    reason: str = Field(
        default="",
        description="Brief explanation of the merge/separate decision.",
    )


def judge_term_overlap(
    proposed_name: str,
    proposed_description: str,
    candidates: list[dict],
) -> str | None:
    """Ask the Judge LLM whether a proposed Term duplicates an existing one.

    ``candidates`` is a list of ``{"name": str, "description": str, "score": float}``
    from VDB similarity search.

    Returns the existing Term name to merge into, or None to keep separate.
    """
    if not candidates:
        return None

    candidate_lines = "\n".join(
        f"  - {c['name']}: {c.get('content', '')} (similarity: {c['score']:.2f})"
        for c in candidates
    )

    prompt = (
        f"Proposed Term: {proposed_name}\n"
        f"Description: {proposed_description}\n\n"
        f"Existing Terms (from most to least similar):\n{candidate_lines}\n\n"
        f"Should the proposed Term be merged into one of the existing Terms, "
        f"or kept as a separate concept?"
    )

    verdict = invoke_with_structured_output(
        get_llm_client(temperature=0.0, max_tokens=512),
        [SystemMessage(content=_JUDGE_SYSTEM), HumanMessage(content=prompt)],
        TermJudgeVerdict,
    )

    if verdict is None:
        logger.warning("Judge LLM returned no verdict for Term %r", proposed_name)
        return None

    if verdict.merge_into:
        valid_names = {c["name"] for c in candidates}
        if verdict.merge_into not in valid_names:
            logger.warning(
                "Judge suggested merging %r into %r which is not a candidate — ignoring",
                proposed_name,
                verdict.merge_into,
            )
            return None
        logger.info(
            "Judge: merge %r into %r — %s",
            proposed_name,
            verdict.merge_into,
            verdict.reason,
        )
        return verdict.merge_into

    logger.debug("Judge: keep %r separate — %s", proposed_name, verdict.reason)
    return None
