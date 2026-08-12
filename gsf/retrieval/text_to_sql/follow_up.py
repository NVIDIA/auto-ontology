# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve contextual chat follow-ups before the text-to-SQL graph runs."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from gsf.retrieval.text_to_sql.models import FollowUpResolutionModel
from gsf.retrieval.text_to_sql.prompts import create_follow_up_resolution_prompt
from gsf.utils.llm_invoke import invoke_with_structured_output

logger = logging.getLogger(__name__)


def _format_history(history: Sequence[Mapping[str, str | None]]) -> str:
    blocks: list[str] = []
    for index, turn in enumerate(history, start=1):
        sql = (turn.get("sql_code") or "").strip()
        block = (
            f"Turn {index}\n"
            f"User: {(turn.get('question') or '').strip()}\n"
            f"Assistant: {(turn.get('response') or '').strip()}"
        )
        if sql:
            block += f"\nFinal SQL: {sql}"
        blocks.append(block)
    return "\n\n".join(blocks)


def resolve_follow_up(
    *,
    question: str,
    history: Sequence[Mapping[str, str | None]],
    llm: BaseChatModel | None,
) -> tuple[bool, str]:
    """Return ``(is_follow_up, processing_question)`` with safe fallback."""

    if not history or llm is None:
        return False, question

    prompt = create_follow_up_resolution_prompt(
        question=question,
        conversation_history=_format_history(history),
    )
    try:
        result = invoke_with_structured_output(
            llm, [HumanMessage(content=prompt)], FollowUpResolutionModel
        )
    except Exception:  # noqa: BLE001 — context is optional; chat must still run
        logger.exception("Follow-up resolution failed; using original question")
        return False, question

    if result is None:
        logger.warning("Follow-up resolution returned no result")
        return False, question

    standalone = result.standalone_question.strip()
    if not result.is_follow_up:
        # Enforce the contract locally instead of allowing an independent
        # question to be silently changed by the model.
        return False, question
    return True, standalone or question


__all__ = ["resolve_follow_up"]
