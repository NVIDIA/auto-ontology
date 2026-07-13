# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""On-demand LLM description suggestion for a single SqlAttribute."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from gsf.dal.sql_attributes import (
    get_full_sql_attribute_by_id,
    set_sql_attribute_description_suggestion,
)
from gsf.utils.llm_invoke import get_llm_client, invoke_with_structured_output

logger = logging.getLogger(__name__)

_DESCRIPTION_SUGGESTION_MAX_CHARS = 500

_SYSTEM_PROMPT = f"""\
You're a helpful AI assistant
with a specialization in databases and BI
with a business qualification.
In a specific business model, attributes are collected within a Business Term.
A SQL attribute is an attribute that can be defined by a SQL expression.
You will be provided with a specific attribute and the Business Term that \
contains the attribute.
Your job is to describe in up to 3 sentences and no more than \
{_DESCRIPTION_SUGGESTION_MAX_CHARS} characters the role of the given \
attribute in the business to the business expert.
Do not include the attribute name or raw SQL in your answer.
Provide clear, assertive, and confident responses.
Be certain in your responses; do not use words like 'likely' or 'maybe'.
Do not reference the industry in your answers.
Do not apologize no matter what."""


class _DescriptionSuggestionResponse(BaseModel):
    """LLM output: a suggested description for a SqlAttribute."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(
        max_length=_DESCRIPTION_SUGGESTION_MAX_CHARS,
        description="The suggested description for the SqlAttribute.",
    )


def _fit_description_suggestion(description: str) -> str:
    """Keep LLM output within the description-size budget before caching it.

    Also guards against a response that was itself cut off mid-sentence
    (e.g. by the LLM's ``max_tokens`` limit) by trimming back to the last
    complete sentence, even when the raw text is already within budget.
    """
    suggestion = " ".join(description.split())
    if len(suggestion) > _DESCRIPTION_SUGGESTION_MAX_CHARS:
        suggestion = suggestion[:_DESCRIPTION_SUGGESTION_MAX_CHARS].rstrip()

    if suggestion and suggestion[-1] in ".!?":
        return suggestion

    sentence_end = max(
        suggestion.rfind("."), suggestion.rfind("!"), suggestion.rfind("?")
    )
    if sentence_end > 0:
        return suggestion[: sentence_end + 1]
    return ""


def suggest_sql_attribute_description(attr_id: str) -> str | None:
    """Return a cached or freshly-generated description suggestion for a SqlAttribute.

    Reads the attribute's name, parent Term, and SQL from Neo4j via
    :func:`get_full_sql_attribute_by_id`. When a ``description_suggestion``
    is already stored on the node, it is returned as-is (no LLM call).
    Otherwise the LLM drafts one, it is persisted on the node via
    :func:`set_sql_attribute_description_suggestion`, and returned.

    Returns ``None`` when the attribute doesn't exist or the LLM call
    fails/returns nothing usable — callers should treat that as "no
    suggestion available" rather than an error.
    """
    row = get_full_sql_attribute_by_id(attr_id)
    if row is None:
        return None

    cached = row.get("description_suggestion")
    if cached:
        return cached

    prompt = (
        f"The attribute: '{row.get('name', '')}', "
        f"the term that contains the attribute: '{row.get('term_name', '')}'."
    )

    llm = get_llm_client(temperature=0.0, max_tokens=400)
    response = invoke_with_structured_output(
        llm,
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)],
        _DescriptionSuggestionResponse,
    )
    if response is None:
        return None

    suggestion = _fit_description_suggestion(response.description)
    if not suggestion:
        return None

    set_sql_attribute_description_suggestion(attr_id, suggestion)
    return suggestion
