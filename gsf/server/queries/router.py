# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query helpers — translate SQL into plain English via an NVIDIA NIM LLM.

Backs the "Translate with AI" toggle in the catalog Query tab. The frontend
posts one SQL string at a time and renders the returned sentence; failures are
surfaced as HTTP errors so the UI can fall back gracefully.
"""

from __future__ import annotations

import logging
import os

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

_CHAT_ENDPOINT = os.environ.get(
    "CHAT_ENDPOINT",
    os.environ.get("EMBED_ENDPOINT", "https://integrate.api.nvidia.com/v1"),
)
_CHAT_MODEL = os.environ.get("CHAT_MODEL", "meta/llama-3.1-8b-instruct")
_REQUEST_TIMEOUT = 30.0

_SYSTEM_PROMPT = (
    "You are a data analyst. Describe what a SQL query returns in one or two "
    "short, plain-English sentences a non-technical stakeholder can understand. "
    "Start with a verb such as 'Returns', 'Lists', 'Counts', or 'Updates'. "
    "Name the tables and the key filters or groupings. Do not explain what SQL "
    "is, do not restate the query, and do not use code formatting."
)


class ExplainRequest(BaseModel):
    """A single SQL statement to translate."""

    sql: str = Field(..., min_length=1, max_length=20_000)


class ExplainResponse(BaseModel):
    explanation: str


def _api_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        raise HTTPException(
            status_code=503,
            detail="AI translation is unavailable: NVIDIA_API_KEY is not set.",
        )
    return key


@router.post("/queries/explain", response_model=ExplainResponse)
async def explain_query(request: ExplainRequest) -> ExplainResponse:
    """Translate a single SQL statement into a plain-English summary."""
    payload = {
        "model": _CHAT_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": request.sql.strip()},
        ],
        "temperature": 0.2,
        "max_tokens": 160,
    }

    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            response = await client.post(
                f"{_CHAT_ENDPOINT}/chat/completions",
                headers={"Authorization": f"Bearer {_api_key()}"},
                json=payload,
            )
    except httpx.HTTPError as exc:
        logger.warning("explain_query: upstream request failed: %s", exc)
        raise HTTPException(
            status_code=502, detail="AI translation service is unreachable."
        ) from exc

    if response.status_code == 429:
        raise HTTPException(
            status_code=429,
            detail="AI translation is rate limited; try again shortly.",
        )
    if response.status_code >= 400:
        logger.warning(
            "explain_query: upstream %s: %s",
            response.status_code,
            response.text[:300],
        )
        raise HTTPException(status_code=502, detail="AI translation failed.")

    try:
        explanation = response.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, ValueError) as exc:
        logger.warning("explain_query: unexpected response shape: %s", exc)
        raise HTTPException(status_code=502, detail="AI translation failed.") from exc

    if not explanation:
        raise HTTPException(status_code=502, detail="AI translation was empty.")

    return ExplainResponse(explanation=explanation)
