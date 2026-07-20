# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API routes for turning free-text into data objects or PQL."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from gsf.server.metadata import service as dal

router = APIRouter()


class TextRequest(BaseModel):
    """Payload for the text-to-data / text-to-pql routes."""

    question: str = Field(..., min_length=1)


@router.post("/text-to-data")
def text_to_data(body: TextRequest) -> dict:
    """Return the data objects gathered before PQL creation for a question:
    the relevant tables, their catalog join paths, and each table's columns.

    Returns 422 when the flow cannot produce a result for the question.
    """
    try:
        result = dal.text_to_data(body.question)
    except dal.PredictionFlowError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"data": result}


@router.post("/text-to-pql")
def text_to_pql(body: TextRequest) -> dict:
    """Return the PQL generated for a question (generation only, no prediction).

    Returns 422 when the flow cannot produce a result for the question.
    """
    try:
        result = dal.text_to_pql(body.question)
    except dal.PredictionFlowError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"data": result}
