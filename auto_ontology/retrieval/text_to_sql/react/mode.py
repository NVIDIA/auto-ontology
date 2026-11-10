# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tools an episode calls, and how it may answer: with SQL, with a KumoRFM
prediction, or either."""

from __future__ import annotations

from enum import StrEnum

from auto_ontology.retrieval.text_to_sql.react import config


class ToolName(StrEnum):
    RETRIEVE = "retrieve"
    RUN_SQL = "run_sql"
    SUBMIT = "submit"
    PREDICT = "predict"


ANSWERING_TOOLS = frozenset({ToolName.SUBMIT, ToolName.PREDICT})


class EpisodeMode(StrEnum):
    SQL = "sql"
    PREDICT = "predict"
    AUTO = "auto"


def episode_mode(prediction_override: bool | None) -> EpisodeMode:
    """``True`` forces a prediction, ``False`` forbids one, and ``None`` offers
    both when KumoRFM is configured."""
    if prediction_override is True:
        return EpisodeMode.PREDICT
    if prediction_override is None and config.prediction_enabled():
        return EpisodeMode.AUTO
    return EpisodeMode.SQL
