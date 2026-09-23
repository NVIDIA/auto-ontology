# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KumoRFM-backed prediction for the text-to-SQL agent."""

from auto_ontology.retrieval.kumo.predictor import (
    PredictionContext,
    build_prediction_context,
    run_prediction,
)

__all__ = [
    "PredictionContext",
    "build_prediction_context",
    "run_prediction",
]
