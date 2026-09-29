# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InteractivePhase(str, Enum):
    ROUND1_CLARIFY = "round1_clarify"
    ROUND2_CLARIFY = "round2_clarify"


class TurnType(str, Enum):
    INITIAL = "initial"  # first message for a phase (clarify + submit)
    DEBUG = "debug"  # prior submission was rejected; retry with feedback
    FOLLOW_UP = "follow_up"  # prior submission completed; a new question follows


@dataclass
class AskUserAction:
    question: str


@dataclass
class SubmitSQLAction:
    sql: str
