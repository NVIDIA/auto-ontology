# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent skills name operations, permissions, tools, variables, and files
that exist, so an agent following them reaches the product it was told about."""

from __future__ import annotations

from dev_tools.check_skill_claims import check_all


def test_every_skill_claim_matches_the_code() -> None:
    findings = check_all()
    assert not findings, "\n".join(str(f) for f in findings)
