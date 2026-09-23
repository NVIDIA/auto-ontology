# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment bootstrap shared by all Auto Ontology process entrypoints."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

_repo_root = Path(__file__).resolve().parent.parent


def load_env() -> None:
    """Load the repo-root ``.env`` file into the process environment."""
    load_dotenv(_repo_root / ".env")
