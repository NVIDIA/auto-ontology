# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Copy the shared permission grants into the Python package.

``frontend/auth/access.json`` is the source of truth. The generated package
resource makes an installed wheel self-contained:

    uv run python -m dev_tools.generate_permission_resource
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "frontend" / "auth" / "access.json"
TARGET = ROOT / "auto_ontology" / "server" / "users" / "access.json"


def generate() -> None:
    """Copy the source bytes exactly so drift is easy to detect."""
    TARGET.write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    generate()
