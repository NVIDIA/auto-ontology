# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for files that must survive Python wheel packaging."""

from __future__ import annotations

import subprocess
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
PERMISSION_RESOURCE = "auto_ontology/server/users/access.json"


def test_wheel_contains_the_permission_resource(tmp_path: Path) -> None:
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1

    with ZipFile(wheels[0]) as wheel:
        packaged = wheel.read(PERMISSION_RESOURCE)

    source = (ROOT / "frontend" / "auth" / "access.json").read_bytes()
    assert packaged == source
