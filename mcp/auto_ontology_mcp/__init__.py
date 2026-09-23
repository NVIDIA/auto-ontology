# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP server exposing the Auto Ontology semantic layer to agent harnesses."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _package_version

DISTRIBUTION = "auto-ontology-mcp"


def get_version() -> str:
    """Return this package's version, as advertised in the MCP handshake.

    Falls back rather than raising: running from a source tree that was never
    installed is a legitimate way to develop, and it should not be the reason
    the server refuses to start.
    """
    try:
        return _package_version(DISTRIBUTION)
    except PackageNotFoundError:
        return "unknown"


__all__ = ["DISTRIBUTION", "get_version"]
