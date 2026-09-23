# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Settings for the Auto Ontology MCP server, read from the environment.

The server talks to the **public** Auto Ontology API, not to the Python services behind
it: those are ClusterIP-only and reachable solely through the Next.js proxy,
which is where authentication and permission checks live. So all it needs is the
base URL of a Auto Ontology deployment.

It holds no credential of its own, and there is no setting that would give it
one. Callers sign in against Auto Ontology itself — the same login page and the same SSO
behind it that they would meet in a browser — and every call then runs as the
person who signed in, with exactly their permissions. Nothing to mint, paste, or
share, and no way to configure a single identity that everybody silently
inherits.

That leaves ``AUTO_ONTOLOGY_API_URL`` as the only setting most deployments touch. Pointing
at a laptop's stack or at a production instance differs only in that value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# The description of the public API that tools are generated from, shipped
# inside the package so the server runs from an ordinary install rather than
# only from a source checkout. Kept identical to docs/openapi/auto-ontology-api.json by
# `pnpm openapi`, and enforced by CI — a stale copy would silently yield a
# stale tool surface.
DEFAULT_SPEC_PATH = Path(__file__).resolve().parent / "auto-ontology-api.json"

DEFAULT_API_URL = "http://localhost:3000"

# 3000 frontend, 3001 backend, 3002 ingestion — this is the next free one.
DEFAULT_PORT = 3003
DEFAULT_HOST = "0.0.0.0"

# Catalog and glossary reads are ordinary API calls.
DEFAULT_TIMEOUT_S = 30.0

# One chat turn is many sequential LLM calls. The backend caps individual SQL
# statements at 30s but puts no ceiling on a whole run, so this is a client-side
# guard against waiting forever rather than a mirror of a server-side limit.
DEFAULT_CHAT_TIMEOUT_S = 900.0


class ConfigError(RuntimeError):
    """The environment does not describe a usable server."""


@dataclass(frozen=True)
class Settings:
    """Everything the server needs to start."""

    api_url: str
    spec_path: Path
    host: str
    port: int
    timeout_s: float
    chat_timeout_s: float
    # Where callers reach this server, as opposed to where it binds. Clients are
    # told to come back here after signing in, and they check that the address
    # the server names for itself is the one they dialled — so a bind address
    # that is not how anyone addresses it needs this set explicitly.
    public_url: str = ""


def _positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be greater than zero, got {value}")
    return value


def _port(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if not 1 <= value <= 65535:
        raise ConfigError(f"{name} must be a valid port, got {value}")
    return value


def _public_url(host: str, port: int) -> str:
    """Where callers reach this server, for sign-in redirects and metadata."""
    raw = (os.environ.get("AUTO_ONTOLOGY_MCP_PUBLIC_URL") or "").strip()
    if raw:
        return raw.rstrip("/")

    # 0.0.0.0 means "every interface", which is a fine thing to bind to and a
    # useless thing to send a browser to.
    hostname = "localhost" if host in {"", "0.0.0.0", "::"} else host
    return f"http://{hostname}:{port}"


def load_settings() -> Settings:
    """Build :class:`Settings` from the environment.

    Raises :class:`ConfigError` with an actionable message rather than failing
    later on the first request.
    """

    spec_path = Path(
        (os.environ.get("AUTO_ONTOLOGY_OPENAPI_SPEC") or "").strip()
        or DEFAULT_SPEC_PATH
    )
    if not spec_path.is_file():
        # The packaged copy is always present in a sound install, so this is
        # either a bad AUTO_ONTOLOGY_OPENAPI_SPEC override or a broken package.
        raise ConfigError(
            f"OpenAPI spec not found at {spec_path}. Unset AUTO_ONTOLOGY_OPENAPI_SPEC to "
            "use the copy shipped with auto-ontology-mcp, or reinstall the package."
        )

    api_url = (os.environ.get("AUTO_ONTOLOGY_API_URL") or DEFAULT_API_URL).strip()
    host = (os.environ.get("AUTO_ONTOLOGY_MCP_HOST") or DEFAULT_HOST).strip()
    port = _port("AUTO_ONTOLOGY_MCP_PORT", DEFAULT_PORT)

    return Settings(
        # Trailing slashes make httpx base_url joins produce doubled separators.
        api_url=api_url.rstrip("/"),
        spec_path=spec_path,
        host=host,
        port=port,
        timeout_s=_positive_float("AUTO_ONTOLOGY_MCP_TIMEOUT_S", DEFAULT_TIMEOUT_S),
        chat_timeout_s=_positive_float(
            "AUTO_ONTOLOGY_MCP_CHAT_TIMEOUT_S", DEFAULT_CHAT_TIMEOUT_S
        ),
        public_url=_public_url(host, port),
    )


__all__ = [
    "ConfigError",
    "DEFAULT_SPEC_PATH",
    "Settings",
    "load_settings",
]
