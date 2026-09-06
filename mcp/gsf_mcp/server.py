# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assemble the GSF MCP server from the published OpenAPI spec."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import Icon

from gsf_mcp import chat, readiness
from gsf_mcp.config import ConfigError, Settings
from gsf_mcp.gsf_auth import build_gsf_auth, gsf_access_token
from gsf_mcp.tools import (
    apply_description,
    missing_from_spec,
    names_by_operation_id,
    route_maps,
)
from gsf_mcp import get_version

logger = logging.getLogger(__name__)

SERVER_NAME = "gsf"

# Shipped in the package and inlined as a data URI rather than linked, so it
# needs no network and no reachable GSF to display — a client may well draw its
# server list before anything is connected.
ICON_PATH = Path(__file__).resolve().parent / "nvidia-mark.svg"
ICON_MIME_TYPE = "image/svg+xml"

# Advertised to the host at initialize. Tool descriptions say what each tool
# does; this says how they fit together, which is the part a model otherwise
# has to infer from names — and infers badly, usually by reaching straight for
# the expensive one.
INSTRUCTIONS = """\
GSF (Generative Semantic Fabric) answers questions about an organisation's
structured data. It holds a compiled semantic layer — a glossary of business
terms mapped onto real database columns and reviewed SQL expressions — over the
databases this deployment is connected to.

Use `ask_question` for anything that needs an actual answer from the data. It
runs a full text-to-SQL agent and returns the answer, the SQL it ran, and the
rows.

The other tools exist so you can understand the vocabulary before you ask, and
check your assumptions after. A productive sequence is usually:

1. `search_terms` to find out what a business word means here. Deployments
   differ: "active customer" is a defined term with specific SQL behind it,
   not something to guess at.
2. `check_answerable` if you are unsure the question is in scope. It is far
   cheaper than `ask_question` and tells you whether the semantic layer covers
   the entities involved.
3. `ask_question` to get the answer.

`check_readiness` answers a different question: whether this deployment can
answer anything at all. A deployment with no database connection reads normally
and still fails every question, so reach for it before a first question here, or
when `ask_question` comes back empty — an empty answer is far more often a
deployment that is not set up than a question that was misunderstood.

Work through the semantic layer, not around it. `get_term_columns` tells you
which physical columns a term stands for, `get_term_sql_attributes` and
`get_sql_attribute` give you expressions that were already reviewed here, and
`describe_table` says what a table means rather than only its shape. There is
deliberately no tool for browsing databases, schemas, or raw columns:
questions are answered against the glossary.

Every tool reads. Nothing here modifies the catalog, the glossary, or the
underlying databases.
"""


def load_spec(settings: Settings) -> dict[str, Any]:
    """Read and validate the OpenAPI document the tools are generated from."""
    try:
        spec = json.loads(settings.spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{settings.spec_path} is not valid JSON: {exc}") from exc

    if not isinstance(spec, dict) or not spec.get("paths"):
        raise ConfigError(f"{settings.spec_path} has no paths; is it an OpenAPI spec?")

    # A curated entry that matches nothing would otherwise drop that tool
    # silently, leaving a server that looks healthy but is missing capability.
    missing = missing_from_spec(spec)
    if missing:
        raise ConfigError(
            "The OpenAPI spec no longer publishes these curated operations: "
            + ", ".join(missing)
            + ". Regenerate the spec, or update gsf_mcp/tools.py to match."
        )
    return spec


def load_icons() -> list[Icon]:
    """The NVIDIA mark, for clients that show an icon beside each server.

    Purely cosmetic, so a missing or unreadable file degrades to no icon rather
    than stopping a working server from starting.

    ``sizes`` is left unset deliberately: it is optional, an SVG has no natural
    pixel size, and clients have historically disagreed about whether the field
    is a string or a list.
    """
    try:
        svg = ICON_PATH.read_bytes()
    except OSError as exc:
        logger.warning("No icon at %s (%s); serving without one.", ICON_PATH, exc)
        return []

    encoded = base64.b64encode(svg).decode("ascii")
    return [
        Icon(src=f"data:{ICON_MIME_TYPE};base64,{encoded}", mimeType=ICON_MIME_TYPE)
    ]


BEARER_HEADER = "authorization"

# Not a credential this server accepts — one it refuses. FastMCP copies the
# incoming request's headers onto the outbound call, and GSF resolves an API key
# ahead of a bearer token, so a caller who sent one would be acting as its owner
# rather than as whoever they signed in as. It is stripped on the way out.
API_KEY_HEADER = "x-api-key"


class CallerAuth(httpx.Auth):
    """Authenticate every outbound call as the caller who triggered it.

    This server holds no credential, and there is no setting that would give it
    one. Identity comes from the sign-in the caller completed against GSF, and
    the access token GSF issued for it is forwarded upstream unchanged: GSF is
    both the authorization server and the resource server here, so there is
    nothing to exchange and no id token to unwrap.

    Attaching this as httpx auth rather than at each call site is deliberate: it
    covers the generated tools and the hand-written streaming one through the
    single client they share, so no tool can be added later that forgets to
    authenticate.
    """

    def auth_flow(self, request: httpx.Request):  # type: ignore[override]
        token = gsf_access_token()
        if not token:
            raise ToolError(
                "This request carried no signed-in session. Sign in again "
                "through the GSF deployment this server is pointed at."
            )

        # A request must never carry two identities, whichever slot the
        # caller's arrived in, so drop both before setting ours.
        for name in (API_KEY_HEADER, BEARER_HEADER):
            if name in request.headers:
                del request.headers[name]
        request.headers[BEARER_HEADER] = f"Bearer {token}"
        yield request


def build_client(settings: Settings) -> httpx.AsyncClient:
    """HTTP client for the public GSF API.

    The credential is not set here: :class:`CallerAuth` resolves one per
    request, so a single client can serve callers with different identities.
    """
    return httpx.AsyncClient(
        base_url=settings.api_url,
        auth=CallerAuth(),
        timeout=settings.timeout_s,
        # An agent harness may fan out across several tools at once.
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    )


def build_server(settings: Settings) -> tuple[FastMCP, httpx.AsyncClient]:
    """Build the server and the client it owns.

    The client is returned rather than hidden so the caller can close it; it
    outlives any single request because the streaming chat tool holds it open.
    """
    spec = load_spec(settings)
    client = build_client(settings)

    mcp: FastMCP = FastMCP.from_openapi(
        openapi_spec=spec,
        client=client,
        name=SERVER_NAME,
        route_maps=route_maps(),
        mcp_names=names_by_operation_id(spec),
        mcp_component_fn=apply_description,
        # Forwarded to the FastMCP constructor. Without an explicit version the
        # handshake advertises FastMCP's own, which reads as a GSF version to
        # anyone looking at the client's server list.
        instructions=INSTRUCTIONS,
        version=get_version(),
        icons=load_icons(),
        auth=build_gsf_auth(settings),
    )

    chat.register(mcp, settings, client)
    readiness.register(mcp, settings, client)

    logger.info(
        "GSF MCP server built against %s (spec: %s). Callers sign in against "
        "that GSF deployment; this server holds no credentials.",
        settings.api_url,
        settings.spec_path,
    )
    return mcp, client


__all__ = [
    "API_KEY_HEADER",
    "BEARER_HEADER",
    "ICON_PATH",
    "INSTRUCTIONS",
    "SERVER_NAME",
    "CallerAuth",
    "build_client",
    "build_server",
    "load_icons",
    "load_spec",
]
