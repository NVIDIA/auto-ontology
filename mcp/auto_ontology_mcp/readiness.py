# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``check_readiness`` — whether this deployment can answer questions yet.

Hand-written rather than generated, because the answer is not any one endpoint.
A deployment can only answer a question when a catalog has been ingested, a live
database connection exists to run the SQL against, *and* the semantic layer has
been compiled to resolve the question with. Those three facts come from
unrelated routes, and no one of them is sufficient alone.

The combination is worth a tool of its own because its failure is silent. A
compiled glossary over no connection reads perfectly: ``search_terms`` returns
terms, ``get_term`` returns columns, and the layer reports itself as built.
Nothing looks wrong until ``ask_question`` spends minutes writing SQL it can
never execute and returns an empty answer — which is
indistinguishable, from the caller's side, from a question that was simply not
understood. One cheap call up front turns that dead end into a fact.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field

from auto_ontology_mcp.config import Settings

logger = logging.getLogger(__name__)

STATUS_PATH = "/api/semantic-compilation/status"
CONNECTIONS_PATH = "/api/connections"
DATABASES_PATH = "/api/datasources/dbs"


class _Forbidden:
    """A probe the caller is authenticated for but not permitted to read."""

    __slots__ = ()


FORBIDDEN = _Forbidden()


class Readiness(BaseModel):
    """What this deployment can currently do."""

    ready: bool = Field(
        description=(
            "True when a question can be understood, executed, and has data to "
            "run against."
        )
    )
    semantic_layer_built: bool | None = Field(
        default=False,
        description=(
            "Whether the glossary of business terms has been compiled. Null "
            "when the caller is not permitted to see it."
        ),
    )
    can_execute_sql: bool | None = Field(
        default=False,
        description=(
            "Whether a live database connection exists to run SQL against. "
            "Null when the caller is not permitted to see it, which is normal "
            "for non-admins: reading connections is often admin-only."
        ),
    )
    catalog_present: bool | None = Field(
        default=False,
        description=(
            "Whether any database has been ingested for questions to reach. "
            "Null when the caller is not permitted to see it."
        ),
    )
    databases: list[str] = Field(
        default_factory=list,
        description=(
            "Databases present in the catalog. These can be non-empty while "
            "can_execute_sql is false: the catalog outlives the connection it "
            "was ingested from."
        ),
    )
    blockers: list[str] = Field(
        default_factory=list,
        description="What stands in the way, and what to do about it. Empty when ready.",
    )
    unverified: list[str] = Field(
        default_factory=list,
        description=(
            "Facts this caller is not permitted to check. These do not make a "
            "deployment unready — they are gaps in what can be seen, not in "
            "what works, so treat a ready verdict alongside them as ready."
        ),
    )


async def _probe(client: httpx.AsyncClient, path: str) -> Any:
    """GET *path*, returning ``None`` when it cannot be read.

    A readiness check that dies on its first bad response is not much of a
    readiness check, so an unreadable endpoint degrades to an unknown that the
    caller is told about.

    The two rejections mean different things and cannot share a fate. A 401 is
    the credential itself: every probe would fail the same way, and reporting
    "not ready" for what is really a stale token would send the caller off
    fixing the wrong thing. A 403 is this caller against this route, and the
    three probes do not need the same permission — a viewer may run chat and
    browse the catalog while only admins read connections. Refusing to answer
    at all there would deny a caller the verdict over a fact they never needed.
    """
    try:
        response = await client.get(path)
    except httpx.HTTPError as exc:
        logger.warning("Readiness probe %s failed: %s", path, exc)
        return None

    if response.status_code == 401:
        raise ToolError(
            "Auto Ontology rejected the credentials while checking readiness. Sign in "
            "again. (HTTP 401)"
        )
    if response.status_code == 403:
        logger.info("Readiness probe %s is not permitted for this caller", path)
        return FORBIDDEN
    if response.status_code != 200:
        logger.warning(
            "Readiness probe %s returned HTTP %d", path, response.status_code
        )
        return None
    try:
        return response.json()
    except ValueError:
        logger.warning("Readiness probe %s did not return JSON", path)
        return None


def _summarise(status: Any, connections: Any, databases: Any) -> Readiness:
    """Fold the three probes into one verdict.

    A forbidden probe is recorded as unverified rather than as a blocker, so a
    caller who cannot read connections still gets a usable verdict instead of a
    deployment being called broken on the strength of what they cannot see.
    """
    blockers: list[str] = []
    unverified: list[str] = []

    if status is FORBIDDEN:
        blockers.append(
            "This account is not permitted to use chat on this deployment "
            f"({STATUS_PATH} returned 403), so it cannot ask questions here "
            "regardless of how the deployment is set up. Ask an administrator "
            "for chat access."
        )
        built: bool | None = None
    elif status is None:
        blockers.append(
            "Could not read whether the semantic layer is compiled "
            f"({STATUS_PATH} was unreadable). Auto Ontology may be starting up or "
            "partially deployed."
        )
        built = False
    else:
        built = bool(status.get("calculated"))
        if not built:
            blockers.append(
                "The semantic layer has not been compiled, so there is no "
                "glossary to resolve a question against. Compile it in the Auto Ontology "
                "UI; nothing can be asked of the data until it finishes."
            )

    if connections is FORBIDDEN:
        unverified.append(
            "Whether a live database connection exists could not be checked: "
            f"this account is not permitted to read {CONNECTIONS_PATH}, which "
            "is often admin-only. Questions can still be asked; if one comes "
            "back empty, a missing connection is a likely cause."
        )
        can_execute: bool | None = None
    elif connections is None:
        blockers.append(
            f"Could not read the configured connections ({CONNECTIONS_PATH} was "
            "unreadable), so it is unknown whether SQL can execute."
        )
        can_execute = False
    else:
        can_execute = int(connections.get("count") or 0) > 0
        if not can_execute:
            blockers.append(
                "No database connection is configured, so generated SQL cannot "
                "execute. Questions will still be accepted and will still cost "
                "a full agent run before failing with an empty answer. Add a "
                "connection in the Auto Ontology UI."
            )

    if databases is FORBIDDEN:
        unverified.append(
            "Whether any database has been ingested could not be checked: this "
            f"account is not permitted to read {DATABASES_PATH}. Questions can "
            "still be asked."
        )
        names: list[str] = []
        catalog_present: bool | None = None
    elif databases is None:
        blockers.append(
            f"Could not read the catalog ({DATABASES_PATH} was unreadable), so "
            "it is unknown whether any data has been ingested."
        )
        names = []
        catalog_present = False
    else:
        names = [
            str(entry.get("name"))
            for entry in (databases.get("data") or [])
            if entry.get("name")
        ]
        catalog_present = bool(names)
        if not names:
            blockers.append(
                "No database has been ingested, so there is nothing for a "
                "question to reach. A connection can exist before its catalog "
                "does: ingestion runs after the connection is saved, and an "
                "unfinished or failed ingest leaves the catalog empty. Check "
                "the ingestion service, or re-save the connection in the Auto Ontology UI."
            )

    return Readiness(
        ready=not blockers,
        semantic_layer_built=built,
        can_execute_sql=can_execute,
        catalog_present=catalog_present,
        databases=names,
        blockers=blockers,
        unverified=unverified,
    )


def register(mcp: FastMCP, settings: Settings, client: httpx.AsyncClient) -> None:
    """Attach ``check_readiness`` to *mcp*."""

    @mcp.tool(
        name="check_readiness",
        description=(
            "Report whether this Auto Ontology deployment can answer questions at all: "
            "whether a catalog has been ingested, whether a live database "
            "connection exists to run SQL against, and whether the semantic "
            "layer has been compiled. All three are required.\n\n"
            "Use it before the first question against an unfamiliar "
            "deployment, and whenever ask_question returns an empty answer. It is "
            "three cheap reads, and it separates 'this deployment is not set "
            "up' from 'the question was not understood' — which otherwise look "
            "identical and are fixed in completely different places.\n\n"
            "Facts the signed-in account may not read come back under "
            "'unverified' rather than as blockers, so a ready verdict with "
            "unverified entries still means questions can be asked."
        ),
    )
    async def check_readiness() -> Readiness:
        """Probe the deployment and report what it can currently do."""
        status, connections, databases = await asyncio.gather(
            _probe(client, STATUS_PATH),
            _probe(client, CONNECTIONS_PATH),
            _probe(client, DATABASES_PATH),
        )
        readiness = _summarise(status, connections, databases)
        if not readiness.ready:
            logger.info("Deployment not ready: %s", "; ".join(readiness.blockers))
        return readiness


__all__ = [
    "CONNECTIONS_PATH",
    "DATABASES_PATH",
    "FORBIDDEN",
    "STATUS_PATH",
    "Readiness",
    "register",
]
