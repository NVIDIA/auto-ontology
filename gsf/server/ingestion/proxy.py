# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP client for the ingestion service API.

Lightweight on purpose: it only depends on ``httpx`` so the server can trigger
ingest over HTTP without importing ``gsf.ingestion_service.ingest`` (which pulls
in heavy ``nemo_retriever`` deps and requires ``REASONING_API_KEY`` at import time).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_INGESTION_SERVICE_URL = "http://localhost:3002"

# The service endpoints return 202 immediately, so a short timeout is enough.
_REQUEST_TIMEOUT = 10.0


def _base_url() -> str:
    """Base URL of the ingestion service, configurable for local vs Docker."""
    url = os.environ.get("INGESTION_SERVICE_URL", DEFAULT_INGESTION_SERVICE_URL)
    return url.rstrip("/")


def trigger_ingest(connection: dict[str, Any]) -> None:
    """Ask the ingestion service to ingest a connection (best-effort)."""
    url = f"{_base_url()}/ingest"
    try:
        httpx.post(url, json=connection, timeout=_REQUEST_TIMEOUT).raise_for_status()
    except Exception:
        logger.exception("Failed to trigger ingest via %s", url)


def trigger_ingest_delete(database_name: str) -> None:
    """Ask the ingestion service to tear down a database's data (best-effort)."""
    url = f"{_base_url()}/ingest/delete"
    try:
        httpx.post(
            url,
            json={"database_name": database_name},
            timeout=_REQUEST_TIMEOUT,
        ).raise_for_status()
    except Exception:
        logger.exception("Failed to trigger ingest delete via %s", url)


def trigger_semantic_compile() -> None:
    """Ask the ingestion service to run semantic compilation (best-effort).

    The service starts its semantic scheduler if it isn't running yet, so this
    also takes effect when compilation is enabled while the service is up.
    """
    url = f"{_base_url()}/semantic/compile"
    try:
        httpx.post(url, timeout=_REQUEST_TIMEOUT).raise_for_status()
    except Exception:
        logger.exception("Failed to trigger semantic compile via %s", url)
