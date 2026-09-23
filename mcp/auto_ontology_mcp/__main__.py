# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auto Ontology MCP server entrypoint.

Exposes the Auto Ontology semantic layer to any MCP-capable agent harness. It holds no
database connections, no model configuration, and no credentials: it is an HTTP
client of the public Auto Ontology API and needs only the URL of a Auto Ontology deployment, so it
runs equally well beside one or on a laptop far away from it.

Usage::

    export AUTO_ONTOLOGY_API_URL=https://auto_ontology.example.com
    auto-ontology-mcp

Without installing anything::

    uvx --from auto-ontology-mcp auto-ontology-mcp

Callers sign in against Auto Ontology itself in a browser, so there is nothing to give the
server up front and nothing for a user to mint by hand.
"""

from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

# Load before importing config so a `.env` beside the working directory can
# supply the URL, matching how the rest of Auto Ontology is configured locally.
load_dotenv()

from auto_ontology_mcp import get_version  # noqa: E402
from auto_ontology_mcp.config import ConfigError, load_settings  # noqa: E402
from auto_ontology_mcp.server import build_server  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        settings = load_settings()
    except ConfigError as exc:
        logger.error("%s", exc)
        return 2

    logger.info(
        "Starting Auto Ontology MCP server — version %s, listening on %s",
        get_version(),
        settings.public_url,
    )

    mcp, _client = build_server(settings)

    # FastMCP owns the event loop and closes the client's connections when the
    # transport shuts down, so there is no separate teardown to run here.
    mcp.run(transport="http", host=settings.host, port=settings.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
