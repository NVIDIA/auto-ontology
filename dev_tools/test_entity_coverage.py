# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the entity-coverage pipeline for a single question (local debug).

Usage::

    uv run python -m dev_tools.test_entity_coverage \\
        --question "How many shipments were delivered last month?"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from gsf.env import load_env

load_env()

from gsf.connectors import get_connectors  # noqa: E402
from gsf.connectors.registry import invalidate_connectors_cache  # noqa: E402
from gsf.retrieval.entity_coverage.main import get_coverage_response  # noqa: E402
from gsf.retrieval.entity_coverage.state import (  # noqa: E402
    DEFAULT_MAX_DISTANCE,
    EntityCoveragePayload,
)
from gsf.server.chat.settings_dal import (  # noqa: E402
    fetch_acronyms,
    fetch_custom_prompts,
)
from gsf.utils.retriever import (  # noqa: E402
    close_retrievers,
    get_data_objects_retriever,
    get_semantic_objects_retriever,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("test_entity_coverage")


def _cleanup() -> None:
    """Close pools/loops so the process can exit after a one-shot debug run."""
    close_retrievers()
    invalidate_connectors_cache()
    try:
        from nemo_retriever.tabular_data.neo4j import neo4j_connection

        if neo4j_connection._conn is not None:
            neo4j_connection._conn.close()
            neo4j_connection._conn = None
    except Exception:
        logger.exception("Failed to close Neo4j connection during cleanup")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test entity-coverage endpoint logic")
    parser.add_argument(
        "--question",
        "-q",
        required=True,
        help="Natural-language question to grade for entity coverage",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=DEFAULT_MAX_DISTANCE,
        help=f"Max L2 distance for a candidate (default {DEFAULT_MAX_DISTANCE})",
    )
    args = parser.parse_args(argv)

    connectors = get_connectors()
    if not connectors:
        logger.error("No connectors configured (check CONNECTION_STRINGS in .env)")
        return 1

    # Same settings the API reads, so a debug run sees the Glossary the UI shows.
    acronyms = fetch_acronyms()
    custom_prompts = fetch_custom_prompts()
    logger.info("Loaded %d glossary definition(s)", len(acronyms))

    payload: EntityCoveragePayload = {
        "question": args.question,
        "data_retriever": get_data_objects_retriever(),
        "semantic_retriever": get_semantic_objects_retriever(),
        "connectors": connectors,
        "acronyms": acronyms,
        "custom_prompts": custom_prompts,
        "max_distance": args.max_distance,
    }

    try:
        logger.info("Running entity coverage for: %s", args.question)
        result = get_coverage_response(payload)
        print(json.dumps(result, indent=2, default=str))
        return 0
    finally:
        _cleanup()


if __name__ == "__main__":
    sys.exit(main())
