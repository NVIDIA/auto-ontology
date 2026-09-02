# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build every fixture database in one command.

The Postgres fixture (``pagila``) and the SQLite one (``chinook``) have nothing
in common beyond being fixtures, so each has its own module. This is the entry
point that runs both, so setup is one command rather than two.

Usage::

    docker compose up -d postgres
    uv run --no-sync python -m dev_tools.seed_fixtures

See ``dev_tools/sql/README.md`` for what each fixture covers and why.
"""

from __future__ import annotations

import logging

from dev_tools import build_sqlite_fixtures, seed_local_postgres

logger = logging.getLogger("dev_tools.seed_fixtures")


def main() -> None:
    logger.info("Seeding Postgres fixtures...")
    seed_local_postgres.seed()

    logger.info("Building SQLite fixtures...")
    for built in build_sqlite_fixtures.build_all():
        logger.info("Ready: %s (%s bytes)", built.name, f"{built.stat().st_size:,}")

    logger.info("All fixtures ready.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
