# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point for semantic layer compilation.

Usage:
    python -m auto_ontology.semantic --database-name <database_name>
"""

from __future__ import annotations

import argparse
import logging

from auto_ontology.env import load_env

load_env()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile business taxonomy (Term, ColumnAttribute, sample values)"
    )
    parser.add_argument(
        "--database-name",
        required=True,
        help="Database name — must match the SQL connector and tabular ingest.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from auto_ontology.semantic.compile import run_semantic_compilation

    run_semantic_compilation(args.database_name)


if __name__ == "__main__":
    main()
