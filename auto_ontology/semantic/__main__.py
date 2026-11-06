# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point for semantic layer compilation.

Usage:
    python -m auto_ontology.semantic --database-name <database_name>
    python -m auto_ontology.semantic --database-name <database_name> --only-table-roles
    python -m auto_ontology.semantic --database-name <database_name> --only-semantic-fks
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
        "--only-table-roles",
        action="store_true",
        help=(
            "Only classify table roles and sibling tables, on a database that "
            "is already compiled."
        ),
    )
    parser.add_argument(
        "--only-semantic-fks",
        action="store_true",
        help="Only resolve semantic FKs, on a database that is already compiled.",
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

    from auto_ontology.semantic.compile import (
        run_semantic_compilation,
        run_semantic_fk_resolution,
        run_table_role_classification,
    )

    if args.only_semantic_fks:
        run_semantic_fk_resolution(args.database_name)
    elif args.only_table_roles:
        run_table_role_classification(args.database_name)
    else:
        run_semantic_compilation(args.database_name)


if __name__ == "__main__":
    main()
