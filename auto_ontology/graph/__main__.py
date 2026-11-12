# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI for the Apache AGE projection.

Usage:
    python -m auto_ontology.graph sync
    python -m auto_ontology.graph status
    python -m auto_ontology.graph neighborhood <term_id> [--depth N]
    python -m auto_ontology.graph query "MATCH (t:Term) RETURN t.name LIMIT 5"

``query`` runs in a read-only transaction that is rolled back, so it cannot
change the graph; ``sync`` is the only command that writes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from auto_ontology.env import load_env

load_env()


def _print(value: object) -> None:
    json.dump(value, sys.stdout, indent=2, ensure_ascii=False, default=str)
    sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m auto_ontology.graph",
        description="Project the Auto Ontology catalog into an Apache AGE graph.",
    )
    parser.add_argument(
        "--graph", help="Graph name (default: $AGE_GRAPH_NAME or auto_ontology)."
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sync = sub.add_parser("sync", help="Rebuild the graph from Postgres.")
    sync.add_argument("--batch-size", type=int, default=1000)

    sub.add_parser("status", help="Show vertex and edge counts per label.")

    hood = sub.add_parser("neighborhood", help="Vertices and edges around one Term.")
    hood.add_argument("term_id")
    hood.add_argument("--depth", type=int, default=1)
    hood.add_argument("--max-nodes", type=int, default=200)

    query = sub.add_parser("query", help="Run a read-only Cypher statement.")
    query.add_argument("cypher")
    query.add_argument(
        "--columns",
        default="v",
        help="Comma-separated names for the RETURN columns, in order (default: v).",
    )
    query.add_argument("--params", help="JSON object bound as $name parameters.")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from auto_ontology.graph import age, projection, queries

    try:
        if args.command == "sync":
            report = projection.sync_graph(graph=args.graph, batch_size=args.batch_size)
            _print(report.as_dict())
        elif args.command == "status":
            _print(queries.graph_status(graph=args.graph))
        elif args.command == "neighborhood":
            result = queries.term_neighborhood(
                args.term_id,
                depth=args.depth,
                max_nodes=args.max_nodes,
                graph=args.graph,
            )
            if result is None:
                print(f"Term {args.term_id!r} is not in the graph", file=sys.stderr)
                return 1
            _print(result)
        else:
            columns = tuple(c.strip() for c in args.columns.split(",") if c.strip())
            params = json.loads(args.params) if args.params else None
            rows = queries.run_cypher(
                args.cypher, params, columns=columns, graph=args.graph
            )
            _print([dict(zip(columns, row, strict=True)) for row in rows])
    except (age.AgeUnavailable, age.GraphNotBuilt, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
