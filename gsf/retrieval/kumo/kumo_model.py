# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KumoRFM model wrapper + graph-context extraction (ported from aiq-rfm).

``KumoModel`` is the thin seam ``pql_gen`` expects: a cheap ``validate_pql`` parse
(the repair-loop signal) and a ``predict`` that transparently batches above the
per-call cap. ``build_graph_context`` derives the prompt's ``graph_ddl``, the FK
``edges`` (for the static lint), the per-column ``stypes``, and the time columns
(for forecast anchoring) from a built ``LocalGraph``.
"""

from __future__ import annotations

from typing import Any

# Per-call entity caps above which predict() switches to batch mode (from aiq-rfm).
_PREDICT_CAP = 1_000
_LINK_PREDICT_CAP = 200


def is_link_prediction(query: str) -> bool:
    """Heuristic: a PQL query is link prediction (recommendation) if it ranks a list."""
    upper = query.upper()
    return "LIST_DISTINCT" in upper or "RANK TOP" in upper


class KumoModel:
    """A built ``KumoRFM`` with the cheap-validate + batching-predict helpers."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def validate_pql(self, query: str) -> Any:
        """Cheaply parse/validate a PQL query against the graph (no inference).

        Raises ``ValueError`` with the server's detail on invalid PQL — the
        repair-loop signal. Wraps ``KumoRFM._parse_query`` (the seam in
        kumorfm 2.23.0), kept here so the call site is swappable.
        """
        return self.model._parse_query(query)

    def predict(
        self,
        query: str,
        indices: list[Any] | None = None,
        *,
        run_mode: str = "fast",
        num_retries: int = 1,
        **kwargs: Any,
    ) -> Any:
        """Run a prediction, transparently batching when ``indices`` exceeds the cap."""
        cap = _LINK_PREDICT_CAP if is_link_prediction(query) else _PREDICT_CAP
        if indices is not None and len(indices) > cap:
            with self.model.batch_mode(batch_size="max", num_retries=num_retries):
                return self.model.predict(
                    query, indices=indices, run_mode=run_mode, **kwargs
                )
        return self.model.predict(query, indices=indices, run_mode=run_mode, **kwargs)


def _col_name(col: Any) -> str | None:
    return getattr(col, "name", None) if col is not None else None


def build_graph_context(
    graph: Any,
) -> tuple[
    str, list[tuple[str, str, str]], dict[str, dict[str, str]], dict[str, str | None]
]:
    """Derive ``(graph_ddl, edges, col_stypes, time_columns)`` from a ``LocalGraph``.

    * ``graph_ddl`` — a compact text schema (tables, columns+stypes, PK, time col, FKs)
      for the prompt's "Graph (tables, keys, links)" section.
    * ``edges`` — ``(src_table, fkey, dst_table)`` list, for the static lint's
      direct-foreign-key check.
    * ``col_stypes`` — ``{table_lower: {col_lower: stype}}`` for the ordinal-comparison lint.
    * ``time_columns`` — ``{table: time_column_name | None}`` for forecast anchoring.
    """
    edges: list[tuple[str, str, str]] = [
        (e.src_table, e.fkey, e.dst_table) for e in graph.edges
    ]

    col_stypes: dict[str, dict[str, str]] = {}
    time_columns: dict[str, str | None] = {}
    ddl_lines: list[str] = []

    for name, table in graph.tables.items():
        columns = list(table.columns)
        col_stypes[name.lower()] = {c.name.lower(): str(c.stype) for c in columns}
        pk = _col_name(table.primary_key)
        time_col = _col_name(table.time_column)
        time_columns[name] = time_col

        col_txt = ", ".join(f"{c.name} {c.stype}" for c in columns)
        markers = []
        if pk:
            markers.append(f"PRIMARY KEY ({pk})")
        if time_col:
            markers.append(f"TIME COLUMN ({time_col})")
        suffix = f"  -- {'; '.join(markers)}" if markers else ""
        ddl_lines.append(f"{name}({col_txt}){suffix}")

    for src, fkey, dst in edges:
        ddl_lines.append(f"FOREIGN KEY {src}.{fkey} -> {dst}.<pk>")

    return "\n".join(ddl_lines), edges, col_stypes, time_columns
