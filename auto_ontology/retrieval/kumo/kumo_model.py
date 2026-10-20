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
    """A KumoRFM handle with the cheap-validate + batching-predict helpers."""

    def __init__(self, model: Any, graph: Any) -> None:
        self.model = model
        self.graph = graph
        self._graph_def: Any = None

    def validate_pql(self, query: str) -> Any:
        """Cheaply parse/validate a PQL query against the graph (no inference).

        Raises ``ValueError`` with the parser's detail on invalid PQL — the
        repair-loop signal. Resolves locally, so it costs no request.
        """
        from kumo_relational_engine.rfm.query_parser import parse_query_locally

        if self._graph_def is None:
            self._graph_def = self.graph._to_api_graph_definition()
        return parse_query_locally(query, self._graph_def)

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
        batch_size = "max" if indices is not None and len(indices) > cap else None
        return self.model.predict(
            query,
            indices=indices,
            run_mode=run_mode,
            batch_size=batch_size,
            num_retries=num_retries,
            **kwargs,
        )


def _col_name(col: Any) -> str | None:
    return getattr(col, "name", None) if col is not None else None


def is_synthetic_key(name: Any) -> bool:
    """True for the surrogate column KumoRFM materializes for a composite key.

    Declaring ``primary_key = ('Customer ID', 'REGION')`` adds a hashed
    ``__kumo_key_*`` column to both ends of the relationship and states the edge in
    terms of it. It names no warehouse column, so it must not reach the DDL the LLM
    reads, the entity-selection SQL, or the seed values.
    """
    return bool(name) and str(name).startswith("__kumo_key")


def key_columns(table: Any) -> list[str]:
    """Real primary-key columns of a graph table, composite or single.

    ``primary_key_columns`` carries the tuple a composite key was declared with.
    ``primary_key`` collapses to the surrogate column in that case, so it is only
    read when it names a real column.
    """
    declared = tuple(getattr(table, "primary_key_columns", ()) or ())
    if declared:
        return [str(c) for c in declared]
    name = _col_name(getattr(table, "primary_key", None))
    return [] if not name or is_synthetic_key(name) else [str(name)]


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

    Names are quoted the way PQL has to spell them, so a column called ``Customer ID``
    reaches the model as ``` `Customer ID` ``` and comes back written that way. The
    surrogate column standing in for a composite key is left out throughout: it names
    nothing the model could reference, and a composite key is instead shown as its
    real columns.
    """
    from auto_ontology.retrieval.kumo.pql_gen import quote_name

    edges: list[tuple[str, str, str]] = []
    for e in graph.edges:
        # A composite edge reports the surrogate on both ends, which spells no real
        # column; the destination's identity is what a query can actually name. Its
        # parts are bracketed so the pair reads as one key rather than two columns.
        dst_key = key_columns(graph[e.dst_table]) if is_synthetic_key(e.fkey) else []
        if len(dst_key) > 1:
            fkey = f"({', '.join(quote_name(c) for c in dst_key)})"
        elif dst_key:
            fkey = quote_name(dst_key[0])
        else:
            fkey = quote_name(str(e.fkey))
        edges.append((e.src_table, fkey, e.dst_table))

    col_stypes: dict[str, dict[str, str]] = {}
    time_columns: dict[str, str | None] = {}
    ddl_lines: list[str] = []

    for name, table in graph.tables.items():
        columns = [c for c in table.columns if not is_synthetic_key(c.name)]
        col_stypes[name.lower()] = {c.name.lower(): str(c.stype) for c in columns}
        pk_cols = key_columns(table)
        time_col = _col_name(table.time_column)
        time_columns[name] = time_col

        col_txt = ", ".join(f"{quote_name(c.name)} {c.stype}" for c in columns)
        markers = []
        if pk_cols:
            markers.append(f"PRIMARY KEY ({', '.join(quote_name(c) for c in pk_cols)})")
        if time_col:
            markers.append(f"TIME COLUMN ({quote_name(time_col)})")
        suffix = f"  -- {'; '.join(markers)}" if markers else ""
        ddl_lines.append(f"{quote_name(name)}({col_txt}){suffix}")

    for src, fkey, dst in edges:
        ddl_lines.append(
            f"FOREIGN KEY {quote_name(src)}.{fkey} -> {quote_name(dst)}.<pk>"
        )

    return "\n".join(ddl_lines), edges, col_stypes, time_columns
