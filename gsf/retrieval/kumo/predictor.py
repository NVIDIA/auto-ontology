# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KumoRFM prediction pipeline (GSF entry point).

Wires the ingested-catalog data into the ported text-to-PQL pipeline:
  1. Load a bounded sample of each ingested-catalog table into DataFrames.
  2. Build a KumoRFM ``LocalGraph`` (metadata + links inferred) and a DuckDB
     mirror of the same frames (so the entity-selection SQL resolves).
  3. Run :func:`gsf.retrieval.kumo.pql_gen.generate_pql` — LLM writes the PQL,
     the static lint + cheap parse validate it, an entity-selection SQL scopes
     the entities, and KumoRFM predicts, with the full repair loop.
  4. Format the result into the standard response dict.

Everything is bounded and defensive: any failure returns a graceful response
dict in the same shape as the SQL path, so the chat never hard-errors.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Bounds so building the graph on a large database stays tractable. The whole
# (capped) dataset is uploaded to the hosted KumoRFM service.
_MAX_TABLES = int(os.environ.get("KUMO_MAX_TABLES", "20"))
# Rows sampled per table into the graph. Unlimited by default; set
# KUMO_MAX_ROWS_PER_TABLE to a positive integer to cap it.
_raw_max_rows = os.environ.get("KUMO_MAX_ROWS_PER_TABLE")
_MAX_ROWS_PER_TABLE: int | None = (
    int(_raw_max_rows) if _raw_max_rows and int(_raw_max_rows) > 0 else None
)
_MAX_PREVIEW_ROWS = int(os.environ.get("KUMO_MAX_PREVIEW_ROWS", "50"))
_MAX_ENTITIES = int(os.environ.get("KUMO_MAX_ENTITIES", "2000"))

_init_lock = threading.Lock()
_initialized = False


def _ensure_init() -> None:
    """Authenticate the KumoRFM SDK once, from env vars."""
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        api_key = os.environ.get("KUMO_RFM_API_KEY")
        if not api_key:
            raise RuntimeError("KUMO_RFM_API_KEY is not set")
        url = os.environ.get("KUMO_RFM_API_URL") or None

        import kumoai.rfm as rfm

        before = time.perf_counter()
        rfm.init(url=url, api_key=api_key)
        _initialized = True
        logger.info(
            "KumoRFM initialized (url=%s) in %.2fs",
            url or "<default>",
            time.perf_counter() - before,
        )


def _quote(schema: str, table: str) -> str:
    """Schema-qualify a table name (double quotes work for Postgres/Snowflake)."""
    return f'"{schema}"."{table}"' if schema else f'"{table}"'


def _load_relevant_frames(
    connectors: list[Any],
    relevant_tables: list[dict[str, Any]],
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Load a bounded sample of each relevant table into a DataFrame.

    Iterates only the tables the candidate-preparation step already found relevant
    (no full-catalog scan). Each table's connector is resolved by ``database_name``,
    falling back to the first connector.

    Returns ``(frames, name_map)`` where ``name_map`` maps each graph table name
    to its schema-qualified SQL name (used to schema-qualify entity-selection SQL).
    """
    if not connectors or not relevant_tables:
        return {}, {}

    db_to_connector = {
        str(getattr(c, "database_name", "") or ""): c for c in connectors
    }
    default_connector = connectors[0]

    frames: dict[str, pd.DataFrame] = {}
    name_map: dict[str, str] = {}
    for t in relevant_tables:
        if len(frames) >= _MAX_TABLES:
            logger.warning(
                "kumo: reached table cap (%d); remaining tables skipped",
                _MAX_TABLES,
            )
            break
        table = str(t.get("name") or "").strip()
        if not table:
            continue
        schema = str(t.get("schema_name") or "").strip()
        connector = db_to_connector.get(
            str(t.get("database_name") or ""), default_connector
        )
        name = table if table not in frames else f"{schema}_{table}"
        limit = f" LIMIT {_MAX_ROWS_PER_TABLE}" if _MAX_ROWS_PER_TABLE else ""
        logger.info(
            "kumo: loading rows for %s.%s (limit=%s)...",
            schema,
            table,
            _MAX_ROWS_PER_TABLE or "none",
        )
        t_start = time.perf_counter()
        try:
            df = connector.execute(f"SELECT * FROM {_quote(schema, table)}{limit}")
        except Exception:
            logger.exception("kumo: failed to load rows for %s.%s", schema, table)
            continue
        elapsed = time.perf_counter() - t_start
        if df is None or df.empty:
            logger.info("kumo: %s.%s returned 0 rows in %.2fs", schema, table, elapsed)
            continue
        logger.info(
            "kumo: loaded %d row(s) x %d col(s) from %s.%s in %.2fs",
            len(df),
            len(df.columns),
            schema,
            table,
            elapsed,
        )
        frames[name] = df
        if schema:
            name_map[name] = _quote(schema, table)
    return frames, name_map


def _error_response(message: str) -> dict[str, Any]:
    return {
        "response": message,
        "sql_code": "",
        "sql_columns": [],
        "custom_analyses_used": [],
        "sql_response_from_db": None,
    }


def _json_safe(value: Any) -> Any:
    """Convert a prediction-frame cell into a JSON-serializable Python value.

    KumoRFM returns pandas ``Timestamp`` (e.g. ``ANCHOR_TIMESTAMP``) and numpy
    scalars (``float64`` / ``bool_`` / ``int64``) that ``json.dumps`` can't
    encode. Timestamps/datetimes become ISO strings, numpy scalars become native
    Python, and NaN/NaT become ``None``.
    """
    import datetime

    import numpy as np

    try:
        if not isinstance(value, (list, dict, tuple)) and pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_safe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: _json_safe(v) for k, v in row.items()} for row in rows]


def _format_result(result: Any) -> dict[str, Any]:
    """Shape a :class:`PqlGenerationResult` into the standard response dict."""
    if not result.success:
        message = result.error or "KumoRFM could not answer this prediction."
        final = _error_response(f"Prediction could not be completed. {message}")
        final["sql_code"] = result.pql or ""
        return final

    parts = [f"Prediction for: {result.question}"]
    if result.pql:
        parts.append(f"PQL: `{result.pql}`")
    if result.note:
        parts.append(result.note)
    parts.append(
        f"KumoRFM scored {result.num_entities} entit"
        f"{'y' if result.num_entities == 1 else 'ies'}."
        + (" (truncated preview)" if result.truncated else "")
    )
    return {
        "response": "\n\n".join(parts),
        "sql_code": result.pql or "",
        "sql_columns": result.columns,
        "custom_analyses_used": [],
        "sql_response_from_db": _json_safe_rows(result.rows) or None,
    }


@dataclass
class PredictionContext:
    """Everything :func:`run_prediction` needs, built once from the relevant tables.

    Holds live, non-serializable KumoRFM handles (``kumo_model``) plus the graph
    context strings. It is passed between the ``prepare_prediction_graph`` and
    ``kumo_predict`` graph nodes via ``path_state`` within a single run only — it
    is never checkpointed or serialized.
    """

    kumo_model: Any
    connector: Any
    graph_ddl: str
    graph_edges: Any
    graph_col_stypes: Any
    time_columns: Any
    table_names: dict[str, str]
    examples: list[dict[str, str]]


def _col_name_lower(col: Any) -> str | None:
    """Lowercased name of a KumoRFM primary-key/column object (``None`` if unset)."""
    name = getattr(col, "name", None)
    return name.lower() if name else None


def _resolve_column(table: Any, col: str) -> str | None:
    """Case-insensitive column-name match within a graph table (connectors lowercase)."""
    target = (col or "").lower()
    for c in table.columns:
        if c.name.lower() == target:
            return c.name
    return None


def _path_columns(entry: dict[str, Any]) -> list[tuple[str, str]]:
    """Flatten a join-path entry into its ``(table, column)`` node sequence.

    ``entry["path"]`` is a list of hop dicts ``{source_table, source_column,
    target_table, target_column, ...}`` that ``find_join_path`` produced by pairing
    consecutive columns of a Neo4j traversal. Flattening the hops back to
    ``[h0.source, h0.target, h1.source, h1.target, ...]`` restores that column
    sequence, so consecutive columns in DIFFERENT tables are the cross-table
    semantic foreign-key joins (columns within the same table are attribute hops).
    """
    seq: list[tuple[str, str]] = []
    for hop in entry.get("path") or []:
        seq.append(
            (str(hop.get("source_table") or ""), str(hop.get("source_column") or ""))
        )
        seq.append(
            (str(hop.get("target_table") or ""), str(hop.get("target_column") or ""))
        )
    return seq


def _apply_join_paths(graph: Any, join_paths: list[dict[str, Any]] | None) -> int:
    """Link graph tables using the cross-table joins in the catalog join paths.

    ``join_paths`` is ``attribute_join_paths`` from the text-to-SQL state. Each entry
    encodes a traversal from the anchor column to a destination column; the joins are
    the adjacent columns that cross tables (see :func:`_path_columns`). For each such
    join the side whose column is that table's primary key becomes the KumoRFM
    destination and the other side's column becomes the foreign key. Joins that don't
    map cleanly (unknown table/column, neither side a primary key, incompatible key
    dtype) are skipped.

    An edge already present in the graph (e.g. auto-inferred by ``from_data``) still
    counts as covered, so the caller doesn't fall back to heuristic inference for a
    relationship the catalog already describes.

    Returns the number of distinct cross-table joins covered (added or pre-existing).
    """
    lookup = {name.lower(): name for name in graph.tables}
    existing = {(e.src_table, e.fkey, e.dst_table) for e in graph.edges}
    covered = 0
    processed: set[tuple[str, str, str]] = set()
    for entry in join_paths or []:
        seq = _path_columns(entry)
        for (a_tbl, a_col_raw), (b_tbl, b_col_raw) in zip(seq, seq[1:]):
            a_name = lookup.get(a_tbl.lower())
            b_name = lookup.get(b_tbl.lower())
            if not a_name or not b_name or a_name == b_name:
                continue
            a_graph, b_graph = graph[a_name], graph[b_name]
            a_col = _resolve_column(a_graph, a_col_raw)
            b_col = _resolve_column(b_graph, b_col_raw)
            if not a_col or not b_col:
                continue
            # Orient the edge FK(src) -> PK(dst): the side whose join column is that
            # table's primary key is the destination.
            if _col_name_lower(b_graph.primary_key) == b_col.lower():
                src, fkey, dst = a_name, a_col, b_name
            elif _col_name_lower(a_graph.primary_key) == a_col.lower():
                src, fkey, dst = b_name, b_col, a_name
            else:
                continue
            key = (src, fkey, dst)
            if key in processed:
                continue
            processed.add(key)
            if key in existing:
                covered += 1
                continue
            try:
                graph.link(src, fkey, dst)
            except Exception:
                logger.debug(
                    "kumo: skipped join-path link %s.%s -> %s",
                    src,
                    fkey,
                    dst,
                    exc_info=True,
                )
                continue
            covered += 1
    return covered


def build_prediction_context(
    connectors: list[Any],
    relevant_tables: list[dict[str, Any]] | None = None,
    join_paths: list[dict[str, Any]] | None = None,
    examples: list[dict[str, str]] | None = None,
) -> PredictionContext | dict[str, Any]:
    """Build the KumoRFM graph + model scoped to the relevant tables.

    ``relevant_tables`` (as produced by the candidate-preparation step) scopes the
    KumoRFM graph to the tables relevant to the question. ``join_paths``
    (``attribute_join_paths``) from the text-to-SQL state supplies the table
    relationships: its catalog-derived joins are used as the graph's edges, and
    KumoRFM's own heuristic ``infer_links`` is used only as a fallback when no usable
    join path is available. ``examples`` are verified ``{question, query}`` PQL
    few-shots carried into generation. Returns a :class:`PredictionContext` on
    success, or a graceful error response dict when there is nothing to build a graph
    from.
    """
    logger.info("kumo: build_prediction_context start (initializing KumoRFM)")
    _ensure_init()

    import kumoai.rfm as rfm

    from gsf.retrieval.kumo.kumo_model import KumoModel, build_graph_context

    if not connectors:
        return _error_response("No database connection is configured.")

    logger.info(
        "kumo: loading sample rows for %d relevant table(s)...",
        len(relevant_tables or []),
    )
    _load_start = time.perf_counter()
    frames, name_map = _load_relevant_frames(connectors, relevant_tables or [])
    if not frames:
        return _error_response(
            "No relevant tables were available to build a prediction graph."
        )
    logger.info(
        "kumo: loaded %d frame(s) in %.2fs; building graph from %d table(s)",
        len(frames),
        time.perf_counter() - _load_start,
        len(frames),
    )

    _graph_start = time.perf_counter()
    graph = rfm.LocalGraph.from_data(frames, infer_metadata=True, verbose=False)
    logger.info(
        "kumo: LocalGraph.from_data (metadata inferred) in %.2fs",
        time.perf_counter() - _graph_start,
    )
    covered = _apply_join_paths(graph, join_paths)
    if covered:
        logger.info("kumo: using %d catalog join edge(s)", covered)
    else:
        # No usable catalog join paths — fall back to KumoRFM's link heuristics.
        logger.info("kumo: no catalog join paths; inferring links heuristically")
        try:
            graph.infer_links()
        except Exception:
            logger.exception(
                "kumo: infer_links failed; proceeding without inferred links"
            )

    graph_ddl, edges, col_stypes, time_columns = build_graph_context(graph)
    kumo_model = KumoModel(rfm.KumoRFM(graph, verbose=False))

    # Entity-selection SQL runs against the live GSF database connection (the
    # first configured connector — the source of the catalog tables). ``table_names``
    # maps bare graph table names to their schema-qualified form so the SQL resolves.
    return PredictionContext(
        kumo_model=kumo_model,
        connector=connectors[0],
        graph_ddl=graph_ddl,
        graph_edges=edges,
        graph_col_stypes=col_stypes,
        time_columns=time_columns,
        table_names=name_map,
        examples=examples or [],
    )


def run_prediction(
    question: str, llm: Any, context: PredictionContext
) -> dict[str, Any]:
    """Generate + repair the PQL, predict, and format — given a prepared context."""
    from gsf.retrieval.kumo.pql_gen import generate_pql

    result = generate_pql(
        question,
        llm=llm,
        kumo_model=context.kumo_model,
        connector=context.connector,
        graph_ddl=context.graph_ddl,
        graph_edges=context.graph_edges,
        graph_col_stypes=context.graph_col_stypes,
        time_columns=context.time_columns,
        table_names=context.table_names,
        max_entities=_MAX_ENTITIES,
        max_preview_rows=_MAX_PREVIEW_ROWS,
        examples=context.examples,
    )

    return _format_result(result)
