# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relational prediction pipeline (Auto Ontology entry point).

Wires the ingested-catalog data into the ported text-to-PQL pipeline:
  1. Load a bounded sample of each ingested-catalog table into DataFrames.
  2. Build a relational ``Graph`` (metadata + links inferred) and a DuckDB
     mirror of the same frames (so the entity-selection SQL resolves).
  3. Run :func:`auto_ontology.retrieval.kumo.pql_gen.generate_pql` — LLM writes the PQL,
     the static lint + cheap parse validate it, an entity-selection SQL scopes
     the entities, and KumoRFM predicts, with the full repair loop.
  4. Format the result into the standard response dict.

Everything is bounded and defensive: any failure returns a graceful response
dict in the same shape as the SQL path, so the chat never hard-errors.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from auto_ontology.retrieval.kumo.budget import (
    Budget,
    RefusedForCapacity,
    Spend,
)
from auto_ontology.retrieval.kumo.column_reference import build_column_reference
from auto_ontology.retrieval.kumo.graph_cache import (
    BuildTimedOut,
    CacheKey,
    GraphCache,
    built_graph_fingerprint,
    catalog_fingerprint,
    join_fingerprint,
)
from auto_ontology.retrieval.kumo.pql_gen import quote_ident
from auto_ontology.retrieval.kumo.kumo_model import key_columns
from auto_ontology.retrieval.kumo.telemetry import (
    CACHE_DISABLED,
    CACHE_HIT,
    CACHE_MISS,
    GraphIdentity,
    RunRecord,
    emit,
    llm_model_name,
    redact_error,
    redact_literals,
)

logger = logging.getLogger(__name__)


class TableUnavailable(RefusedForCapacity):
    """A table the question needs could not be read.

    Distinct from a table that read back empty, which is an answer. A read that
    failed leaves the graph missing a table the question was scoped to, and a
    prediction built from what remains answers a narrower question without
    saying so.
    """


# Bounds so building the graph on a large database stays tractable. The whole
# (capped) dataset is uploaded to the hosted KumoRFM service.
_raw_max_rows = os.environ.get("KUMO_MAX_ROWS_PER_TABLE")
_MAX_ROWS_PER_TABLE: int | None = (
    int(_raw_max_rows) if _raw_max_rows and int(_raw_max_rows) > 0 else None
)
_MAX_PREVIEW_ROWS = int(os.environ.get("KUMO_MAX_PREVIEW_ROWS", "50"))
_MAX_ENTITIES = int(os.environ.get("KUMO_MAX_ENTITIES", "2000"))

_GRAPH_CACHE = GraphCache.from_env()
PROMPT_VERSION = "1"

_init_lock = threading.Lock()
_initialized = False

_client: Any = None


def _ensure_init() -> Any:
    """Open the RelationalClient once, from env vars, and return it.

    The client is the only supported entry point to the engine, which refuses
    direct use. Opening it makes no request, so a bad URL surfaces on the first
    prediction rather than here.
    """
    global _client
    if _client is not None:
        return _client
    with _init_lock:
        if _client is not None:
            return _client
        url = os.environ.get("KUMO_RFM_API_URL")
        if not url:
            raise RuntimeError("KUMO_RFM_API_URL is not set")
        api_key = os.environ.get("KUMO_RFM_API_KEY") or None

        from kumo_relational_client import RelationalClient

        before = time.perf_counter()
        _client = RelationalClient(url, api_key=api_key)
        logger.info(
            "Relational client opened (url=%s) in %.2fs",
            url,
            time.perf_counter() - before,
        )
        return _client


def _quote(schema: str, table: str) -> str:
    """Schema-qualify a table name (double quotes work for Postgres/Snowflake)."""
    return f'"{schema}"."{table}"' if schema else f'"{table}"'


def _catalog_key_columns(entry: dict[str, Any]) -> list[str]:
    """Primary-key columns the catalog recorded for a table.

    Every path spells the field ``pk``, the catalog's own name for it — both
    retrieval (see ``relevant_tables``) and the PQL-example enrichment. It may
    hold a single name or several for a composite key.
    """
    raw = entry.get("pk")
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    if isinstance(raw, (list, tuple)):
        return [str(c).strip() for c in raw if str(c or "").strip()]
    return []


def _projection(table: dict[str, Any]) -> str:
    """The columns to read, or ``*`` when the catalog does not name them.

    Reading every column costs memory and warehouse time for data no prediction
    can reach: the graph is built from the columns the catalog knows, and a
    column absent from it is invisible to the model whether or not it was read.

    Falls back to ``*`` rather than guessing. A projection that misses a column
    the graph needs would build a different graph, which is worse than reading
    more than necessary.
    """
    names = [
        str(column.get("name")).strip()
        for column in table.get("columns") or []
        if isinstance(column, dict) and str(column.get("name") or "").strip()
    ]
    if not names:
        return "*"
    return ", ".join(quote_ident(name) for name in names)


def _load_relevant_frames(
    connectors: list[Any],
    relevant_tables: list[dict[str, Any]],
    spend: Spend,
) -> tuple[dict[str, pd.DataFrame], dict[str, str], dict[str, list[str]]]:
    """Load a bounded sample of each relevant table into a DataFrame.

    Iterates only the tables the candidate-preparation step already found relevant
    (no full-catalog scan). Each table's connector is resolved by ``database_name``,
    falling back to the first connector.

    Returns ``(frames, name_map, key_columns)`` where ``name_map`` maps each graph
    table name to its schema-qualified SQL name (used to schema-qualify
    entity-selection SQL) and ``key_columns`` maps it to the catalog's primary-key
    columns (see :func:`_declare_primary_keys`). Both are keyed by the graph table
    name chosen here, so neither has to re-derive it.
    """
    if not connectors or not relevant_tables:
        return {}, {}, {}

    db_to_connector = {
        str(getattr(c, "database_name", "") or ""): c for c in connectors
    }
    default_connector = connectors[0]

    frames: dict[str, pd.DataFrame] = {}
    name_map: dict[str, str] = {}
    key_columns: dict[str, list[str]] = {}
    for t in relevant_tables:
        spend.check_deadline("reading tables")
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
            df = connector.execute(
                f"SELECT {_projection(t)} FROM {_quote(schema, table)}{limit}"
            )
        except Exception as error:
            logger.exception("kumo: failed to load rows for %s.%s", schema, table)
            raise TableUnavailable(
                f"{schema}.{table} could not be read, so a prediction over it "
                f"would answer from the tables that did load and report a total "
                f"that reads like the whole. ({type(error).__name__})"
            ) from error
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
        spend.add_table(name, df)
        frames[name] = df
        if schema:
            name_map[name] = _quote(schema, table)
        catalog_keys = _catalog_key_columns(t)
        if catalog_keys:
            key_columns[name] = catalog_keys
    return frames, name_map, key_columns


def _refused(message: str, identity: "GraphIdentity | None" = None) -> dict[str, Any]:
    """Report a request that never became a prediction, and record that it did not.

    A refusal during preparation exits before ``run_prediction``, so without
    this the only requests that leave a trace are the ones that got far enough
    to ask the model. A question refused for cost or for a table that would not
    read is exactly the kind someone asks about later.
    """
    emit(
        RunRecord(
            outcome="refused",
            error=redact_error(message),
            graph=identity or GraphIdentity(),
        )
    )
    return _error_response(message)


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
    if result.truncated and result.population > result.num_entities:
        parts.append(
            f"KumoRFM scored {result.num_entities:,} of {result.population:,} "
            f"entities. These results cover only those scored, not the whole "
            f"population."
        )
    else:
        parts.append(
            f"KumoRFM scored {result.num_entities:,} entit"
            f"{'y' if result.num_entities == 1 else 'ies'}."
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
    # Per table (casefolded), the identity of every loaded row: bare values for a
    # single-column key, one tuple per row for a composite one.
    entity_ids: dict[str, list[Any]]
    examples: list[dict[str, str]]
    column_reference: str
    identity: GraphIdentity


def _fkey_name(fkey: Any) -> str:
    """Comparable text for an edge's foreign key.

    A composite key reads back as the single surrogate column rather than the tuple
    that declared it, so this is normally already a string; the tuple branch keeps
    the comparison total if a later SDK reports the columns themselves.
    """
    return ", ".join(str(c) for c in fkey) if isinstance(fkey, tuple) else str(fkey)


def _resolve_column(table: Any, col: str) -> str | None:
    """Case-insensitive column-name match within a graph table (connectors lowercase)."""
    target = (col or "").lower()
    for c in table.columns:
        if c.name.lower() == target:
            return c.name
    return None


def _declare_primary_keys(
    graph: Any, key_columns_by_table: dict[str, list[str]]
) -> int:
    """Set each table's primary key from the catalog, as a tuple when composite.

    Metadata inference picks no key at all when several columns are ``stype=ID`` and
    none resembles the table name, and it never infers a composite key. A table keyed
    on ``('Customer ID', 'REGION')`` would therefore reach KumoRFM with no identity,
    which costs it every edge (nothing can be oriented towards a key that isn't
    there) and leaves it unusable as a prediction entity.

    Declaring only what the catalog actually recorded: a table whose key columns are
    absent from the loaded frame is left to inference. Returns the number of tables
    whose key was declared.
    """
    declared = 0
    for name, table in graph.tables.items():
        wanted = key_columns_by_table.get(name) or []
        if not wanted:
            continue
        resolved = [c for c in (_resolve_column(table, w) for w in wanted) if c]
        if len(resolved) != len(wanted):
            logger.debug(
                "kumo: catalog key %s not fully present in %s; leaving inference",
                wanted,
                name,
            )
            continue
        if [c.lower() for c in key_columns(table)] == [c.lower() for c in resolved]:
            continue
        try:
            table.primary_key = resolved[0] if len(resolved) == 1 else tuple(resolved)
        except Exception:
            logger.debug(
                "kumo: could not declare key %s on %s", resolved, name, exc_info=True
            )
            continue
        declared += 1
        logger.info("kumo: declared %s primary key %s from catalog", name, resolved)
    return declared


def _path_columns(entry: dict[str, Any]) -> list[tuple[str, str]]:
    """Flatten a join-path entry into its ``(table, column)`` node sequence.

    ``entry["path"]`` is a list of hop dicts ``{source_table, source_column,
    target_table, target_column, ...}`` that ``find_join_path`` produced by pairing
    consecutive columns of a join-path traversal. Flattening the hops back to
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

    A composite destination key is linked once with the whole tuple: KumoRFM rejects
    a link that names only part of an identity, and each part arrives here as its own
    single-column join, so the parts are collected per table pair before linking. A
    part the join paths never mention is taken from the source's same-named column
    when it has one, which is what a sharded warehouse looks like (``ORDERS`` and
    ``PEOPLE`` both carrying ``REGION``).

    Returns the number of distinct relationships covered (added or pre-existing).
    """
    lookup = {name.lower(): name for name in graph.tables}
    existing = {(e.src_table, e.fkey, e.dst_table) for e in graph.edges}
    existing_pairs = {(e.src_table, e.dst_table) for e in graph.edges}
    # (src, dst) -> {destination key column (lowercased): source column}
    pending: dict[tuple[str, str], dict[str, str]] = {}
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
            # Orient the edge FK(src) -> PK(dst): the side whose join column is part
            # of that table's primary key is the destination.
            a_keys = {c.lower() for c in key_columns(a_graph)}
            b_keys = {c.lower() for c in key_columns(b_graph)}
            if b_col.lower() in b_keys:
                src, src_col, dst, dst_col = a_name, a_col, b_name, b_col
            elif a_col.lower() in a_keys:
                src, src_col, dst, dst_col = b_name, b_col, a_name, a_col
            else:
                continue
            pending.setdefault((src, dst), {}).setdefault(dst_col.lower(), src_col)

    covered = 0
    for (src, dst), mapping in pending.items():
        dst_keys = key_columns(graph[dst])
        fkey: Any
        if len(dst_keys) > 1:
            parts = [
                mapping.get(k.lower()) or _resolve_column(graph[src], k)
                for k in dst_keys
            ]
            if not all(parts):
                logger.debug(
                    "kumo: %s cannot reference all of %s's identity %s; skipped",
                    src,
                    dst,
                    dst_keys,
                )
                continue
            fkey = tuple(parts)
        else:
            fkey = next(iter(mapping.values()))
        if (src, fkey, dst) in existing or (
            isinstance(fkey, tuple) and (src, dst) in existing_pairs
        ):
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


def _deduplicate_inferred_links(graph: Any) -> int:
    """Keep one deterministic FK when inference links the same table pair twice.

    KumoRFM cannot disambiguate an aggregation when a child table has multiple
    foreign keys to the same parent (for example ``job_id`` and
    ``restart_of_job_id``). Prefer the foreign key whose name exactly matches
    the destination primary key, then the shortest/lexicographically first
    name. Catalog-derived join paths do not use this fallback because they
    already scope the graph to the relationship relevant to the question.
    """
    grouped: dict[tuple[str, str], list[Any]] = {}
    for edge in graph.edges:
        grouped.setdefault((edge.src_table, edge.dst_table), []).append(edge)

    removed = 0
    for (_src, dst), edges in grouped.items():
        if len(edges) < 2:
            continue
        # Only a single-column key gives a name worth matching against; a composite
        # one reads back as the surrogate column on both ends, so such edges fall
        # through to the shortest/lexicographically-first tie-break.
        dst_keys = key_columns(graph[dst])
        primary_key = dst_keys[0].lower() if len(dst_keys) == 1 else ""
        keep = min(
            edges,
            key=lambda edge: (
                _fkey_name(edge.fkey).lower() != primary_key,
                len(_fkey_name(edge.fkey)),
                _fkey_name(edge.fkey).lower(),
            ),
        )
        for edge in edges:
            if edge == keep:
                continue
            graph.unlink(edge.src_table, edge.fkey, edge.dst_table)
            removed += 1
            logger.info(
                "kumo: removed ambiguous inferred link %s.%s -> %s "
                "(keeping %s.%s -> %s)",
                edge.src_table,
                edge.fkey,
                edge.dst_table,
                keep.src_table,
                keep.fkey,
                keep.dst_table,
            )
    return removed


def _entity_ids(graph: Any, frames: dict[str, pd.DataFrame]) -> dict[str, list[Any]]:
    """Identity of every loaded row, per table, for scoping a prediction.

    A composite identity is one tuple per row: naming a single one of its columns
    picks out no row. The surrogate column KumoRFM adds for such a key is absent from
    the frame, so the real key columns are read instead.
    """
    entity_ids: dict[str, list[Any]] = {}
    for name, table in graph.tables.items():
        keys = key_columns(table)
        frame = frames.get(name)
        if not keys or frame is None or any(k not in frame.columns for k in keys):
            continue
        rows = frame[keys].dropna().drop_duplicates()
        entity_ids[name.casefold()] = (
            rows[keys[0]].tolist()
            if len(keys) == 1
            else [tuple(r) for r in rows.to_numpy().tolist()]
        )
    return entity_ids


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
    logger.info("kumo: build_prediction_context start")

    if not connectors:
        return _refused("No database connection is configured.")

    key = _cache_key(connectors, relevant_tables or [], join_paths)
    if key is not None:
        try:
            cached, built_here = _GRAPH_CACHE.get_or_build(
                key,
                lambda: _build_context(connectors, relevant_tables, join_paths, []),
                worth_keeping=_worth_keeping,
            )
        except BuildTimedOut as waited:
            logger.info("kumo: gave up waiting for a build in progress: %s", waited)
            return _refused(str(waited))
        if isinstance(cached, PredictionContext):
            # The key already says this request's connector reaches the same
            # database, so use it: the one the graph was built with belongs to
            # an earlier request and may since have been closed.
            return dataclasses.replace(
                cached,
                connector=connectors[0],
                examples=examples or [],
                identity=dataclasses.replace(
                    cached.identity,
                    cache=CACHE_MISS if built_here else CACHE_HIT,
                ),
            )
        return cached

    return _build_context(connectors, relevant_tables, join_paths, examples)


def _worth_keeping(built: Any) -> bool:
    """Whether a build is worth holding for the requests that follow it.

    A failure is not: it would be served to them as a hit and would count
    against the entry bound, evicting a graph that worked to make room for one
    that did not.

    Nor is a graph whose edges the engine GUESSED. The catalog said nothing
    about those relationships, so inference chose them from names and values,
    and a second request over the same tables can be given a different shape.
    Holding the first one freezes whichever build won the race for the whole
    TTL. Rebuilding is the cheaper mistake.
    """
    if not isinstance(built, PredictionContext):
        return False
    return built.identity.edges_from != "inferred"


def _cache_key(
    connectors: list[Any],
    relevant_tables: list[dict[str, Any]],
    join_paths: list[dict[str, Any]] | None,
) -> "CacheKey | None":
    """What two requests must share before one may reuse the other's graph.

    None when the tables name no database, or when the connectors cannot say
    which physical source they read: an entry that could not say where its data
    came from would be shared with a question about different data.
    """
    databases = {
        str(t.get("database_name") or "")
        for t in relevant_tables
        if isinstance(t, dict)
    }
    if not relevant_tables or len(databases) != 1 or not next(iter(databases)):
        return None
    connector = _connector_identity(connectors)
    if connector is None:
        return None
    import kumo_relational_client

    return CacheKey(
        connector=connector,
        database=next(iter(databases)),
        tables=tuple(
            sorted(
                str(t.get("name") or "") for t in relevant_tables if isinstance(t, dict)
            )
        ),
        schema_fingerprint=catalog_fingerprint(relevant_tables),
        join_fingerprint=join_fingerprint(join_paths),
        prompt_version=PROMPT_VERSION,
        engine_version=str(getattr(kumo_relational_client, "__version__", "")),
    )


_CONNECTION_ATTRIBUTES = (
    "_connection_string",
    "_connect_kwargs",
    "_settings",
    "_warehouse",
    "_host",
    "_account",
    "_catalog",
    "_http_path",
    "_db_path",
)


def _connector_identity(connectors: list[Any]) -> str | None:
    """Which physical sources a request reads from, or ``None`` if unknowable.

    Two deployments can describe the same catalog while pointing at different
    warehouses, accounts, or files, so a key naming only the database would
    answer one from the other's data. Two DuckDB files both named ``sales``
    report the same ``database_name``.

    Returns ``None`` when a connector exposes nothing that distinguishes its
    source, which disables caching for that request rather than serving it from
    another deployment's graph. A connector added later inherits that refusal
    until it is taught to say where it points.

    Hashed rather than kept, because what distinguishes two connections is also
    what authenticates them: a connection string carries a password, and a cache
    key is written to logs.
    """
    identities = []
    for connector in connectors:
        distinguishing = [
            repr(getattr(connector, attribute))
            for attribute in _CONNECTION_ATTRIBUTES
            if getattr(connector, attribute, None) is not None
        ]
        if not distinguishing:
            logger.debug(
                "Not caching: %s exposes no connection identity.",
                type(connector).__name__,
            )
            return None
        terms = [type(connector).__name__, str(getattr(connector, "database_name", ""))]
        identities.append("\x1f".join(terms + distinguishing))
    digest = hashlib.sha256("\x1e".join(sorted(identities)).encode("utf-8"))
    return digest.hexdigest()


def _build_context(
    connectors: list[Any],
    relevant_tables: list[dict[str, Any]] | None,
    join_paths: list[dict[str, Any]] | None,
    examples: list[dict[str, str]] | None,
) -> "PredictionContext | dict[str, Any]":
    """Read the tables and build the graph and model over them.

    Every way this can be refused for what it would cost is answered here, so
    a refusal raised by a later phase reaches the asker as an answer rather
    than escaping as an exception.
    """
    try:
        return _build_context_within_budget(
            connectors, relevant_tables, join_paths, examples
        )
    except RefusedForCapacity as refusal:
        logger.info("kumo: refused, %s", refusal)
        return _refused(str(refusal))


def _build_context_within_budget(
    connectors: list[Any],
    relevant_tables: list[dict[str, Any]] | None,
    join_paths: list[dict[str, Any]] | None,
    examples: list[dict[str, str]] | None,
) -> "PredictionContext | dict[str, Any]":
    """Read the tables and build the graph and model over them."""
    import kumo_relational_client
    from kumo_relational_client import relational as rfm

    from auto_ontology.retrieval.kumo.kumo_model import KumoModel, build_graph_context

    client = _ensure_init()

    logger.info(
        "kumo: loading sample rows for %d relevant table(s)...",
        len(relevant_tables or []),
    )
    _load_start = time.perf_counter()
    spend = Spend(Budget.from_env())
    frames, name_map, catalog_keys = _load_relevant_frames(
        connectors, relevant_tables or [], spend
    )
    if not frames:
        return _refused(
            "No relevant tables were available to build a prediction graph."
        )
    logger.info(
        "kumo: loaded %d frame(s) in %.2fs; building graph from %d table(s)",
        len(frames),
        time.perf_counter() - _load_start,
        len(frames),
    )

    _graph_start = time.perf_counter()
    # Passing an explicit empty edge list suppresses LocalGraph's automatic
    # relationship inference. This lets catalog join paths take precedence and
    # avoids inferring the same links twice.
    graph = rfm.Graph.from_data(
        frames,
        edges=[],
        infer_metadata=True,
        verbose=False,
    )
    logger.info(
        "kumo: LocalGraph.from_data (metadata inferred) in %.2fs",
        time.perf_counter() - _graph_start,
    )
    # Before any linking: an edge is oriented towards a primary key, so a table whose
    # key inference missed can take part in no relationship at all.
    spend.check_deadline("declaring keys")
    _declare_primary_keys(graph, catalog_keys)
    covered = _apply_join_paths(graph, join_paths)
    if covered:
        logger.info("kumo: using %d catalog join edge(s)", covered)
    else:
        # No usable catalog join paths — fall back to KumoRFM's link heuristics.
        logger.info("kumo: no catalog join paths; inferring links heuristically")
        try:
            graph.infer_links()
            _deduplicate_inferred_links(graph)
        except Exception:
            logger.exception(
                "kumo: infer_links failed; proceeding without inferred links"
            )

    spend.check_deadline("building the graph")
    graph_ddl, edges, col_stypes, time_columns = build_graph_context(graph)
    spend.check_deadline("creating the model")
    kumo_model = KumoModel(client.relational(graph), graph)
    entity_ids = _entity_ids(graph, frames)

    # Entity-selection SQL runs against the live Auto Ontology database connection (the
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
        entity_ids=entity_ids,
        examples=examples or [],
        column_reference=build_column_reference(relevant_tables or [], col_stypes),
        identity=GraphIdentity(
            fingerprint=built_graph_fingerprint(graph),
            engine_version=str(getattr(kumo_relational_client, "__version__", "")),
            prompt_version=PROMPT_VERSION,
            tables=len(frames),
            rows=int(sum(len(frame) for frame in frames.values())),
            bytes=spend.nbytes,
            edges=len(edges),
            edges_from="catalog" if covered else "inferred",
            build_seconds=round(time.perf_counter() - _load_start, 3),
            cache=CACHE_DISABLED,
        ),
    )


def run_prediction(
    question: str, llm: Any, context: PredictionContext
) -> dict[str, Any]:
    """Generate + repair the PQL, predict, and format — given a prepared context."""
    from auto_ontology.retrieval.kumo.pql_gen import generate_pql

    started = time.perf_counter()
    record = RunRecord(
        outcome="failed",
        llm_model=llm_model_name(llm),
        graph=context.identity,
    )
    try:
        return _run_prediction(question, llm, context, generate_pql, record)
    except Exception as error:
        record.error = redact_error(f"{type(error).__name__}: {error}")
        raise
    finally:
        record.seconds = round(time.perf_counter() - started, 3)
        emit(record)


def _run_prediction(
    question: str,
    llm: Any,
    context: PredictionContext,
    generate_pql: Any,
    record: RunRecord,
) -> dict[str, Any]:
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
        available_entity_ids=context.entity_ids,
        max_entities=_MAX_ENTITIES,
        max_preview_rows=_MAX_PREVIEW_ROWS,
        examples=context.examples,
        column_reference=context.column_reference,
    )

    record.pql = redact_literals(getattr(result, "pql", "") or "")
    record.attempts = int(getattr(result, "attempts", 0) or 0)
    record.entities = int(getattr(result, "num_entities", 0) or 0)
    record.rows_returned = len(getattr(result, "rows", []) or [])
    record.error = redact_error(str(getattr(result, "error", "") or ""))
    record.outcome = "answered" if getattr(result, "success", False) else "refused"

    return _format_result(result)
