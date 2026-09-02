"""Resolve SEMANTIC_FK edges after taxonomy compilation.

Algorithm
---------
1. Find every Column node that has no SEMANTIC_FK edge and no HAS_ATTRIBUTE
   edge (i.e. a FK column that the taxonomy pass skipped).
2. For each such column:
   a. **Declared FK fast path** — if a FOREIGN_KEY edge already exists in the
      graph, look up the ColumnAttribute attached to the target Column and
      create SEMANTIC_FK directly. A declared FK that targets a column on the
      *same* table is rejected — that's a self-reference, never the entity
      the FK points to.
   b. **LLM / VDB fallback** — if there is no declared FK, embed the column
      context, search the *semantic* VDB for the top-5 most similar
      ColumnAttribute records (excluding candidates on the FK's own table),
      ask the LLM to pick the best match, and fall back to probing the live
      database for a candidate whose sample values match when the LLM
      abstains. The hit's ``metadata["id"]`` is the ColumnAttribute id
      directly — no additional lookup needed.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from nemo_retriever.graph.retriever import Retriever
from nemo_retriever.tabular_data.sql_database import SQLDatabase
from sqlglot import exp

from gsf.connectors import get_connectors
from gsf.dal.attributes import (
    find_column_attribute_by_column_id,
    find_unlinked_fk_columns,
    merge_semantic_fk,
)
from gsf.retrieval.text_to_sql.db_probe.executor import ProbeExecutor
from gsf.utils.llm_invoke import (
    get_non_reasoning_llm_client,
    invoke_with_structured_output,
)
from gsf.utils.model_config import resolve
from gsf.utils.sample_values import parse_sample_values
from gsf.semantic.models import FkHitSelection
from gsf.vdb import get_semantic_vdb

logger = logging.getLogger(__name__)

_EMBED_ENDPOINT = resolve("EMBED", "ENDPOINT")
_EMBED_MODEL = resolve("EMBED", "MODEL")
_NVIDIA_API_KEY = resolve("EMBED", "API_KEY")
_WORKERS = 2
_SYSTEM_PROMPT = """\
You are a database schema expert. You will be given a foreign-key column \
description and a list of candidate primary-key columns retrieved from a \
vector search. Each candidate is prefixed with its unique column_id. \
Your task is to decide which candidate (if any) is the column that the \
foreign key references.

Rules:
- Return the exact column_id of the best matching candidate, or null if none fit.
- Only pick a candidate when you are confident it is the PK being referenced.
- Do NOT guess. If unsure, return null.
"""


def resolve_semantic_fks(database_name: str) -> int:
    """Create SEMANTIC_FK edges for all unlinked FK columns.

    Runs after the taxonomy while-loop in ``compile_semantic_layer``.
    Returns the total number of SEMANTIC_FK edges created.
    """
    candidates = find_unlinked_fk_columns(database_name)
    if not candidates:
        logger.info("resolve_semantic_fks: no unlinked FK columns found")
        return 0

    logger.info(
        "resolve_semantic_fks: %d candidate column(s) to process", len(candidates)
    )

    declared_written = 0
    total_candidates = len(candidates)
    llm_queue: list[tuple[int, dict[str, Any]]] = []

    def _log_start(index: int, col: dict[str, Any]) -> None:
        logger.info(
            "Resolving semantic FK edges… (%d/%d): %s.%s",
            index,
            total_candidates,
            col.get("table_name"),
            col.get("name"),
        )

    for index, col in enumerate(candidates, start=1):
        fk_target_col_id: str | None = col.get("fk_target_col_id")
        if fk_target_col_id:
            if col.get("fk_target_table_id") == col.get("table_id"):
                _log_start(index, col)
                logger.debug(
                    "resolve_semantic_fks [declared]: %s.%s declared FK targets its "
                    "own table — rejecting self-reference",
                    col.get("table_name"),
                    col.get("name"),
                )
                continue
            attr_id = find_column_attribute_by_column_id(fk_target_col_id)
            if attr_id:
                _log_start(index, col)
                merge_semantic_fk(col["id"], attr_id)
                declared_written += 1
                logger.debug(
                    "resolve_semantic_fks [declared]: %s.%s → attr %s",
                    col.get("table_name"),
                    col.get("name"),
                    attr_id,
                )
            else:
                logger.debug(
                    "resolve_semantic_fks [declared]: target column %s has no ColumnAttribute — queuing for LLM",
                    fk_target_col_id,
                )
                llm_queue.append((index, col))
        else:
            llm_queue.append((index, col))

    logger.info(
        "resolve_semantic_fks: %d declared FK(s) resolved; %d queued for inference",
        declared_written,
        len(llm_queue),
    )

    if not llm_queue:
        return declared_written

    retriever = _build_retriever(database_name)
    if retriever is None:
        logger.warning(
            "resolve_semantic_fks: EMBED_API_KEY not set — skipping LLM/VDB path "
            "for %d column(s)",
            len(llm_queue),
        )
        return declared_written
    connector = _resolve_connector(database_name)

    inferred_written = 0

    def _resolve_one(index: int, col: dict[str, Any]) -> bool:
        _log_start(index, col)
        try:
            attr_id = _resolve_via_vdb(col, retriever, database_name, connector)
            if attr_id:
                merge_semantic_fk(col["id"], attr_id)
                logger.debug(
                    "resolve_semantic_fks [resolved]: %s.%s → attr %s",
                    col.get("table_name"),
                    col.get("name"),
                    attr_id,
                )
                return True
        except Exception:
            logger.exception(
                "resolve_semantic_fks [llm]: unexpected error for column %s",
                col.get("id"),
            )
        return False

    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        futures = {
            pool.submit(_resolve_one, index, col): col for index, col in llm_queue
        }
        for future in as_completed(futures):
            if future.result():
                inferred_written += 1

    total = declared_written + inferred_written
    logger.info(
        "resolve_semantic_fks: done — %d declared + %d inferred = %d total "
        "SEMANTIC_FK edge(s)",
        declared_written,
        inferred_written,
        total,
    )
    return total


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_connector(database_name: str) -> SQLDatabase | None:
    """Return the configured connector matching *database_name*."""
    target = database_name.casefold()
    for connector in get_connectors():
        name = getattr(connector, "database_name", None)
        if isinstance(name, str) and name.casefold() == target:
            return connector
    return None


def _build_retriever(database_name: str) -> Retriever | None:
    """Build a Retriever backed by the semantic VDB, or None when the API key is absent."""
    if not _NVIDIA_API_KEY:
        return None

    vdb = get_semantic_vdb(database_name=database_name)
    return Retriever(
        vdb_kwargs={"vdb": vdb},
        embed_kwargs={
            "model_name": _EMBED_MODEL,
            "embed_invoke_url": _EMBED_ENDPOINT,
            "api_key": _NVIDIA_API_KEY,
        },
    )


def _build_query_text(col: dict[str, Any]) -> str:
    """Build an embedding query string from a column's context."""
    parts: list[str] = [f"column_name: {col.get('name', '')}"]
    desc = (col.get("description") or "").strip()
    if desc:
        parts.append(f"description: {desc}")
    sample_str = _format_sample_values(col.get("sample_values"))
    if sample_str:
        parts.append(sample_str)
    return ", ".join(parts)


def _format_sample_values(raw: str | None) -> str:
    """Return a 'sample_values: ...' string, or empty when unavailable."""
    if not raw:
        return ""
    try:
        import json

        values = json.loads(raw)
        non_null = [str(v) for v in values if v is not None and len(str(v)) <= 30]
        if not non_null:
            return ""
        return "sample_values: " + ", ".join(non_null)
    except Exception:
        return ""


def _resolve_via_vdb(
    col: dict[str, Any],
    retriever: Retriever,
    database_name: str,
    connector: SQLDatabase | None = None,
) -> str | None:
    """Search the semantic VDB for a matching ColumnAttribute and return its id.

    Runs two queries and merges up to 6 unique hits:
    - name + description query (top 3): captures semantic context
    - name-only query (top 3): catches cases where description is noisy or absent

    Candidates on the FK column's own table are excluded — a FK never
    references a column on the table it lives on. Candidates whose owning
    column is known (profiled) to be non-unique are excluded at the VDB
    query level — a non-unique column can't validly serve as the referenced
    key.

    The hit's ``metadata["id"]`` is the ColumnAttribute id directly —
    no additional graph lookup is required.

    Returns the ColumnAttribute id from a confident LLM match, or — when the
    LLM abstains — from probing the live database for a candidate whose
    physical column actually contains the FK's sample values. None if
    neither step finds a match.
    """
    vdb_kwargs = {
        "where": {
            "label": "ColumnAttribute",
            "database_name": database_name,
            "is_unique": True,
        }
    }
    full_query = _build_query_text(col)
    name_query = f"column_name: {col.get('name', '')}"

    full_hits = retriever.query(full_query, top_k=3, vdb_kwargs=vdb_kwargs) or []
    name_hits = retriever.query(name_query, top_k=3, vdb_kwargs=vdb_kwargs) or []

    seen_ids: set[str] = set()
    merged: list[dict[str, Any]] = []
    for hit in full_hits + name_hits:
        hit_id = (hit.get("metadata") or {}).get("id", "")
        if hit_id and hit_id not in seen_ids:
            seen_ids.add(hit_id)
            merged.append(hit)

    source_table_id = col.get("table_id")
    if source_table_id and merged:
        merged = [
            hit
            for hit in merged
            if (hit.get("metadata") or {}).get("table_id")
            and (hit.get("metadata") or {}).get("table_id") != source_table_id
        ]

    if not merged:
        return None

    selected = _llm_pick_hit(col, merged)
    if selected:
        return selected
    return _match_hit_by_sample_values(col, merged, connector)


def _match_hit_by_sample_values(
    col: dict[str, Any],
    hits: list[dict[str, Any]],
    connector: SQLDatabase | None,
) -> str | None:
    """Choose the best VDB hit whose physical column contains every FK sample.

    Runs only when the LLM abstains — probes each same-named candidate's live
    column via a bounded, read-only ``ProbeExecutor`` query and keeps hits
    where every one of the FK's distinct sample values actually appears.
    """
    samples = _distinct_samples(col.get("sample_values"))
    if connector is None or not samples:
        return None

    source_column = str(col.get("name") or "").lower()
    if not source_column:
        return None
    matching_name_hits = [
        hit
        for hit in hits
        if str((hit.get("metadata") or {}).get("source_column") or "").lower()
        == source_column
    ]
    if not matching_name_hits:
        return None

    matches: list[dict[str, Any]] = []
    with ProbeExecutor(connector, max_calls=len(matching_name_hits)) as executor:
        for hit in matching_name_hits:
            sql = _sample_match_sql(
                hit.get("metadata") or {},
                samples,
                getattr(connector, "dialect", None),
            )
            if not sql:
                continue
            result = executor.run(sql, purpose="semantic FK sample match")
            if not result["ok"]:
                continue
            matched_values = {
                str(next(iter(row.values())))
                for row in (result.get("rows") or [])
                if row and next(iter(row.values())) is not None
            }
            if set(samples).issubset(matched_values):
                matches.append(hit)

    if not matches:
        return None

    best = min(matches, key=_hit_score)
    metadata = best.get("metadata") or {}
    selected = metadata.get("id")
    if selected:
        logger.info(
            "resolve_semantic_fks [sql-fallback]: %s.%s matched %d sample value(s) "
            "in %s.%s.%s",
            col.get("table_name"),
            col.get("name"),
            len(samples),
            metadata.get("schema_name") or "",
            metadata.get("table_name") or "",
            metadata.get("source_column") or "",
        )
    return selected


def _distinct_samples(raw: Any) -> list[str]:
    """Parse, deduplicate, and preserve the order of stored sample values."""
    values = parse_sample_values(raw) or []
    return list(dict.fromkeys(value for value in values if value is not None))


def _sample_match_sql(
    metadata: dict[str, Any],
    samples: list[str],
    dialect: str | None,
) -> str | None:
    """Build a dialect-quoted, bounded query for candidate sample membership."""
    table_name = metadata.get("table_name")
    column_name = metadata.get("source_column")
    if not table_name or not column_name or not samples:
        return None

    schema_name = metadata.get("schema_name")
    table = exp.Table(
        this=exp.to_identifier(str(table_name)),
        db=exp.to_identifier(str(schema_name)) if schema_name else None,
    )
    column = exp.column(str(column_name))
    text_column = exp.cast(column, "TEXT")
    query = (
        exp.select(text_column.copy().as_("matched_value"))
        .distinct()
        .from_(table)
        .where(
            text_column.copy().isin(*(exp.Literal.string(value) for value in samples))
        )
        .limit(len(samples))
    )
    try:
        return query.sql(dialect=dialect or None, identify=True)
    except ValueError:
        return query.sql(identify=True)


def _hit_score(hit: dict[str, Any]) -> float:
    """Return VDB distance, treating missing or malformed scores as worst."""
    try:
        return float(hit.get("score", float("inf")))
    except (TypeError, ValueError):
        return float("inf")


def _llm_pick_hit(
    col: dict[str, Any],
    hits: list[dict[str, Any]],
) -> str | None:
    """Ask the LLM which candidate column the foreign key references.

    Returns the ``column_id`` from the chosen hit's metadata, or ``None`` when
    the LLM is not confident enough to pick any candidate.

    The ``column_id`` label only has to be internally consistent between the
    candidate list and the instruction that references it — but it *is* part of
    the prompt, so changing it changes model behaviour.
    """
    col_ctx = (
        f"Foreign-key column: {col.get('name', '')} "
        f"(table: {col.get('table_name', '')})\n"
        f"Description: {col.get('description') or '(none)'}\n"
        f"Sample values: {_format_sample_values(col.get('sample_values')) or '(none)'}"
    )

    candidates_lines: list[str] = []
    for hit in hits:
        column_id = (hit.get("metadata") or {}).get("id", "")
        text = (hit.get("text") or "")[:300]
        candidates_lines.append(f"column_id={column_id} | {text}")
    candidates_block = "\n".join(candidates_lines)

    human_text = (
        f"{col_ctx}\n\n"
        f"Candidate PK columns (from vector search):\n{candidates_block}\n\n"
        "Which candidate does this FK column reference? "
        "Return its exact column_id value, or null if none are a confident match."
    )

    result = invoke_with_structured_output(
        get_non_reasoning_llm_client(max_tokens=4096),
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human_text)],
        FkHitSelection,
    )
    if result is None:
        return None

    selected = result.selected_id
    if not selected:
        return None

    valid_ids = {(hit.get("metadata") or {}).get("id") for hit in hits}
    if selected not in valid_ids:
        logger.warning(
            "resolve_semantic_fks: LLM returned unknown id %r for column %s — discarding",
            selected,
            col.get("id"),
        )
        return None

    return selected
