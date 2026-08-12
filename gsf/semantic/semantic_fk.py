"""Resolve SEMANTIC_FK edges after taxonomy compilation.

Algorithm
---------
1. Find every Column node that has no SEMANTIC_FK edge and no HAS_ATTRIBUTE
   edge (i.e. a FK column that the taxonomy pass skipped).
2. For each such column:
   a. **Declared FK fast path** — if a FOREIGN_KEY edge already exists in the
      graph, look up the ColumnAttribute attached to the target Column and
      create SEMANTIC_FK directly.
   b. **LLM / VDB fallback** — if there is no declared FK, embed the column
      context, search the *semantic* VDB for the top-5 most similar
      ColumnAttribute records, ask the LLM to pick the best match, and create
      SEMANTIC_FK when a match is found.  The hit's ``metadata["id"]`` is the
      ColumnAttribute Neo4j UUID directly — no additional graph lookup needed.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from nemo_retriever.graph.retriever import Retriever

from gsf.dal.attributes import (
    find_column_attribute_by_column_id,
    find_unlinked_fk_columns,
    merge_semantic_fk,
)
from gsf.utils.llm_invoke import (
    get_non_reasoning_llm_client,
    invoke_with_structured_output,
)
from gsf.utils.model_config import resolve
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
vector search. Each candidate is prefixed with its unique neo4j_id. \
Your task is to decide which candidate (if any) is the column that the \
foreign key references.

Rules:
- Return the exact neo4j_id of the best matching candidate, or null if none fit.
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
    llm_queue: list[dict[str, Any]] = []

    for col in candidates:
        fk_target_col_id: str | None = col.get("fk_target_col_id")
        if fk_target_col_id:
            attr_id = find_column_attribute_by_column_id(fk_target_col_id)
            if attr_id:
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
                llm_queue.append(col)
        else:
            llm_queue.append(col)

    logger.info(
        "resolve_semantic_fks: %d declared FK(s) resolved; %d queued for LLM",
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

    llm_written = 0

    def _resolve_one(col: dict[str, Any]) -> bool:
        try:
            attr_id = _resolve_via_vdb(col, retriever, database_name)
            if attr_id:
                merge_semantic_fk(col["id"], attr_id)
                logger.debug(
                    "resolve_semantic_fks [llm]: %s.%s → attr %s",
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
        futures = {pool.submit(_resolve_one, col): col for col in llm_queue}
        for future in as_completed(futures):
            if future.result():
                llm_written += 1

    total = declared_written + llm_written
    logger.info(
        "resolve_semantic_fks: done — %d declared + %d LLM = %d total SEMANTIC_FK edge(s)",
        declared_written,
        llm_written,
        total,
    )
    return total


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
) -> str | None:
    """Search the semantic VDB for a matching ColumnAttribute and return its id.

    Runs two queries and merges up to 6 unique hits:
    - name + description query (top 3): captures semantic context
    - name-only query (top 3): catches cases where description is noisy or absent

    The hit's ``metadata["id"]`` is the ColumnAttribute Neo4j UUID directly —
    no additional graph lookup is required.

    Returns the ColumnAttribute id if a confident LLM match is found, else None.
    """
    vdb_kwargs = {
        "where": {
            "label": "ColumnAttribute",
            "database_name": database_name,
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

    if not merged:
        return None

    return _llm_pick_hit(col, merged)


def _llm_pick_hit(
    col: dict[str, Any],
    hits: list[dict[str, Any]],
) -> str | None:
    """Ask the LLM to select the Neo4j column ID of the best matching hit.

    Returns the ``neo4j_id`` string from the chosen hit's metadata, or ``None``
    when the LLM is not confident enough to pick any candidate.
    """
    col_ctx = (
        f"Foreign-key column: {col.get('name', '')} "
        f"(table: {col.get('table_name', '')})\n"
        f"Description: {col.get('description') or '(none)'}\n"
        f"Sample values: {_format_sample_values(col.get('sample_values')) or '(none)'}"
    )

    candidates_lines: list[str] = []
    for hit in hits:
        neo4j_id = (hit.get("metadata") or {}).get("id", "")
        text = (hit.get("text") or "")[:300]
        candidates_lines.append(f"neo4j_id={neo4j_id} | {text}")
    candidates_block = "\n".join(candidates_lines)

    human_text = (
        f"{col_ctx}\n\n"
        f"Candidate PK columns (from vector search):\n{candidates_block}\n\n"
        "Which candidate does this FK column reference? "
        "Return its exact neo4j_id value, or null if none are a confident match."
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
