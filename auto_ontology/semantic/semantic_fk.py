# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve SEMANTIC_FK edges after taxonomy compilation.

Algorithm
---------
1. Find every Column node that has no SEMANTIC_FK edge and no HAS_ATTRIBUTE
   edge (i.e. a FK column that the taxonomy pass skipped).
2. For each such column:
   a. **Declared FK fast path** — if a FOREIGN_KEY edge already exists in the
      graph, look up the ColumnAttribute attached to the target Column and
      create SEMANTIC_FK directly. A declared FK that targets *the column
      itself* is rejected; one that targets another column of the same table
      is a self-reference (``posts.parent_id → posts.id``) and is kept.
   b. **Inferred** — otherwise, search the *semantic* VDB for ColumnAttributes
      whose names and descriptions resemble the column, let the LLM pick one,
      and confirm the pick by comparing the two columns' ascending distinct
      values (``semantic.fk_values``); the other hits are tried in distance
      order when the pick fails. When no hit is confirmed, every key column
      whose stored values hold the FK's is shortlisted and the LLM picks from
      that list instead. Neither path offers another term family's copy of a
      table the column's own family has. A key on the column's own table is
      offered marked as a self-reference, and only the LLM's explicit pick of
      it is accepted — the distance-order fallback never lands there, since a
      table's own dense ``id`` holds nearly any integer FK's values.
   c. **Validated** — an inferred edge is stored only when joining the two
      columns returns a row.

``PREVENT_SEMANTIC_FK_DB_QUERIES`` turns off everything that queries the
warehouse — on-demand probes, the value-only fallback and the join check —
leaving the LLM's vector-search pick unconfirmed.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from sqlglot import exp

from auto_ontology.connectors import get_connectors
from auto_ontology.connectors.base import SQLDatabase
from auto_ontology.dal.attributes import (
    fetch_semantic_fk_join_groups,
    fetch_value_key_candidates,
    find_column_attribute_by_column_id,
    find_unlinked_fk_columns,
    merge_semantic_fk,
    set_semantic_fk_join_default,
)
from auto_ontology.dal.datasources import store_column_distinct_values
from auto_ontology.dal.term_families import fetch_table_families
from auto_ontology.infra.feature_flags import is_semantic_fk_db_query_prevented
from auto_ontology.semantic.fk_values import (
    MIN_CONTAINMENT,
    containment,
    contains,
    is_value_search_ambiguous,
)
from auto_ontology.semantic.visit_enter import (
    KEY_DISTINCT_VALUES_LIMIT,
    execute_capped,
    probe_ascending_distinct_values,
)
from auto_ontology.utils.embedding import EmbedParams
from auto_ontology.utils.llm_invoke import (
    get_llm_client,
    invoke_with_structured_output,
)
from auto_ontology.utils.model_config import resolve
from auto_ontology.utils.retriever import Retriever
from auto_ontology.utils.sample_values import (
    parse_sample_values,
    stringify_sample_values,
)
from auto_ontology.semantic.models import (
    FkHitSelection,
    SemanticFkJoinDefaultSelection,
)
from auto_ontology.vdb import get_semantic_vdb

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
- A candidate marked [same table] is a self-reference: rows pointing at other \
rows of the same entity (a parent, a manager, an accepted answer). Pick it only \
when the column's name or description says so.
- Do NOT guess. If unsure, return null.
"""
_VALUE_SYSTEM_PROMPT = """\
You are a database schema expert. A foreign-key column's name matched no \
primary key, so candidate key columns were found by their values instead: each \
candidate, prefixed with its unique column_id, holds most of the foreign key's \
lowest distinct values. Shared values alone can be a coincidence — consecutive \
ids and short codes overlap across unrelated tables.

Rules:
- Pick a candidate only when its table is plausibly the entity the foreign key \
refers to, judging by names, descriptions and values together.
- A candidate marked [same table] is a self-reference: rows pointing at other \
rows of the same entity (a parent, a manager, an accepted answer). Pick it only \
when the column's name or description says so — the table's own key holds most \
of its integer columns' values.
- Return the exact column_id of that candidate, or null if none is plausible.
- Do NOT guess. If unsure, return null.
"""
# Candidates the value-only search hands the LLM.
_VALUE_SHORTLIST_SIZE = 5

_JOIN_DEFAULT_SYSTEM_PROMPT = """\
You are choosing the default foreign-key relationship used to join two database
tables. Several columns on the same source table reference the same target column,
but each column can represent a different business role.

Choose the relationship that most naturally represents an unqualified join between
the source and target tables. Return exactly one supplied source_column_id. Return
null only when the schema context is genuinely insufficient. Never invent an ID.
"""


def resolve_semantic_fks(database_name: str) -> int:
    """Create SEMANTIC_FK edges for all unlinked FK columns.

    Runs after the taxonomy while-loop in ``compile_semantic_layer``.
    Returns the total number of SEMANTIC_FK edges created.
    """
    created = _resolve_semantic_fks(database_name)
    select_semantic_fk_join_defaults(database_name)
    return created


def _resolve_semantic_fks(database_name: str) -> int:
    """Create missing semantic FK edges before default selection."""
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
            if fk_target_col_id == col["id"]:
                _log_start(index, col)
                logger.debug(
                    "resolve_semantic_fks [declared]: %s.%s declared FK targets "
                    "itself — rejecting",
                    col.get("table_name"),
                    col.get("name"),
                )
                continue
            attr_id = find_column_attribute_by_column_id(fk_target_col_id)
            if attr_id:
                _log_start(index, col)
                # merge_semantic_fk refuses (and logs) a cross-database edge on
                # its own -- should be structurally impossible for a declared
                # FK, whose target is always a column in the same connected
                # database, but treat "not written" the same as "no attribute
                # found" rather than assume.
                if merge_semantic_fk(col["id"], attr_id):
                    declared_written += 1
                    logger.debug(
                        "resolve_semantic_fks [declared]: %s.%s → attr %s",
                        col.get("table_name"),
                        col.get("name"),
                        attr_id,
                    )
                else:
                    llm_queue.append((index, col))
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
    keys: _KeyIndex | None = None
    if is_semantic_fk_db_query_prevented():
        logger.info(
            "resolve_semantic_fks: PREVENT_SEMANTIC_FK_DB_QUERIES is set — "
            "inferred FKs are neither confirmed by values nor validated by a join"
        )
    else:
        connector = _resolve_connector(database_name)
        if connector is not None:
            keys = _KeyIndex(database_name, connector)

    families = _Families(fetch_table_families(database_name))
    inferred_written = 0

    def _resolve_one(index: int, col: dict[str, Any]) -> bool:
        _log_start(index, col)
        try:
            attr_id = _resolve_via_vdb(col, retriever, database_name, keys, families)
            if (
                attr_id
                and keys is not None
                and not _join_returns_rows(keys, col, attr_id)
            ):
                return False
            # The VDB where-filter already scopes hits to database_name, but
            # it's a query-time filter over a collection shared by every
            # ingested database, not physical isolation -- the same shape of
            # mechanism that let Term nodes leak across databases elsewhere.
            # merge_semantic_fk verifies against the catalog before writing
            # rather than trust the filter alone, and logs its own rejection.
            if attr_id and merge_semantic_fk(col["id"], attr_id):
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


def select_semantic_fk_join_defaults(database_name: str) -> int:
    """Persist exactly one join-path default for every semantic FK group."""
    rows = fetch_semantic_fk_join_groups(database_name)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["source_table_id"], row["target_attribute_id"])
        grouped.setdefault(key, []).append(row)

    ambiguous = [group for group in grouped.values() if len(group) > 1]
    llm = None
    if ambiguous:
        try:
            llm = get_llm_client()
        except EnvironmentError:
            logger.warning(
                "Semantic FK join-default selection has no configured LLM; "
                "using deterministic fallbacks for %d group(s)",
                len(ambiguous),
            )
        except Exception:
            logger.exception(
                "Semantic FK join-default client initialization failed; "
                "using deterministic fallbacks for %d group(s)",
                len(ambiguous),
            )

    for (source_table_id, target_attribute_id), group in grouped.items():
        selected_column_id = group[0]["source_column_id"]
        if len(group) > 1:
            llm_selected = _select_join_default_with_llm(group, llm)
            if llm_selected is None:
                selected_column_id = _deterministic_join_default(group)
                logger.warning(
                    "Semantic FK join-default selection fell back to %s for %s -> %s",
                    selected_column_id,
                    group[0]["source_table_name"],
                    group[0]["target_table_name"],
                )
            else:
                selected_column_id = llm_selected

        set_semantic_fk_join_default(
            source_table_id,
            target_attribute_id,
            selected_column_id,
        )
        logger.info(
            "Semantic FK join default: %s.%s -> %s.%s",
            group[0]["source_table_name"],
            next(
                row["source_column_name"]
                for row in group
                if row["source_column_id"] == selected_column_id
            ),
            group[0]["target_table_name"],
            group[0]["target_column_name"],
        )

    return len(grouped)


def _select_join_default_with_llm(
    group: list[dict[str, Any]],
    llm: object | None,
) -> str | None:
    """Return an exact source-column ID selected by the LLM, or ``None``."""
    if llm is None:
        return None

    first = group[0]
    candidates = "\n".join(
        (
            f"- source_column_id: {row['source_column_id']}\n"
            f"  column: {row['source_column_name']}\n"
            f"  description: {row.get('source_column_description') or '(none)'}"
        )
        for row in group
    )
    prompt = f"""\
Source table: {first["source_table_name"]}
Source table description: {first.get("source_table_description") or "(none)"}
Target table: {first["target_table_name"]}
Target table description: {first.get("target_table_description") or "(none)"}
Target column: {first["target_column_name"]}
Target column description: {first.get("target_column_description") or "(none)"}
Target semantic attribute: {first.get("target_attribute_name") or "(none)"}

Candidate relationships:
{candidates}
"""
    try:
        result = invoke_with_structured_output(
            llm,
            [
                SystemMessage(content=_JOIN_DEFAULT_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ],
            SemanticFkJoinDefaultSelection,
        )
    except Exception:
        logger.exception(
            "Semantic FK join-default LLM failed for %s -> %s",
            first["source_table_name"],
            first["target_table_name"],
        )
        return None

    selected = result.selected_column_id if result is not None else None
    valid_ids = {row["source_column_id"] for row in group}
    if selected not in valid_ids:
        if selected is not None:
            logger.warning(
                "Semantic FK join-default LLM returned unknown column id %r",
                selected,
            )
        return None
    return selected


def _deterministic_join_default(group: list[dict[str, Any]]) -> str:
    """Choose a stable fallback when the LLM cannot decide."""
    return min(
        group,
        key=lambda row: (
            str(row.get("source_column_name") or "").casefold(),
            str(row["source_column_id"]),
        ),
    )["source_column_id"]


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

    return Retriever(
        get_semantic_vdb(database_name=database_name),
        EmbedParams(
            model_name=_EMBED_MODEL,
            embed_invoke_url=_EMBED_ENDPOINT,
            api_key=_NVIDIA_API_KEY,
        ),
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


def _format_sample_values(raw: Any) -> str:
    """Return a 'sample_values: ...' string, or empty when unavailable."""
    values = stringify_sample_values(raw, max_len=30)
    if not values:
        return ""
    return "sample_values: " + ", ".join(values)


class _KeyIndex:
    """The run's candidate key columns and their ascending distinct values.

    A candidate whose values were never stored — compiled before the values
    existed, or its probe failed — is probed when first compared, and the
    result stored so the next compile need not ask again.
    """

    def __init__(self, database_name: str, connector: SQLDatabase) -> None:
        self.connector = connector
        self._rows = {
            row["attribute_id"]: row
            for row in fetch_value_key_candidates(database_name)
        }
        self._values: dict[str, list[Any] | None] = {
            attribute_id: parse_sample_values(row.get("distinct_values"))
            for attribute_id, row in self._rows.items()
        }
        self._lock = threading.Lock()

    def target(self, attribute_id: str) -> dict[str, Any] | None:
        return self._rows.get(attribute_id)

    def values(self, attribute_id: str) -> list[Any] | None:
        row = self._rows.get(attribute_id)
        if row is None:
            return None
        with self._lock:
            cached = self._values.get(attribute_id)
        if cached is not None:
            return cached or None
        probed = probe_ascending_distinct_values(
            self.connector,
            row.get("schema_name"),
            row["table_name"],
            row["column_name"],
        )
        if probed:
            store_column_distinct_values(row["table_id"], {row["column_name"]: probed})
        with self._lock:
            self._values[attribute_id] = probed or []
        return probed or None

    def stored(self) -> list[tuple[dict[str, Any], list[Any]]]:
        """Candidates with values already in hand; never probes."""
        with self._lock:
            return [
                (self._rows[attribute_id], values)
                for attribute_id, values in self._values.items()
                if values
            ]


class _Families:
    """The term families of one database, by table and by base name."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._members: dict[str, set[tuple[str, str]]] = {}
        self._bases: dict[str, set[str]] = {}
        for row in rows:
            self._members.setdefault(row["table_id"], set()).add(
                (row["family_id"], row["base_name"])
            )
            self._bases.setdefault(row["family_id"], set()).add(row["base_name"])

    def crosses(self, source_table_id: str | None, target_table_id: str | None) -> bool:
        """Whether the target is another family's copy of a table the source's has.

        ``OLD_ROOMS.hotel_id`` refers to ``OLD_HOTELS``, never ``NEW_HOTELS``.
        Only that is refused: when the source's family has no ``HOTELS`` of
        its own, ``NEW_HOTELS`` is the only one there is, and a table outside
        every family — a shared dimension — is open to all of them.
        """
        source = self._members.get(source_table_id or "")
        target = self._members.get(target_table_id or "")
        if not source or not target:
            return False
        source_families = {family for family, _ in source}
        if any(family in source_families for family, _ in target):
            return False
        return any(
            base in self._bases[family]
            for family in source_families
            for _, base in target
        )


_NO_FAMILIES = _Families([])


def _resolve_via_vdb(
    col: dict[str, Any],
    retriever: Retriever,
    database_name: str,
    keys: _KeyIndex | None = None,
    families: _Families = _NO_FAMILIES,
) -> str | None:
    """Return the id of the ColumnAttribute *col* references, or ``None``.

    The LLM picks among the vector-search hits; with *keys* (warehouse
    queries allowed) the pick has to be confirmed by values, and the other
    hits are tried in distance order when it is not. A column without stored
    values cannot be compared, so its pick stands unconfirmed. When no hit is
    confirmed the value-only search runs.

    A candidate that is another family's copy of a table the column's own
    family has is never offered (:meth:`_Families.crosses`).
    """
    hits = [
        hit
        for hit in _vector_hits(col, retriever, database_name)
        if not families.crosses(
            col.get("table_id"), (hit.get("metadata") or {}).get("table_id")
        )
    ]
    selected = _llm_pick_hit(col, hits) if hits else None
    if keys is None:
        return selected
    fk_values = parse_sample_values(col.get("distinct_values"))
    if not fk_values:
        return selected
    confirmed = _confirm_by_values(col, fk_values, hits, selected, keys)
    if confirmed:
        return confirmed
    return _match_by_values_only(col, fk_values, keys, families)


def _vector_hits(
    col: dict[str, Any],
    retriever: Retriever,
    database_name: str,
) -> list[dict[str, Any]]:
    """Up to 6 unique ColumnAttribute hits resembling *col*.

    Runs two queries of top 3 each — name + description, which captures
    semantic context, and name only, which survives a noisy or absent
    description. The FK column itself is excluded, and so are candidates whose
    owning column is not unique, which cannot be the referenced key; another
    key of its own table stays, as a possible self-reference. The hit's
    ``metadata["id"]`` is the ColumnAttribute id.
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

    return [
        hit
        for hit in merged
        if (hit.get("metadata") or {}).get("table_id")
        and not (
            _on_own_table(col, hit)
            and (hit.get("metadata") or {}).get("source_column") == col.get("name")
        )
    ]


def _on_own_table(col: dict[str, Any], hit: dict[str, Any]) -> bool:
    """Whether the candidate *hit* lives on the FK column's own table."""
    table_id = col.get("table_id")
    return bool(table_id) and (hit.get("metadata") or {}).get("table_id") == table_id


def _confirm_by_values(
    col: dict[str, Any],
    fk_values: list[Any],
    hits: list[dict[str, Any]],
    selected: str | None,
    keys: _KeyIndex,
) -> str | None:
    """The first hit — the LLM's pick, then by distance — whose values hold *col*'s.

    A hit on *col*'s own table is tried only as the LLM's pick.
    """
    ordered = [selected] if selected else []
    for hit in sorted(hits, key=_hit_score):
        hit_id = (hit.get("metadata") or {}).get("id")
        if hit_id and hit_id not in ordered and not _on_own_table(col, hit):
            ordered.append(hit_id)
    for attribute_id in ordered:
        if contains(
            fk_values, keys.values(attribute_id), limit=KEY_DISTINCT_VALUES_LIMIT
        ):
            if attribute_id != selected:
                logger.info(
                    "resolve_semantic_fks [values]: %s.%s confirmed by values on "
                    "hit %s rather than the LLM's pick %s",
                    col.get("table_name"),
                    col.get("name"),
                    attribute_id,
                    selected,
                )
            return attribute_id
    if selected:
        logger.info(
            "resolve_semantic_fks [values]: %s.%s — no vector-search hit holds "
            "its values, including the LLM's pick %s",
            col.get("table_name"),
            col.get("name"),
            selected,
        )
    return None


def _match_by_values_only(
    col: dict[str, Any],
    fk_values: list[Any],
    keys: _KeyIndex,
    families: _Families,
) -> str | None:
    """Pick among the key columns whose stored values hold *col*'s."""
    if is_value_search_ambiguous(str(col.get("name") or ""), fk_values):
        return None
    scored: list[tuple[float, dict[str, Any], list[Any]]] = []
    for row, values in keys.stored():
        if row["column_id"] == col.get("id") or families.crosses(
            col.get("table_id"), row["table_id"]
        ):
            continue
        share = containment(fk_values, values, limit=KEY_DISTINCT_VALUES_LIMIT)
        if share is not None and share >= MIN_CONTAINMENT:
            scored.append((share, row, values))
    if not scored:
        return None
    scored.sort(
        key=lambda item: (-item[0], item[1]["table_name"], item[1]["column_name"])
    )
    candidates = [
        {
            "metadata": {"id": row["attribute_id"], "table_id": row["table_id"]},
            "text": (
                f"{row.get('schema_name') or ''}.{row['table_name']}."
                f"{row['column_name']} ({row.get('data_type') or 'unknown type'}); "
                f"holds {share:.0%} of the foreign key's comparable values; "
                f"lowest values: {', '.join(stringify_sample_values(values, max_len=30) or [])}"
            ),
        }
        for share, row, values in scored[:_VALUE_SHORTLIST_SIZE]
    ]
    selected = _llm_pick_hit(
        col,
        candidates,
        system_prompt=_VALUE_SYSTEM_PROMPT,
        source="matched by values",
    )
    if selected:
        logger.info(
            "resolve_semantic_fks [values-only]: %s.%s → attr %s, from %d candidate(s)",
            col.get("table_name"),
            col.get("name"),
            selected,
            len(candidates),
        )
    return selected


def _join_returns_rows(keys: _KeyIndex, col: dict[str, Any], attribute_id: str) -> bool:
    """Whether joining *col* to the attribute's column returns a row.

    Retried with both sides cast to text when the plain join fails, as it does
    when the warehouse refuses to compare the two types; a join that runs and
    returns nothing is final.
    """
    target = keys.target(attribute_id)
    if target is None:
        logger.warning(
            "resolve_semantic_fks [join]: %s.%s — attr %s is not a known key "
            "column; not stored",
            col.get("table_name"),
            col.get("name"),
            attribute_id,
        )
        return False
    label = (
        f"{col.get('table_name')}.{col.get('name')} -> "
        f"{target['table_name']}.{target['column_name']}"
    )
    for as_text in (False, True):
        sql = _join_probe_sql(keys.connector, col, target, as_text=as_text)
        try:
            df = execute_capped(keys.connector, sql)
        except Exception as exc:
            logger.info(
                "resolve_semantic_fks [join]: %s failed%s (%s)",
                label,
                " as text" if as_text else "",
                type(exc).__name__,
            )
            continue
        if df is not None and len(df) > 0:
            return True
        logger.info(
            "resolve_semantic_fks [join]: %s returns no rows; not stored", label
        )
        return False
    logger.info("resolve_semantic_fks [join]: %s could not run; not stored", label)
    return False


def _join_probe_sql(
    connector: SQLDatabase,
    col: dict[str, Any],
    target: dict[str, Any],
    *,
    as_text: bool,
) -> str:
    """``SELECT 1 ... JOIN ... LIMIT 1`` over the two columns."""
    dialect = getattr(connector, "dialect", None) or None
    left: exp.Expression = exp.Column(
        this=exp.to_identifier(str(col["name"]), quoted=True),
        table=exp.to_identifier("s"),
    )
    right: exp.Expression = exp.Column(
        this=exp.to_identifier(str(target["column_name"]), quoted=True),
        table=exp.to_identifier("t"),
    )
    if as_text:
        left, right = exp.cast(left, "TEXT"), exp.cast(right, "TEXT")
    condition = exp.EQ(this=left, expression=right)
    try:
        rendered = condition.sql(dialect=dialect)
    except ValueError:
        rendered = condition.sql()
    source = connector.qualify(col.get("schema_name"), col["table_name"])
    target_table = connector.qualify(target.get("schema_name"), target["table_name"])
    return (
        f"SELECT 1 AS hit FROM {source} s JOIN {target_table} t ON {rendered} LIMIT 1"
    )


def _hit_score(hit: dict[str, Any]) -> float:
    """Return the hit's vector distance (lower is closer); missing or malformed is worst.

    The retriever reports it as ``_distance``. This used to read ``score``,
    which no hit ever carried, so every candidate tied and the first one won.
    """
    try:
        return float(hit.get("_distance", float("inf")))
    except (TypeError, ValueError):
        return float("inf")


def _llm_pick_hit(
    col: dict[str, Any],
    hits: list[dict[str, Any]],
    *,
    system_prompt: str = _SYSTEM_PROMPT,
    source: str = "from vector search",
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
    lowest = stringify_sample_values(col.get("distinct_values"), max_len=30)
    if lowest:
        col_ctx += f"\nLowest distinct values: {', '.join(lowest)}"

    candidates_lines: list[str] = []
    for hit in hits:
        column_id = (hit.get("metadata") or {}).get("id", "")
        text = (hit.get("text") or "")[:300]
        marker = " [same table]" if _on_own_table(col, hit) else ""
        candidates_lines.append(f"column_id={column_id}{marker} | {text}")
    candidates_block = "\n".join(candidates_lines)

    human_text = (
        f"{col_ctx}\n\n"
        f"Candidate PK columns ({source}):\n{candidates_block}\n\n"
        "Which candidate does this FK column reference? "
        "Return its exact column_id value, or null if none are a confident match."
    )

    result = invoke_with_structured_output(
        get_llm_client(max_tokens=4096),
        [SystemMessage(content=system_prompt), HumanMessage(content=human_text)],
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
