# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Term read + update orchestration."""

from __future__ import annotations

import logging
from typing import Any

from auto_ontology.dal import attributes as attributes_dal
from auto_ontology.dal import sql_attributes as sql_attr_dal
from auto_ontology.dal import terms as terms_dal
from auto_ontology.dal.users import resolve_accessible_catalog_ids
from auto_ontology.retrieval.data_access.semantic_search import search_semantic_index
from auto_ontology.semantic.constants import LABEL_TERM
from auto_ontology.semantic.embed import build_semantic_embedder
from auto_ontology.server.datasources import service as datasources_service
from auto_ontology.utils import (
    get_embed_params,
    get_semantic_objects_retriever,
    sample_values_edit_error,
    stringify_sample_values,
)
from auto_ontology.utils.embedding import embed_docs_into_vdb

logger = logging.getLogger(__name__)


def _name_rank(name: str | None, needle: str) -> int:
    """0 for an exact name, 1 for a prefix, 2 for any other substring."""
    lowered = (name or "").lower()
    if lowered == needle:
        return 0
    if lowered.startswith(needle):
        return 1
    return 2


def search_terms(query: str, *, limit: int = 10) -> dict[str, Any]:
    """Terms for a search box: name matches first, then matches by meaning.

    Two reads, blended. The name read is the same literal substring filter the
    Terms list uses, so typing an exact term name always puts it on top. The
    meaning read embeds *query* and ranks Terms in the semantic vector index by
    cosine distance, which finds "client" for *Customer* or "revenue" for
    *Invoice Line* — names a substring can't reach.

    Vector hits are hydrated through :func:`terms_dal.fetch_all_terms`, so a
    hit outside the visible glossary is dropped rather than returned bare.

    The embedding call can fail (endpoint down, key rejected); the name matches
    are still returned and ``semantic`` is ``False`` so the caller can say so.

    Each item carries ``match`` (``"name"`` or ``"meaning"``) and
    ``similarity`` (``1 - cosine distance``, ``None`` when the term had no
    vector hit).
    """
    needle = query.strip()
    if not needle:
        return {"data": [], "count": 0, "semantic": True}

    by_name = terms_dal.fetch_all_terms(search=needle, limit=limit)

    distance: dict[str, float] = {}
    semantic = True
    try:
        hits = search_semantic_index(
            get_semantic_objects_retriever(),
            needle,
            label_filter=[LABEL_TERM],
            per_label_k=limit,
        )
    except Exception:  # noqa: BLE001 - name matches still answer the search
        logger.warning("Semantic term search failed for %r", needle, exc_info=True)
        hits = []
        semantic = False
    for hit in hits:
        term_id = hit.get("id")
        score = hit.get("score")
        if term_id and isinstance(score, (int, float)) and term_id not in distance:
            distance[term_id] = float(score)

    name_ids = {row["id"] for row in by_name}
    by_meaning = (
        terms_dal.fetch_all_terms(
            term_ids=[tid for tid in distance if tid not in name_ids]
        )
        if any(tid not in name_ids for tid in distance)
        else []
    )

    def similarity(term_id: str) -> float | None:
        return round(1 - distance[term_id], 4) if term_id in distance else None

    lowered = needle.lower()
    ranked_names = sorted(
        by_name,
        key=lambda row: (
            _name_rank(row.get("name"), lowered),
            (row.get("name") or "").lower(),
        ),
    )
    ranked_meaning = sorted(by_meaning, key=lambda row: distance[row["id"]])
    items = [
        {**row, "match": "name", "similarity": similarity(row["id"])}
        for row in ranked_names
    ] + [
        {**row, "match": "meaning", "similarity": similarity(row["id"])}
        for row in ranked_meaning
    ]
    items = items[:limit]
    return {"data": items, "count": len(items), "semantic": semantic}


def list_terms_page(
    *,
    zone_ids: list[str] | None = None,
    search: str | None = None,
    skip: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Assemble one page of the Terms list with its per-card count breakdowns.

    Resolving *zone_ids* to accessible catalog ids costs several store round
    trips, so it happens once here and is threaded through the reads that
    accept it (see ``resolve_accessible_catalog_ids``).

    ``total`` counts every term matching *search* and *zone_ids*, not the ones
    on this page, so the caller can tell whether further pages exist.

    The three count lists are per-term breakdowns keyed by ``term_id``. When a
    *limit* is given they cover only the terms on this page — the whole point
    of paging is not to describe the rest of the glossary — so a caller
    holding pages must merge them rather than replace what it already has.
    Called without a *limit* they still cover every visible term.
    """
    data_ids_by_zone = resolve_accessible_catalog_ids(zone_ids)
    terms = terms_dal.fetch_all_terms(
        zone_ids=zone_ids,
        search=search,
        data_ids_by_zone=data_ids_by_zone,
        skip=skip,
        limit=limit,
    )
    # `terms` only holds every matching term when neither paging argument was
    # given — a `skip` alone (no `limit`) still returns a partial read, so it
    # needs the same separate count query and the same page-scoped id list.
    paged = bool(skip) or limit is not None
    page_term_ids = [term["id"] for term in terms] if paged else None
    total = (
        terms_dal.count_terms(
            zone_ids=zone_ids, search=search, data_ids_by_zone=data_ids_by_zone
        )
        if paged
        else len(terms)
    )
    return {
        "terms": terms,
        "total": total,
        "column_attribute_counts": terms_dal.fetch_column_attribute_counts(
            zone_ids=zone_ids,
            data_ids_by_zone=data_ids_by_zone,
            term_ids=page_term_ids,
        ),
        "sql_attribute_counts": sql_attr_dal.fetch_sql_attribute_counts(
            zone_ids=zone_ids,
            data_ids_by_zone=data_ids_by_zone,
            term_ids=page_term_ids,
        ),
        "related_counts": terms_dal.fetch_related_terms_counts(
            zone_ids=zone_ids,
            term_ids=page_term_ids,
            data_ids_by_zone=data_ids_by_zone,
        ),
    }


def update_column_attribute(
    term_id: str,
    attr_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    sample_values: list[str] | None = None,
    certified: bool | None = None,
    modified_by: str | None = None,
    touch: bool = False,
) -> dict[str, Any] | None:
    """Update ColumnAttribute metadata and refresh related VDB rows.

    ``sample_values`` live on the owning Column. When provided, this updates
    that Column (same path as catalog column edit), re-embeds the Column in
    the data VDB, then re-embeds this ColumnAttribute in the semantic VDB.

    Neither the certification flag nor the description is embedding content —
    the attribute embeds its column's description — so an update touching only
    those skips the VDB refresh entirely.

    Raises ``ValueError`` when the owning Column's declared SQL type does not
    accept hand-written samples.
    """
    if sample_values is not None:
        # Refused ahead of the metadata write below, so a rejected patch does
        # not leave a renamed attribute behind. A missing attribute or Column
        # is left to that write, which answers 404 rather than a type
        # complaint; the Column write itself re-checks the type.
        owning_column = attributes_dal.fetch_column_attribute_column(attr_id)
        if owning_column is not None:
            error = sample_values_edit_error(owning_column.get("data_type"))
            if error is not None:
                raise ValueError(error)

    row = attributes_dal.update_column_attribute(
        attr_id,
        term_id,
        name=name,
        description=description,
        certified=certified,
        modified_by=modified_by,
        touch=touch,
    )
    if row is None:
        return None

    if name is None and sample_values is None:
        row["sample_values"] = stringify_sample_values(row.get("sample_values"))
        return row

    if sample_values is not None:
        column_id = row.get("column_id")
        if not column_id:
            raise ValueError(
                "ColumnAttribute has no owning Column; sample values cannot be updated"
            )
        # Reuses catalog Column patch: writes Column.sample_values and
        # refreshes the Column row in the data VDB. It also best-effort
        # refreshes semantic ColumnAttribute rows; we still re-embed this
        # attribute below so Term synonyms stay in the semantic text.
        datasources_service.update_node_properties(
            column_id,
            {"sample_values": sample_values},
        )
        row["sample_values"] = sample_values
    else:
        row["sample_values"] = stringify_sample_values(row.get("sample_values"))

    embedder = build_semantic_embedder(row.get("database_name") or "", reset=False)
    if embedder is not None:
        embedder.vdb.delete_by_id(attr_id)
        attr = {
            "id": row["id"],
            "name": row["name"],
            "column_description": row.get("column_description"),
            "family_prefix": row.get("family_prefix"),
            "term_name": row.get("term_name"),
            "source_column": row.get("source_column"),
            "sample_values": row.get("sample_values"),
            "is_unique": row.get("is_unique"),
            "table_id": row.get("table_id"),
            "table_name": row.get("table_name"),
            "schema_name": row.get("schema_name"),
        }
        # Pass Term synonyms only — omit Term name so we re-embed just this attr row.
        term_ctx = {"synonyms": row.get("term_synonyms") or []}
        embedder.embed_term(term_ctx, [attr])

    return row


def refresh_term_embeddings(term_id: str, *, refresh_dependent_attrs: bool) -> None:
    """Best-effort semantic VDB refresh for one Term update."""
    term, column_attrs = terms_dal.fetch_term_and_column_attributes_for_embedding(
        term_id
    )
    if term is None:
        return

    embedder = build_semantic_embedder(term.get("database_name") or "", reset=False)
    if embedder is None:
        return

    vdb = embedder.vdb
    vdb.delete_by_id(term_id)
    if refresh_dependent_attrs:
        for attr in column_attrs:
            attr_id = attr.get("id")
            if attr_id:
                vdb.delete_by_id(attr_id)

    embedder.embed_term(term, column_attrs if refresh_dependent_attrs else [])

    if refresh_dependent_attrs:
        embed_params = get_embed_params()
        sql_attrs = sql_attr_dal.fetch_sql_attributes_by_term_id(term_id)
        for sql_attr in sql_attrs:
            attr_id = sql_attr.get("id")
            if not attr_id:
                continue
            vdb.delete_by_id(attr_id)
            embed_docs_into_vdb(
                sql_attr_dal.fetch_sql_attribute_docs(attr_id),
                embed_params,
                vdb,
                term.get("database_name"),
            )
