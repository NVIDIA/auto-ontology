# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Term read + update orchestration."""

from __future__ import annotations

import logging
from typing import Any

from gsf.dal import sql_attributes as sql_attr_dal
from gsf.dal import terms as terms_dal
from gsf.semantic.embed import build_semantic_embedder
from gsf.utils import get_embed_params
from gsf.utils.embedding import embed_docs_into_vdb

logger = logging.getLogger(__name__)


def get_all_terms_with_attributes(
    zone_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return every term available to the user as ``{name, attributes}``.

    Terms and their attributes are zone-scoped via *zone_ids* (``None`` → admin /
    no filter, ``[]`` → viewer with no access → empty). Each term's ``attributes``
    list merges its ColumnAttributes and SqlAttributes, each projected to just
    ``{name, description}``. Attributes are bucketed onto terms by ``term_name``
    (the term's natural key), which both attribute kinds carry.
    """
    terms, column_attrs = terms_dal.fetch_all_terms_and_attributes(zone_ids=zone_ids)
    sql_attrs = sql_attr_dal.fetch_sql_attributes(zone_ids=zone_ids)

    attrs_by_term: dict[str, list[dict[str, Any]]] = {}
    for attr in (*column_attrs, *sql_attrs):
        attrs_by_term.setdefault(attr.get("term_name"), []).append(
            {"name": attr.get("name"), "description": attr.get("description")}
        )

    return [
        {
            "name": term.get("name"),
            "attributes": attrs_by_term.get(term.get("name"), []),
        }
        for term in terms
    ]


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
