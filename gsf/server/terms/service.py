# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Term read + update orchestration."""

from __future__ import annotations

import logging
from typing import Any

from gsf.dal import attributes as attributes_dal
from gsf.dal import sql_attributes as sql_attr_dal
from gsf.dal import terms as terms_dal
from gsf.semantic.embed import build_semantic_embedder
from gsf.server.datasources import service as datasources_service
from gsf.utils import get_embed_params, parse_sample_values
from gsf.utils.embedding import embed_docs_into_vdb

logger = logging.getLogger(__name__)


def update_column_attribute(
    term_id: str,
    attr_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    sample_values: list[str] | None = None,
) -> dict[str, Any] | None:
    """Update ColumnAttribute metadata and refresh related VDB rows.

    ``sample_values`` live on the owning Column. When provided, this updates
    that Column (same path as catalog column edit), re-embeds the Column in
    the data VDB, then re-embeds this ColumnAttribute in the semantic VDB.
    """
    row = attributes_dal.update_column_attribute(
        attr_id,
        term_id,
        name=name,
        description=description,
    )
    if row is None:
        return None

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
        row["sample_values"] = parse_sample_values(row.get("sample_values"))

    embedder = build_semantic_embedder(row.get("database_name") or "", reset=False)
    if embedder is not None:
        embedder.vdb.delete_by_id(attr_id)
        attr = {
            "id": row["id"],
            "name": row["name"],
            "description": row.get("description"),
            "term_name": row.get("term_name"),
            "source_column": row.get("source_column"),
            "sample_values": row.get("sample_values"),
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
