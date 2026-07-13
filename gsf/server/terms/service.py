# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Term update orchestration."""

from __future__ import annotations

import logging

from gsf.dal import sql_attributes as sql_attr_dal
from gsf.dal import terms as terms_dal
from gsf.semantic.embed import build_semantic_embedder
from gsf.utils import get_embed_params
from gsf.utils.embedding import embed_docs_into_vdb

logger = logging.getLogger(__name__)


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
