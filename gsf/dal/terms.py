# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend selector for ``gsf.dal.terms``.

Resolves to the Neo4j or Postgres implementation once, at import, from
``GSF_STORE``. See :mod:`gsf.infra.store`.

Phase 11 deletes this file and promotes ``pg/terms.py`` in its place.
"""

from gsf.infra.store import USE_PG

if USE_PG:
    from gsf.dal.pg.terms import (  # noqa: F401
        get_term_certification,
        term_is_in_scope,
        semantic_layer_calculated,
        get_term_record_for_table,
        get_slim_term_by_id,
        update_term,
        merge_term,
        fetch_term_synonyms,
        fetch_all_terms,
        count_terms,
        fetch_all_terms_and_attributes,
        get_full_term_by_id,
        fetch_term_zones_map,
        fetch_table_schema_map,
        fetch_terms_with_sqls,
        fetch_terms_and_attributes_for_table,
        fetch_term_and_column_attributes_for_embedding,
        fetch_column_attribute_embedding_contexts_by_column_id,
        fetch_column_attribute_counts,
        fetch_column_attributes_by_term_id,
        count_column_attributes_by_term_id,
        find_column_attribute_by_column_id,
        fetch_related_term_ids,
        fetch_terms_by_ids,
        fetch_related_terms,
        fetch_related_terms_counts,
        fetch_term_table_pairs,
        build_term_table_maps,
    )
else:
    from gsf.dal.neo4j.terms import (  # noqa: F401
        get_term_certification,
        term_is_in_scope,
        semantic_layer_calculated,
        get_term_record_for_table,
        get_slim_term_by_id,
        update_term,
        merge_term,
        fetch_term_synonyms,
        fetch_all_terms,
        count_terms,
        fetch_all_terms_and_attributes,
        get_full_term_by_id,
        fetch_term_zones_map,
        fetch_table_schema_map,
        fetch_terms_with_sqls,
        fetch_terms_and_attributes_for_table,
        fetch_term_and_column_attributes_for_embedding,
        fetch_column_attribute_embedding_contexts_by_column_id,
        fetch_column_attribute_counts,
        fetch_column_attributes_by_term_id,
        count_column_attributes_by_term_id,
        find_column_attribute_by_column_id,
        fetch_related_term_ids,
        fetch_terms_by_ids,
        fetch_related_terms,
        fetch_related_terms_counts,
        fetch_term_table_pairs,
        build_term_table_maps,
    )

__all__ = [
    "get_term_certification",
    "term_is_in_scope",
    "semantic_layer_calculated",
    "get_term_record_for_table",
    "get_slim_term_by_id",
    "update_term",
    "merge_term",
    "fetch_term_synonyms",
    "fetch_all_terms",
    "count_terms",
    "fetch_all_terms_and_attributes",
    "get_full_term_by_id",
    "fetch_term_zones_map",
    "fetch_table_schema_map",
    "fetch_terms_with_sqls",
    "fetch_terms_and_attributes_for_table",
    "fetch_term_and_column_attributes_for_embedding",
    "fetch_column_attribute_embedding_contexts_by_column_id",
    "fetch_column_attribute_counts",
    "fetch_column_attributes_by_term_id",
    "count_column_attributes_by_term_id",
    "find_column_attribute_by_column_id",
    "fetch_related_term_ids",
    "fetch_terms_by_ids",
    "fetch_related_terms",
    "fetch_related_terms_counts",
    "fetch_term_table_pairs",
    "build_term_table_maps",
]
