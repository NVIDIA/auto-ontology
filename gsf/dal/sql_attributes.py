# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend selector for ``gsf.dal.sql_attributes``.

Resolves to the Neo4j or Postgres implementation once, at import, from
``GSF_STORE``. See :mod:`gsf.infra.store`.

**The exception types are re-exported too, and must be the same objects.**
``service.py`` catches ``SqlAttributeNameConflict`` imported from here, so if
each backend defined its own class the ``except`` would silently stop matching
and a name conflict would surface as a 500 instead of a 409. Both are defined in
the implementation modules and selected here, exactly like the functions.

Phase 11 deletes this file and promotes ``pg/sql_attributes.py`` in its place.
"""

from gsf.infra.store import USE_PG

if USE_PG:
    from gsf.dal.pg.sql_attributes import (  # noqa: F401
        SqlAttributeExpressionConflict,
        SqlAttributeNameConflict,
        SqlAttributeSqlError,
        clear_sql_attribute_description_suggestion,
        clear_sql_attribute_description_suggestions_for_term,
        count_sql_attributes_by_term_id,
        delete_sql_attribute_node,
        detach_existing_sql_edges,
        fetch_sql_attribute_counts,
        fetch_sql_attribute_docs,
        fetch_sql_attributes_by_term_id,
        fetch_sql_attributes_with_sql,
        fetch_tables_from_sql_attributes,
        find_attr_by_expression,
        find_attr_by_name,
        get_full_sql_attribute_by_id,
        get_sql_attribute_by_id,
        link_to_term,
        list_sql_attributes,
        set_sql_attribute_description_suggestion,
        update_sql_attribute,
    )
else:
    from gsf.dal.neo4j.sql_attributes import (  # noqa: F401
        SqlAttributeExpressionConflict,
        SqlAttributeNameConflict,
        SqlAttributeSqlError,
        clear_sql_attribute_description_suggestion,
        clear_sql_attribute_description_suggestions_for_term,
        count_sql_attributes_by_term_id,
        delete_sql_attribute_node,
        detach_existing_sql_edges,
        fetch_sql_attribute_counts,
        fetch_sql_attribute_docs,
        fetch_sql_attributes_by_term_id,
        fetch_sql_attributes_with_sql,
        fetch_tables_from_sql_attributes,
        find_attr_by_expression,
        find_attr_by_name,
        get_full_sql_attribute_by_id,
        get_sql_attribute_by_id,
        link_to_term,
        list_sql_attributes,
        set_sql_attribute_description_suggestion,
        update_sql_attribute,
    )

__all__ = [
    "SqlAttributeExpressionConflict",
    "SqlAttributeNameConflict",
    "SqlAttributeSqlError",
    "clear_sql_attribute_description_suggestion",
    "clear_sql_attribute_description_suggestions_for_term",
    "count_sql_attributes_by_term_id",
    "delete_sql_attribute_node",
    "detach_existing_sql_edges",
    "fetch_sql_attribute_counts",
    "fetch_sql_attribute_docs",
    "fetch_sql_attributes_by_term_id",
    "fetch_sql_attributes_with_sql",
    "fetch_tables_from_sql_attributes",
    "find_attr_by_expression",
    "find_attr_by_name",
    "get_full_sql_attribute_by_id",
    "get_sql_attribute_by_id",
    "link_to_term",
    "list_sql_attributes",
    "set_sql_attribute_description_suggestion",
    "update_sql_attribute",
]
