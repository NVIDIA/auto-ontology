# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend selector for ``gsf.dal.attributes``.

Resolves to the Neo4j or Postgres implementation once, at import, from
``GSF_STORE``. See :mod:`gsf.infra.store`.

The import list is explicit rather than ``import *``: it *is* the frozen public
surface both implementations must provide, and a star-import would silently
paper over anything the Postgres side had not implemented.

Phase 11 deletes this file and promotes ``pg/attributes.py`` in its place.
"""

from gsf.infra.store import USE_PG

if USE_PG:
    from gsf.dal.pg.attributes import (  # noqa: F401
        merge_column_attribute,
        update_column_attribute,
        find_column_attribute_by_column_id,
        fetch_attr_column_contexts,
        find_unlinked_fk_columns,
        fetch_column_attribute_columns_map,
        merge_semantic_fk,
        find_join_path,
    )
else:
    from gsf.dal.neo4j.attributes import (  # noqa: F401
        merge_column_attribute,
        update_column_attribute,
        find_column_attribute_by_column_id,
        fetch_attr_column_contexts,
        find_unlinked_fk_columns,
        fetch_column_attribute_columns_map,
        merge_semantic_fk,
        find_join_path,
    )

#: The selector's public surface, spelled out because ``test_dal_surface``
#: resolves re-exports by ``__all__`` — a re-exported function's ``__module__``
#: points at the implementation, not at this file, so without this the module
#: appears to export nothing.
__all__ = [
    "merge_column_attribute",
    "update_column_attribute",
    "find_column_attribute_by_column_id",
    "fetch_attr_column_contexts",
    "find_unlinked_fk_columns",
    "fetch_column_attribute_columns_map",
    "merge_semantic_fk",
    "find_join_path",
]
