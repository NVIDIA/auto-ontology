# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend selector for ``gsf.dal.model_interchange``.

Resolves to the Neo4j or Postgres implementation once, at import, from
``GSF_STORE``. See :mod:`gsf.infra.store`.

The exception types are selected alongside the functions and must be the *same
objects* on both sides — the router catches ``UnknownDatabaseIdsError`` to turn
it into a 404 and ``ModelImportValidationError`` into a 422, and a per-backend
copy would stop matching silently and surface both as 500s.

Phase 11 deletes this file and promotes ``pg/model_interchange.py`` in its place.
"""

from gsf.infra.store import USE_PG

if USE_PG:
    from gsf.dal.pg.model_interchange import (  # noqa: F401
        ModelInterchangeError,
        UnknownDatabaseIdsError,
        ModelImportValidationError,
        validate_database_ids,
        fetch_export_rows,
        assemble_export_document,
        resolve_sql_column_ids,
        apply_import_model,
    )
else:
    from gsf.dal.neo4j.model_interchange import (  # noqa: F401
        ModelInterchangeError,
        UnknownDatabaseIdsError,
        ModelImportValidationError,
        validate_database_ids,
        fetch_export_rows,
        assemble_export_document,
        resolve_sql_column_ids,
        apply_import_model,
    )

__all__ = [
    "ModelInterchangeError",
    "UnknownDatabaseIdsError",
    "ModelImportValidationError",
    "validate_database_ids",
    "fetch_export_rows",
    "assemble_export_document",
    "resolve_sql_column_ids",
    "apply_import_model",
]
