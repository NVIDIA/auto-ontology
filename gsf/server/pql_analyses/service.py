# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PqlAnalysis write orchestration (the predictive twin of custom_analyses/service.py).

All direct store access lives in ``gsf/dal/pql_analyses.py``. This module keeps the
orchestration: uniqueness checks, node persistence, and VDB embedding. Unlike
the SQL variant there is no SQL parse/validate step — a PQL query resolves its tables
against the prediction graph at predict time, so it is stored as-is.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from gsf.dal.pql_analyses import (
    PqlAnalysisNameConflict,
    PqlAnalysisPqlConflict,
    delete_pql_analysis_node,
    embed_pql_analyses,
    find_pql_analysis_by_name,
    find_pql_analysis_by_pql,
    get_pql_analysis_by_id,
    list_pql_analyses,
    upsert_pql_analysis_node,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PqlAnalysisNameConflict",
    "PqlAnalysisPqlConflict",
    "list_pql_analyses",
    "create_pql_analysis",
    "update_pql_analysis",
    "delete_pql_analysis",
]


def _embed(analysis_id: str) -> None:
    from gsf.utils import get_embed_params
    from gsf.vdb import get_semantic_vdb

    embed_pql_analyses(
        embed_params=get_embed_params(),
        vdb=get_semantic_vdb(),
        analysis_id=analysis_id,
    )


def create_pql_analysis(
    name: str,
    description: str,
    pql: str,
) -> dict[str, Any]:
    """Create a fresh ``PqlAnalysis`` and embed it.

    Raises :class:`PqlAnalysisNameConflict` when ``name`` is already used and
    :class:`PqlAnalysisPqlConflict` when ``pql`` is already attached. Returns
    ``{id, name, description, pql}``.
    """
    name_conflict = find_pql_analysis_by_name(name, exclude_id=None)
    if name_conflict is not None:
        raise PqlAnalysisNameConflict(
            f"another PqlAnalysis already uses name {name!r} "
            f"(id={name_conflict['id']!r})",
        )

    pql_conflict = find_pql_analysis_by_pql(pql, exclude_id=None)
    if pql_conflict is not None:
        raise PqlAnalysisPqlConflict(
            f"this PQL is already used by PqlAnalysis {pql_conflict['name']!r} "
            f"(id={pql_conflict['id']!r})",
        )

    analysis_id = str(uuid.uuid4())
    upsert_pql_analysis_node(analysis_id, name, description, pql)
    _embed(analysis_id)

    return {"id": analysis_id, "name": name, "description": description, "pql": pql}


def update_pql_analysis(
    analysis_id: str,
    name: str,
    description: str,
    pql: str,
) -> dict[str, Any] | None:
    """Replace name/description/pql of an existing ``PqlAnalysis`` by id.

    Returns the updated row or ``None`` when no analysis with ``analysis_id`` exists.
    Raises :class:`PqlAnalysisNameConflict` / :class:`PqlAnalysisPqlConflict`.
    """
    if get_pql_analysis_by_id(analysis_id) is None:
        return None

    name_conflict = find_pql_analysis_by_name(name, exclude_id=analysis_id)
    if name_conflict is not None:
        raise PqlAnalysisNameConflict(
            f"another PqlAnalysis already uses name {name!r} "
            f"(id={name_conflict['id']!r})",
        )

    pql_conflict = find_pql_analysis_by_pql(pql, exclude_id=analysis_id)
    if pql_conflict is not None:
        raise PqlAnalysisPqlConflict(
            f"this PQL is already used by PqlAnalysis {pql_conflict['name']!r} "
            f"(id={pql_conflict['id']!r})",
        )

    upsert_pql_analysis_node(analysis_id, name, description, pql)

    from gsf.vdb import get_semantic_vdb

    get_semantic_vdb().delete_by_id(analysis_id)
    _embed(analysis_id)

    return {"id": analysis_id, "name": name, "description": description, "pql": pql}


def delete_pql_analysis(analysis_id: str) -> dict[str, str] | None:
    """Remove a PqlAnalysis and its VDB embedding.

    Returns ``{"id": analysis_id}`` on success, or ``None`` when not found.
    """
    if get_pql_analysis_by_id(analysis_id) is None:
        return None

    delete_pql_analysis_node(analysis_id)

    from gsf.vdb import get_semantic_vdb

    get_semantic_vdb().delete_by_id(analysis_id)

    return {"id": analysis_id}
