# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CustomAnalysis write orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from gsf.server.custom_analyses import service


@patch("gsf.server.custom_analyses.service.embed_custom_analyses")
@patch("gsf.server.custom_analyses.service.fetch_database_names_for_analysis")
@patch("gsf.vdb.get_semantic_vdb")
@patch("gsf.utils.get_embed_params")
def test_embed_scopes_the_row_to_the_referenced_database(
    _embed_params: MagicMock,
    _vdb: MagicMock,
    fetch_databases: MagicMock,
    embed: MagicMock,
) -> None:
    fetch_databases.return_value = ["WWI"]

    service._embed_analysis("analysis-1")

    assert embed.call_count == 1
    assert embed.call_args.kwargs["database_name"] == "WWI"
    assert embed.call_args.kwargs["analysis_id"] == "analysis-1"


@patch("gsf.server.custom_analyses.service.embed_custom_analyses")
@patch("gsf.server.custom_analyses.service.fetch_database_names_for_analysis")
@patch("gsf.vdb.get_semantic_vdb")
@patch("gsf.utils.get_embed_params")
def test_cross_database_analysis_is_embedded_once_per_database(
    _embed_params: MagicMock,
    _vdb: MagicMock,
    fetch_databases: MagicMock,
    embed: MagicMock,
) -> None:
    fetch_databases.return_value = ["WWI", "SUPERSTORE"]

    service._embed_analysis("analysis-2")

    scoped = [call.kwargs["database_name"] for call in embed.call_args_list]
    assert sorted(scoped) == ["SUPERSTORE", "WWI"]


@patch("gsf.server.custom_analyses.service.embed_custom_analyses")
@patch("gsf.server.custom_analyses.service.fetch_database_names_for_analysis")
@patch("gsf.vdb.get_semantic_vdb")
@patch("gsf.utils.get_embed_params")
def test_unresolvable_database_still_embeds_but_warns(
    _embed_params: MagicMock,
    _vdb: MagicMock,
    fetch_databases: MagicMock,
    embed: MagicMock,
) -> None:
    fetch_databases.return_value = []

    service._embed_analysis("analysis-3")

    assert embed.call_count == 1
    assert embed.call_args.kwargs["database_name"] is None
