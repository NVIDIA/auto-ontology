# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities used across server, ingestion, and dev tools."""

from gsf.utils.embedding import get_embed_params
from gsf.utils.retriever import get_retriever

__all__ = ["get_embed_params", "get_retriever"]
