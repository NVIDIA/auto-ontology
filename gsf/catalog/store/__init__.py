# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The catalog's persistence layer — **the only subtree Phase 4 rewrites**.

Everything above this package is storage-agnostic and forks once, permanently.
Keep it that way: no Cypher, no driver types, and no ``neo4j`` import above
``gsf/catalog/store/``.
"""
