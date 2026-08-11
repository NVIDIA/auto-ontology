# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neo4j implementation of the catalog write path.

Forked verbatim from ``nemo_retriever.tabular_data.ingestion`` in Phase 1 and
pinned to it by ``gsf/catalog/tests/test_fork_parity.py``. Selected when
``GSF_STORE=neo4j`` (the default until Phase 11), which then deletes this
package.
"""
