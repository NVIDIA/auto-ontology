# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Postgres implementation of the GSF data access layer.

Sits alongside the Neo4j implementation during the migration. Which one the
application uses is decided once, at import time, by ``GSF_STORE`` — see
``docs/refactor/drop-neo4j/PLAN.md`` § "Keeping main green".

Every module here must expose **exactly** the public functions its Neo4j
counterpart does, with identical signatures. That is enforced by
``gsf/dal/tests/test_dal_surface.py``, and it is what makes the cutover a
one-line change rather than an application-wide refactor.
"""
