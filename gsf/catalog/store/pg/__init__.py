# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Postgres implementation of the catalog write path.

Selected when ``GSF_STORE=postgres``. Every module here must expose exactly the
public functions its ``store/neo4j/`` counterpart does — the selector in
``store/<mod>.py`` swaps one for the other, so a missing function is a runtime
AttributeError rather than a type error.

The generic layer lives in :mod:`gsf.catalog.store.pg.registry` (label → table)
and :mod:`gsf.catalog.store.pg.rows` (the upsert primitive). Read those before
the rest: the writers above were built against a property graph, and everything
here is the translation of that assumption.
"""
