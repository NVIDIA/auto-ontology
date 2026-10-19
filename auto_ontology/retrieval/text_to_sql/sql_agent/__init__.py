# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentic SQL construction.

Replaces the single structured LLM call in ``construct_sql_from_candidates``
with a bounded tool-calling loop: the model reads the prepared context, may
query the database and run the static validators against its draft, and then
commits. It returns the same :class:`SQLGenerationModel` the single call did,
so everything downstream of construction is unchanged.
"""
