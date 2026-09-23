# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query parameters shared by the paged list endpoints.

Kept in one place so every list reads a page the same way, and so the ceiling
below can't drift between routers.
"""

from __future__ import annotations

from fastapi import Query

# Ceiling on a page, so a single request can't pull a whole collection back and
# undo the point of paging. The lists ask for a fraction of this.
MAX_PAGE_SIZE = 100

SKIP_QUERY = Query(
    default=0,
    ge=0,
    description="Rows to skip before the page starts.",
)
LIMIT_QUERY = Query(
    default=None,
    ge=1,
    le=MAX_PAGE_SIZE,
    description=(
        f"Page size, at most {MAX_PAGE_SIZE}. Omit to return every matching row "
        "in one response."
    ),
)
