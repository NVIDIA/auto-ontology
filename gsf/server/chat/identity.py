# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read identities forwarded by the trusted private Next.js gateway."""

from __future__ import annotations

from fastapi import HTTPException, Request

INTERNAL_USER_HEADER = "x-gsf-user-id"


def resolve_internal_user(request: Request, *, required: bool) -> str | None:
    """Return the GSF user id supplied by the trusted private gateway.

    Direct callers that do not use conversation persistence may omit these
    headers. FastAPI must not be publicly reachable because the header is not
    independently authenticated here.
    """

    user_id = (request.headers.get(INTERNAL_USER_HEADER) or "").strip()
    if user_id:
        return user_id

    if required:
        raise HTTPException(
            status_code=401, detail="Authenticated conversation required"
        )
    return None


__all__ = [
    "INTERNAL_USER_HEADER",
    "resolve_internal_user",
]
