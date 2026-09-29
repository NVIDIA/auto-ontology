# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read identities forwarded by the trusted private Next.js gateway.

Lives at the top of ``auto_ontology.server`` rather than under ``chat``, where it started:
the header is a property of the gateway every router sits behind, and chat was
only the first feature to need a caller's name. Tags read it too, to record who
created and last renamed one.
"""

from __future__ import annotations

from fastapi import HTTPException, Request

INTERNAL_USER_HEADER = "x-auto-ontology-user-id"


def resolve_internal_user(request: Request, *, required: bool) -> str | None:
    """Return the Auto Ontology user id supplied by the trusted private gateway.

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
