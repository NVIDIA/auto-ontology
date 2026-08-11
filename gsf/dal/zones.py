# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend selector for ``gsf.dal.zones``.

Resolves to the Neo4j or Postgres implementation once, at import, from
``GSF_STORE``. See :mod:`gsf.infra.store`.

The import list is explicit rather than ``import *``: it *is* the frozen public
surface both implementations must provide, it doubles as a porting checklist,
and a star-import would silently paper over a function the Postgres side has not
implemented yet.

Phase 11 deletes this file and promotes ``pg/zones.py`` in its place.
"""

from gsf.infra.store import USE_PG

if USE_PG:
    from gsf.dal.pg.zones import (  # noqa: F401
        list_zones,
        get_zone_by_id,
        create_zone,
        update_zone,
        delete_zone,
        set_zone_enabled,
    )
else:
    from gsf.dal.neo4j.zones import (  # noqa: F401
        list_zones,
        get_zone_by_id,
        create_zone,
        update_zone,
        delete_zone,
        set_zone_enabled,
    )

__all__ = [
    "list_zones",
    "get_zone_by_id",
    "create_zone",
    "update_zone",
    "delete_zone",
    "set_zone_enabled",
]
