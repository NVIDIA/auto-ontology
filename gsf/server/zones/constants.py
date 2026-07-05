# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Zone graph labels and relationship types."""

from __future__ import annotations

REL_ZONE_OF = "ZONE_OF"
# Relationship from an Admin node directly to a catalog item that is not
# connected to any zone.  Keeps the Neo4j graph navigable for data that exists
# in the application but has not yet been assigned to any zone.
REL_HAS_DIRECT_ACCESS = "HAS_DIRECT_ACCESS"
REL_PARTICIPANT_OF = "PARTICIPANT_OF"
LABEL_ZONE = "Zone"
