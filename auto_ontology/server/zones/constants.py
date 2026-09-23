# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Zone graph labels and relationship types."""

from __future__ import annotations

REL_ZONE_OF = "ZONE_OF"
LABEL_ZONE = "Zone"
# Label applied to a zone node in place of LABEL_ZONE while it is disabled by
# an admin. Swapping the label (rather than storing a boolean property) keeps
# disabled zones out of every plain `:Zone` match used for data access, so
# disabling a zone immediately revokes the catalog access it granted.
LABEL_ZONE_DISABLED = "disableZone"
# Matches a zone node regardless of its current enabled/disabled label.
ZONE_LABEL_PATTERN = f"{LABEL_ZONE}|{LABEL_ZONE_DISABLED}"
