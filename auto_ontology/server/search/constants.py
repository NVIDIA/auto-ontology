# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants shared by the global-search router and service."""

from __future__ import annotations

# The only ``text_match_option`` global search implements: every token has to
# appear as a substring. Lives here because the request model's default and the
# service's validation have to agree on the wording.
TEXT_MATCH_CONTAINS = "contains"

# One character is rejected before the database: ``%a%`` matches most of the
# catalog and is never a useful search.
#
# Two characters cannot use the ``gin_trgm_ops`` indexes -- Postgres only
# considers a ``LIKE``/``ILIKE`` pattern indexable when it has three
# consecutive non-wildcard characters, so ``%id%`` seq-scans all ten tables.
# The floor stays at 2 anyway: ``id``, ``BU``, ``fk`` are real catalog queries,
# and the UI's 1000 ms debounce is what stops this firing per keystroke.
# Raising it to 3 would make every query indexable, but would drop those
# identifiers. A two-character search is an accepted seq-scan, not a missed
# optimisation.
MIN_SEARCH_LENGTH = 2

# The value ``filters.tags`` carries to mean "objects with no tags at all",
# alongside the tag ids it otherwise holds.
#
# A sentinel in the list rather than a flag beside it, because that is what the
# control on screen is: the panel offers "(Blanks)" as one more option in the
# tag picker, and a person ticking it beside two tags is asking for either.
# Parenthesised so it cannot collide with an id -- ids are UUIDs.
UNTAGGED_FILTER_VALUE = "(blanks)"
