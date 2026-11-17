# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live, read-only database probing.

:class:`ProbeExecutor` is the single choke point that guarantees probes are
``SELECT``-only, ``LIMIT``-bounded, and capped by a call budget.
"""

from auto_ontology.retrieval.text_to_sql.db_probe.config import (
    DB_PROBE_MAX_CALLS,
    DB_PROBE_MAX_ROWS,
)
from auto_ontology.retrieval.text_to_sql.db_probe.executor import (
    ProbeExecutor,
    is_read_only_select,
)

__all__ = [
    "DB_PROBE_MAX_CALLS",
    "DB_PROBE_MAX_ROWS",
    "ProbeExecutor",
    "is_read_only_select",
]
