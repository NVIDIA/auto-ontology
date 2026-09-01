# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enforces the connector dialect contract.

Callers pass ``connector.dialect`` straight into sqlglot with no translation
layer, and sqlglot raises on an unrecognised dialect *name* before it even
looks at the SQL. This test is what keeps that contract honest, so a new
connector reporting an engine name sqlglot doesn't know fails here rather than
at query time.
"""

from __future__ import annotations

import pytest
from sqlglot.dialects import DIALECTS

from gsf.connectors.registry import CONNECTOR_REGISTRY

_KNOWN_DIALECTS = {name.lower() for name in DIALECTS}


@pytest.mark.parametrize("scheme", sorted(CONNECTOR_REGISTRY))
def test_registered_connector_reports_a_dialect_sqlglot_knows(scheme: str) -> None:
    # Read off the class: instantiating a connector would open a connection.
    dialect = CONNECTOR_REGISTRY[scheme].dialect.fget(None)  # type: ignore[attr-defined]

    assert isinstance(dialect, str) and dialect, f"{scheme} reports no dialect"
    assert dialect in _KNOWN_DIALECTS, (
        f"{scheme} reports {dialect!r}, which sqlglot does not know; map it onto "
        f"the closest supported dialect the way HeavyDBDatabase does"
    )
