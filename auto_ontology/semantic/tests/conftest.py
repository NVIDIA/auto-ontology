# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _no_catalog_description_llm() -> Iterator[None]:
    """``process_table`` describes undescribed columns through the LLM; never here."""
    with (
        patch(
            "auto_ontology.semantic.visit_enter.describe_catalog_columns",
            return_value={},
        ),
        patch(
            "auto_ontology.semantic.visit_enter.store_generated_column_descriptions",
            return_value=set(),
        ),
    ):
        yield
