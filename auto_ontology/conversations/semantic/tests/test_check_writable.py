# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pre-write check names every node the table would reject."""

from __future__ import annotations

import pytest

from auto_ontology.conversations.semantic.extract import NodeDraft
from auto_ontology.conversations.semantic.pipeline import _check_writable


def test_a_name_used_twice_identifies_both_nodes() -> None:
    nodes = [
        NodeDraft(
            name="Hub cap",
            type="Metric",
            kinds=["fact"],
            key="cap",
            value="3000",
            turn_indexes={2},
        ),
        NodeDraft(name="hub cap", type="Project", kinds=["object"], turn_indexes={9}),
    ]

    with pytest.raises(ValueError) as raised:
        _check_writable(nodes)

    message = str(raised.value)
    assert "'Hub cap' (Metric, cap = 3000, turns [2])" in message
    assert "'hub cap' (Project, turns [9])" in message


def test_distinct_names_pass() -> None:
    _check_writable(
        [
            NodeDraft(name="Hub", type="Place", kinds=["object"]),
            NodeDraft(name="Hub cap", type="Metric", kinds=["fact"]),
        ]
    )
