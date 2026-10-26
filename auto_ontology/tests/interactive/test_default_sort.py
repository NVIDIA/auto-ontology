# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from auto_ontology.retrieval.interactive.clarify import (
    _DEFAULT_SORT_HINT,
    should_inject_default_sort,
)


def test_named_sort_key_skips_the_metric_default() -> None:
    question = "customers with a risk score above 80, ordered by customer name"
    assert should_inject_default_sort(question) is False


def test_other_sort_key_phrases_skip_the_metric_default() -> None:
    assert should_inject_default_sort("films sorted by title") is False
    assert should_inject_default_sort("films in order of release year") is False
    assert should_inject_default_sort("sort by customer name") is False


def test_explicit_direction_skips_the_metric_default() -> None:
    assert should_inject_default_sort("top 10 customers by risk score") is False
    assert should_inject_default_sort("customers in descending order of spend") is False


def test_metric_default_is_injected_without_a_key_or_direction() -> None:
    assert should_inject_default_sort("customers with a risk score above 80") is True


def test_sort_hint_keeps_a_named_key_ahead_of_the_metric() -> None:
    assert "do not replace it with the metric" in _DEFAULT_SORT_HINT
    assert "neither a sort key nor a direction" in _DEFAULT_SORT_HINT
