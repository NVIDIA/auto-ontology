# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a rule is allowed to be saved as.

Two fields, each because a rule outlives the request that made it and the
search cannot see that on its own.

The search term is the one with two readers that do not agree. Creating a rule
runs the term through ``global_search``, which answers "nothing matched" for a
term it cannot read; replaying one runs it through ``match_selects``, which
raises rather than take a rule's labels away over a term it cannot read. A term
that clears the length floor and still tokenises to nothing falls in the gap:
saved happily, then throwing on every nightly pass afterwards.

The tag filter's untagged sentinel is the one filter a rule's own writes can
falsify — see ``_validated_filters``. Both are perfectly good *searches*, which
is why the route is the only place either can be refused.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from auto_ontology.server.rules.router import (
    _validated_filters,
    _validated_search_term,
)
from auto_ontology.server.search.constants import UNTAGGED_FILTER_VALUE
from auto_ontology.server.search.router import GlobalSearchFilters

#: Terms made only of characters the tokeniser treats as separators. Long
#: enough to clear the floor, which is what makes them worth a second check.
TOKENLESS = ["**", "--", "##", "%%%", "  **  "]


@pytest.mark.parametrize("term", TOKENLESS)
def test_a_term_that_is_only_separators_is_refused(term: str) -> None:
    with pytest.raises(HTTPException) as refused:
        _validated_search_term(term)

    assert refused.value.status_code == 400
    assert "searchable" in refused.value.detail


def test_the_length_floor_still_answers_for_short_terms() -> None:
    """A one-character term is refused for being short, not for being empty.

    Both checks would reject ``"*"``. The floor has to answer first, so the
    message tells the caller the thing they can act on.
    """
    with pytest.raises(HTTPException) as refused:
        _validated_search_term("*")

    assert "at least" in refused.value.detail


@pytest.mark.parametrize("term", ["revenue", "  revenue  ", "cust_id", "q3-2026"])
def test_a_term_with_anything_searchable_in_it_is_kept(term: str) -> None:
    """Including terms that are *mostly* separators, which are still usable."""
    assert _validated_search_term(term) == term.strip()


def test_a_rule_filtering_on_untagged_objects_is_refused() -> None:
    """A rule that tags what has no tags would undo itself on the next pass."""
    with pytest.raises(HTTPException) as refused:
        _validated_filters(GlobalSearchFilters(tags=[UNTAGGED_FILTER_VALUE]))

    assert refused.value.status_code == 400
    assert UNTAGGED_FILTER_VALUE in refused.value.detail


def test_the_sentinel_is_refused_even_beside_real_tags() -> None:
    """The union is what makes it reachable: any one entry is enough for a hit.

    So a rule carrying the sentinel alongside two tags still matches untagged
    objects, still labels them, and still stops matching them afterwards.
    """
    with pytest.raises(HTTPException):
        _validated_filters(GlobalSearchFilters(tags=["t1", UNTAGGED_FILTER_VALUE]))


@pytest.mark.parametrize("tags", [None, [], ["t1"], ["t1", "t2"]])
def test_a_rule_filtering_on_real_tags_is_kept(tags: list[str] | None) -> None:
    """ "Label everything already tagged PII" stays true however often it runs."""
    filters = GlobalSearchFilters(tags=tags)
    assert _validated_filters(filters) is filters
