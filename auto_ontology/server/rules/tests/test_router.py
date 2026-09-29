# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a rule is allowed to be saved as.

Only the search term, because it is the field with two readers that do not
agree. Creating a rule runs the term through ``global_search``, which answers
"nothing matched" for a term it cannot read; replaying one runs it through
``match_selects``, which raises rather than take a rule's labels away over a
term it cannot read. A term that clears the length floor and still tokenises
to nothing falls in the gap: saved happily, then throwing on every nightly
pass afterwards. The route is the only place that can refuse it.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from auto_ontology.server.rules.router import _validated_search_term

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
