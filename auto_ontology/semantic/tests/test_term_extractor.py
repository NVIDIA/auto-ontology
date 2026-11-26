# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for term extraction sanitization."""

from __future__ import annotations

import pytest

from auto_ontology.semantic.models import (
    ColumnAttributeSpec,
    RawTableTermsResult,
    RawTermProposal,
    TermColumnRef,
)
from auto_ontology.semantic import term_extractor
from auto_ontology.semantic.term_extractor import _sanitize_result


def test_sanitize_uses_llm_display_names() -> None:
    specs = [
        ColumnAttributeSpec(
            source_column="amount",
            name="amount",
            datatype="numeric",
        ),
        ColumnAttributeSpec(
            source_column="status",
            name="status",
            datatype="text",
        ),
    ]
    raw = RawTableTermsResult(
        terms=[
            RawTermProposal(
                name="PurchaseOrder",
                description="A purchase order",
                attributes=[
                    TermColumnRef(
                        source_column="amount",
                        display_name="TotalAmount",
                    )
                ],
            )
        ]
    )

    result = _sanitize_result(raw, table={"name": "purchase_orders"}, specs=specs)

    assert len(result.terms) == 1
    assert result.terms[0].name == "Purchase Order"
    assigned = {a.source_column: a.display_name for a in result.terms[0].attributes}
    assert assigned == {"amount": "Total Amount", "status": "status"}
    assert specs[0].display_name == "Total Amount"
    assert specs[1].display_name == "status"


def test_sanitize_assigns_orphan_specs_to_primary_term() -> None:
    specs = [
        ColumnAttributeSpec(
            source_column="amount",
            name="amount",
            datatype="numeric",
        ),
        ColumnAttributeSpec(
            source_column="status",
            name="status",
            datatype="text",
        ),
    ]
    raw = RawTableTermsResult(
        terms=[
            RawTermProposal(
                name="Purchase Order",
                description="A purchase order",
                attributes=[
                    TermColumnRef(
                        source_column="amount",
                        display_name="Total Amount",
                    ),
                    TermColumnRef(
                        source_column="status",
                        display_name="Order Status",
                    ),
                ],
            )
        ]
    )

    result = _sanitize_result(raw, table={"name": "purchase_orders"}, specs=specs)

    assigned = {a.source_column: a.display_name for a in result.terms[0].attributes}
    assert assigned == {
        "amount": "Total Amount",
        "status": "Order Status",
    }
    assert specs[0].display_name == "Total Amount"
    assert specs[1].display_name == "Order Status"


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Customer Order", "Customer Order"),
        ("Numéro Client", "Numéro Client"),
        ("受注", "受注"),
        ("주문 내역", "주문 내역"),
        # Still rejected, as before: lowercase start, double space, leading digit.
        ("purchase order", "Fallback"),
        ("Order  Date", "Fallback"),
        ("2nd Order", "Fallback"),
        ("unknown", "Fallback"),
    ],
)
def test_llm_labels_are_accepted_in_any_script(label: str, expected: str) -> None:
    assert term_extractor._normalize_label(label, fallback="Fallback") == expected


# --------------------------------------------------------------------------
# Placeholder Terms say they are placeholders (#533)
# --------------------------------------------------------------------------


def _specs() -> list[ColumnAttributeSpec]:
    return [
        ColumnAttributeSpec(source_column="adufuturehorizon", name="adufuturehorizon"),
        ColumnAttributeSpec(source_column="ddlotsize", name="ddlotsize"),
    ]


def test_a_failed_model_call_yields_a_placeholder_that_says_why(monkeypatch) -> None:
    monkeypatch.setattr(term_extractor, "get_llm_client", lambda **_: object())
    monkeypatch.setattr(
        term_extractor, "invoke_with_structured_output", lambda *a, **k: None
    )

    result = term_extractor.extract_term({"name": "ibp_location_product"}, {}, _specs())

    assert [term.name for term in result.terms] == ["Ibp Location Product"]
    assert result.fallback_reason == "the model returned no structured result"


def test_a_proposal_with_no_terms_yields_a_placeholder() -> None:
    """A proposed Term with a bad name is renamed after the table, not dropped,
    so only an empty proposal reaches this fallback."""
    raw = RawTableTermsResult(terms=[])

    result = _sanitize_result(raw, table={"name": "orders"}, specs=_specs())

    assert result.fallback_reason == "no proposed Term survived validation"


def test_no_candidate_columns_yields_a_placeholder() -> None:
    result = term_extractor.extract_term({"name": "orders"}, {}, [])

    assert result.fallback_reason == "no candidate columns"


def test_a_real_proposal_is_not_marked_as_a_placeholder() -> None:
    raw = RawTableTermsResult(
        terms=[
            RawTermProposal(
                name="Location Product",
                description="A product stocked at a location",
                attributes=[
                    TermColumnRef(source_column="ddlotsize", display_name="Lot Size")
                ],
            )
        ]
    )

    result = _sanitize_result(
        raw, table={"name": "ibp_location_product"}, specs=_specs()
    )

    assert result.fallback_reason is None
