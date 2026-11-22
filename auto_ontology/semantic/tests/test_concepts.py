# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which tables ``resolve_concepts`` folds into one Concept, and which it refuses."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

from auto_ontology.semantic import concepts
from auto_ontology.semantic.models import ConceptMergeDecision

_CODES = [f"C{n:05d}" for n in range(20)]


def _row(
    table: str,
    column: str,
    *,
    values: list[Any] = _CODES,
    family: str | None = None,
    fk_target: str | None = None,
    is_pk: bool = True,
) -> dict[str, Any]:
    return {
        "concept_id": f"concept-{table}",
        "concept_name": table.title(),
        "term_id": f"term-{table}",
        "term_name": table.title(),
        "term_description": f"The {table}.",
        "table_id": f"table-{table}",
        "table_name": table,
        "table_description": None,
        "schema_name": "sales",
        "column_id": f"{table}.{column}",
        "column_name": column,
        "column_description": None,
        "data_type": "text",
        "distinct_values": json.dumps(values),
        "is_pk": is_pk,
        "fk_target_column_id": fk_target,
        "family_id": family,
    }


@contextmanager
def _resolving(
    rows: list[dict[str, Any]],
    decision: ConceptMergeDecision | None = None,
):
    """Run against *rows*, each table its own Concept, with *decision* from the LLM."""
    decision = decision or ConceptMergeDecision(same_entity=True, name="Customer")
    with (
        patch.object(concepts, "fetch_concept_merge_candidates", return_value=rows),
        patch.object(
            concepts,
            "fetch_table_concept_id",
            side_effect=lambda table_id: f"concept-{table_id.removeprefix('table-')}",
        ),
        patch.object(
            concepts,
            "ensure_concept_key",
            side_effect=lambda concept_id, column_id: f"key-{column_id}",
        ),
        patch.object(concepts, "fetch_concept_fields", return_value=[]),
        patch.object(concepts, "is_semantic_fk_db_query_prevented", return_value=False),
        patch.object(concepts, "_resolve_connector", return_value=None),
        patch.object(concepts, "get_llm_client", return_value=object()),
        patch.object(
            concepts, "invoke_with_structured_output", return_value=decision
        ) as llm,
        patch.object(concepts, "merge_concepts", return_value=True) as merge,
    ):
        yield merge, llm


def test_a_split_entity_is_merged_under_the_llms_name() -> None:
    rows = [_row("customer", "customer_id"), _row("customer_ext", "customer_id")]
    with _resolving(rows) as (merge, _):
        assert concepts.resolve_concepts("db") == 1

    merge.assert_called_once_with(
        "concept-customer",
        "concept-customer_ext",
        key_pairs={"key-customer_ext.customer_id": "key-customer.customer_id"},
        name="Customer",
    )


def test_tables_in_different_families_are_never_paired() -> None:
    rows = [
        _row("customer", "customer_id", family="OLD_"),
        _row("customer_ext", "customer_id", family="NEW_"),
    ]
    with _resolving(rows) as (merge, llm):
        assert concepts.resolve_concepts("db") == 0

    llm.assert_not_called()
    merge.assert_not_called()


def test_a_table_in_a_family_is_not_paired_with_one_outside_every_family() -> None:
    rows = [
        _row("customer", "customer_id", family="NEW_"),
        _row("customer_ext", "customer_id"),
    ]
    with _resolving(rows) as (merge, _):
        assert concepts.resolve_concepts("db") == 0
    merge.assert_not_called()


def test_one_way_containment_is_a_reference_not_a_split() -> None:
    """Every order's customer exists, but most customers have no order."""
    rows = [
        _row("customer", "customer_id"),
        _row("vip", "customer_id", values=_CODES[:5]),
    ]
    with _resolving(rows) as (merge, llm):
        assert concepts.resolve_concepts("db") == 0

    llm.assert_not_called()
    merge.assert_not_called()


def test_the_llm_can_refuse_two_entities_that_share_an_id() -> None:
    rows = [_row("orders", "order_id"), _row("shipments", "order_id")]
    refusal = ConceptMergeDecision(same_entity=False, rationale="A shipment.")
    with _resolving(rows, refusal) as (merge, llm):
        assert concepts.resolve_concepts("db") == 0

    llm.assert_called_once()
    merge.assert_not_called()


def test_a_bare_id_needs_a_declared_fk() -> None:
    rows = [_row("customer", "id"), _row("product", "id")]
    with _resolving(rows) as (merge, llm):
        assert concepts.resolve_concepts("db") == 0
    llm.assert_not_called()
    merge.assert_not_called()


def test_a_satellite_folds_into_the_table_its_key_references() -> None:
    """The declared FK's target is the core table, whatever the names sort to."""
    rows = [
        _row("a_profile", "customer_ref", fk_target="z_customer.id"),
        _row("z_customer", "id"),
    ]
    with _resolving(rows) as (merge, _):
        assert concepts.resolve_concepts("db") == 1

    survivor, absorbed = merge.call_args.args
    assert (survivor, absorbed) == ("concept-z_customer", "concept-a_profile")
    assert merge.call_args.kwargs["key_pairs"] == {
        "key-a_profile.customer_ref": "key-z_customer.id"
    }


def test_short_values_merge_only_on_a_declared_fk() -> None:
    """Ids ``1..20`` are in every surrogate key; a shared name is not evidence."""
    small = list(range(1, 21))
    named = [
        _row("customer", "customer_id", values=small),
        _row("customer_ext", "customer_id", values=small),
    ]
    with _resolving(named) as (merge, _):
        assert concepts.resolve_concepts("db") == 0
    merge.assert_not_called()

    declared = [
        _row("customer", "customer_id", values=small),
        _row(
            "customer_ext",
            "customer_id",
            values=small,
            fk_target="customer.customer_id",
        ),
    ]
    with _resolving(declared) as (merge, _):
        assert concepts.resolve_concepts("db") == 1


def test_tables_already_in_one_concept_are_left_alone() -> None:
    rows = [_row("customer", "customer_id"), _row("customer_ext", "customer_id")]
    with (
        _resolving(rows) as (merge, llm),
        patch.object(concepts, "fetch_table_concept_id", return_value="concept-1"),
    ):
        assert concepts.resolve_concepts("db") == 0
    llm.assert_not_called()
    merge.assert_not_called()


def test_a_failing_join_probe_refuses_the_merge() -> None:
    rows = [_row("customer", "customer_id"), _row("customer_ext", "customer_id")]
    with (
        _resolving(rows) as (merge, llm),
        patch.object(concepts, "_resolve_connector", return_value=MagicMock()),
        patch.object(concepts, "join_probe_returns_rows", return_value=False),
    ):
        assert concepts.resolve_concepts("db") == 0
    llm.assert_not_called()
    merge.assert_not_called()


def test_no_llm_means_no_merge() -> None:
    rows = [_row("customer", "customer_id"), _row("customer_ext", "customer_id")]
    with (
        _resolving(rows) as (merge, _),
        patch.object(concepts, "get_llm_client", side_effect=EnvironmentError),
    ):
        assert concepts.resolve_concepts("db") == 0
    merge.assert_not_called()


def test_no_merge_when_the_shared_column_cannot_be_a_key() -> None:
    """A merge without a shared key would leave a keyless Concept behind."""
    rows = [_row("customer", "birth_date"), _row("customer_ext", "birth_date")]
    with (
        _resolving(rows) as (merge, _),
        patch.object(concepts, "ensure_concept_key", return_value=None),
    ):
        assert concepts.resolve_concepts("db") == 0
    merge.assert_not_called()


def test_a_shared_name_pairs_a_tables_pk_when_it_has_one() -> None:
    """``birth_date`` sorts before ``customer_id`` but is not the identity."""
    rows = [
        _row("customer", "birth_date", is_pk=False),
        _row("customer", "customer_id"),
        _row("customer_ext", "birth_date", is_pk=False),
        _row("customer_ext", "customer_id"),
    ]
    with _resolving(rows) as (merge, _):
        assert concepts.resolve_concepts("db") == 1

    assert merge.call_args.kwargs["key_pairs"] == {
        "key-customer_ext.customer_id": "key-customer.customer_id"
    }


def test_a_non_pk_column_is_not_paired_by_name_on_a_table_with_a_pk() -> None:
    rows = [
        _row("customer", "customer_id"),
        _row("customer", "email", is_pk=False),
        _row("customer_ext", "ext_id"),
        _row("customer_ext", "email", is_pk=False),
    ]
    with _resolving(rows) as (merge, llm):
        assert concepts.resolve_concepts("db") == 0
    llm.assert_not_called()
    merge.assert_not_called()


def test_a_table_without_a_pk_pairs_its_unique_columns_by_name() -> None:
    rows = [
        _row("customer", "customer_id"),
        _row("customer_ext", "customer_id", is_pk=False),
    ]
    with _resolving(rows) as (merge, _):
        assert concepts.resolve_concepts("db") == 1
