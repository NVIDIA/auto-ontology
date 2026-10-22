# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collapsing things the transcript named more than one way."""

from __future__ import annotations

from auto_ontology.conversations.semantic.consolidate import (
    consolidate_graph,
    plan_merges,
)
from auto_ontology.conversations.semantic.extract import (
    EdgeDraft,
    GraphDraft,
    NodeDraft,
    normalise_name,
)


def _thing(name: str, type_: str = "Feature", **kwargs) -> NodeDraft:
    return NodeDraft(name=name, type=type_, kinds=["object"], **kwargs)


def _statement(name: str, key: str, value: str) -> NodeDraft:
    return NodeDraft(
        name=name, type="Metric", kinds=["fact", "state"], key=key, value=value
    )


def _vectors(**named: list[float]) -> dict[str, list[float]]:
    return {normalise_name(name): vector for name, vector in named.items()}


# ---------------------------------------------------------------------------
# What may merge
# ---------------------------------------------------------------------------


def test_two_names_for_one_thing_collapse() -> None:
    nodes = [_thing("API module"), _thing("API integration module")]
    vectors = _vectors(API_module=[1.0, 0.0], API_integration_module=[0.99, 0.14])

    merges = plan_merges(nodes, [], vectors)

    assert len(merges) == 1
    assert merges[0].duplicates == ("api module",)


def test_distant_names_are_left_alone() -> None:
    nodes = [_thing("Autocomplete"), _thing("Deployment pipeline")]
    vectors = _vectors(Autocomplete=[1.0, 0.0], Deployment_pipeline=[0.0, 1.0])

    assert plan_merges(nodes, [], vectors) == []


def test_a_statement_is_never_a_merge_candidate() -> None:
    """Two readings of one property are a revision, and merging them would
    delete it rather than record it."""
    nodes = [
        _statement("coverage 78", "test_coverage", "78%"),
        _statement("coverage 85", "test_coverage", "85%"),
    ]
    vectors = _vectors(coverage_78=[1.0, 0.0], coverage_85=[1.0, 0.0])

    assert plan_merges(nodes, [], vectors) == []


def test_things_of_different_types_do_not_merge() -> None:
    """A Feature and the Person who asked for it can have similar names."""
    nodes = [_thing("Christina", "Person"), _thing("Christina's app", "Project")]
    vectors = _vectors(Christina=[1.0, 0.0], Christinas_app=[1.0, 0.0])

    assert plan_merges(nodes, [], vectors) == []


def test_timestamps_are_protected() -> None:
    """Two dates embed almost identically and are the ends of an interval."""
    nodes = [_thing("March 10 2024", "Timestamp"), _thing("March 12 2024", "Timestamp")]
    vectors = _vectors(March_10_2024=[1.0, 0.0], March_12_2024=[1.0, 0.0])

    assert plan_merges(nodes, [], vectors) == []


def test_a_node_without_a_vector_cannot_merge() -> None:
    nodes = [_thing("API module"), _thing("API integration module")]

    assert plan_merges(nodes, [], _vectors(API_module=[1.0, 0.0])) == []


# ---------------------------------------------------------------------------
# Which survives
# ---------------------------------------------------------------------------


def test_the_most_connected_name_is_kept() -> None:
    """It is the one most of the graph already points at, so the merge rewrites
    the fewest edges."""
    nodes = [_thing("API module"), _thing("API integration module")]
    edges = [
        EdgeDraft(source="api module", target="weather app", role="part_of"),
        EdgeDraft(
            source="api integration module", target="weather app", role="part_of"
        ),
        EdgeDraft(source="api integration module", target="jest", role="tested_with"),
    ]
    vectors = _vectors(API_module=[1.0, 0.0], API_integration_module=[0.99, 0.14])

    assert plan_merges(nodes, edges, vectors)[0].canonical == "api integration module"


def test_the_longer_name_breaks_a_tie() -> None:
    """The more specific of two names is the safer thing to show a reader."""
    nodes = [_thing("API module"), _thing("API integration module")]
    vectors = _vectors(API_module=[1.0, 0.0], API_integration_module=[0.99, 0.14])

    assert plan_merges(nodes, [], vectors)[0].canonical == "api integration module"


def test_aliases_chain_through_single_link() -> None:
    """ "API module" and "the API integration module" need not be close to each
    other if both are close to the name between them."""
    nodes = [
        _thing("API module"),
        _thing("API integration module"),
        _thing("the API integration module"),
    ]
    vectors = _vectors(
        API_module=[1.0, 0.0],
        API_integration_module=[0.98, 0.2],
        the_API_integration_module=[0.93, 0.37],
    )

    merges = plan_merges(nodes, [], vectors)

    assert len(merges) == 1
    assert len(merges[0].duplicates) == 2


# ---------------------------------------------------------------------------
# Applying it
# ---------------------------------------------------------------------------


def test_edges_are_repointed_onto_the_canonical() -> None:
    """The duplicate's edge has to survive the merge, pointing at the survivor:
    it is half the subgraph a question matching either name should reach."""
    graph = GraphDraft(
        nodes=[_thing("API module"), _thing("API integration module")],
        edges=[
            # Two edges, so the longer name also has the higher degree and
            # wins the canonical slot outright.
            EdgeDraft(
                source="api integration module", target="weather app", role="part_of"
            ),
            EdgeDraft(source="api module", target="jest", role="tested_with"),
        ],
    )
    vectors = _vectors(API_module=[1.0, 0.0], API_integration_module=[0.99, 0.14])

    consolidate_graph(graph, vectors)

    assert [n.name for n in graph.nodes] == ["API integration module"]
    assert {(e.source, e.role) for e in graph.edges} == {
        ("api integration module", "part_of"),
        ("api integration module", "tested_with"),
    }


def test_an_edge_that_becomes_a_self_loop_is_dropped() -> None:
    """ "The API module is part of the API integration module" says nothing once
    those are one node."""
    graph = GraphDraft(
        nodes=[_thing("API module"), _thing("API integration module")],
        edges=[
            EdgeDraft(
                source="api module", target="api integration module", role="part_of"
            )
        ],
    )
    vectors = _vectors(API_module=[1.0, 0.0], API_integration_module=[0.99, 0.14])

    consolidate_graph(graph, vectors)

    assert graph.edges == []


def test_duplicate_edges_fold_and_keep_both_sets_of_turns() -> None:
    """A relation stated about both spellings is evidenced by both."""
    graph = GraphDraft(
        nodes=[_thing("API module"), _thing("API integration module")],
        edges=[
            EdgeDraft(
                source="api module", target="jest", role="tested_with", turn_indexes={3}
            ),
            EdgeDraft(
                source="api integration module",
                target="jest",
                role="tested_with",
                turn_indexes={57},
            ),
        ],
    )
    vectors = _vectors(API_module=[1.0, 0.0], API_integration_module=[0.99, 0.14])

    consolidate_graph(graph, vectors)

    assert len(graph.edges) == 1
    assert graph.edges[0].turn_indexes == {3, 57}


def test_the_canonical_absorbs_the_duplicates_turns() -> None:
    """A question matching either name must reach everything either cited."""
    graph = GraphDraft(
        nodes=[
            _thing("API module", turn_indexes={3}),
            _thing("API integration module", turn_indexes={57, 64}),
        ],
        edges=[],
    )
    vectors = _vectors(API_module=[1.0, 0.0], API_integration_module=[0.99, 0.14])

    consolidate_graph(graph, vectors)

    assert graph.nodes[0].turn_indexes == {3, 57, 64}
