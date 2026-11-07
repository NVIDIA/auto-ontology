# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The calibration tool's arithmetic, its labels, and one end-to-end run.

The metrics are pinned on hand-built results because a sign error in a gate
sweep would still print a plausible table. The labels are checked against the
seeded catalog because a typo would score as a miss rather than fail. The
end-to-end run uses the fake embedder: its numbers mean nothing, but both
semantic modes embed with it, so they must agree exactly -- which is the
parity check the real run reports.
"""

from __future__ import annotations

import os
import uuid
from typing import Iterator

import psycopg
import pytest

from auto_ontology.infra.postgres import get_postgres_connection_string
from dev_tools.calibrate_vdb import (
    Entity,
    Run,
    best_point,
    calibrate,
    first_correct_rank,
    gate_sweep,
    load_entities,
    parity,
    ranking_metrics,
    render,
    resolve_targets,
)
from dev_tools.fake_embedder import FakeEmbedder


def _entity(
    text: str, expect: tuple[str, ...] = ("x",), kind: str = "paraphrase"
) -> Entity:
    return Entity(
        entity=text,
        layer="data",
        label="Column",
        database="db",
        kind=kind,
        expect=expect,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_the_first_correct_rank_is_one_based() -> None:
    rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]

    assert first_correct_rank(rows, {"b", "c"}) == 2
    assert first_correct_rank(rows, {"z"}) is None


def test_ranking_metrics() -> None:
    m = ranking_metrics([1, 2, None, 4], top_k=10)

    assert m["recall@1"] == 0.25
    assert m["recall@3"] == 0.5
    assert m["recall@10"] == 0.75
    assert m["mrr"] == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)


def _outcomes() -> list:
    return [
        # Right answer ranked first, at distance 0.2 / rrf 1.0.
        (_entity("good"), [{"id": "x", "score": 0.2, "rrf": 1.0}], {"x"}),
        # Wrong row first, at 0.5 / 0.6.
        (_entity("bad"), [{"id": "y", "score": 0.5, "rrf": 0.6}], {"x"}),
        # No right answer; nearest row at 0.8 / 0.4.
        (_entity("none", expect=()), [{"id": "y", "score": 0.8, "rrf": 0.4}], set()),
    ]


def test_a_cosine_gate_accepts_at_or_below_the_threshold() -> None:
    [tight, mid, loose] = gate_sweep(_outcomes(), "cosine", [0.3, 0.5, 0.9])

    assert (tight.correct, tight.wrong, tight.negatives) == (1, 0, 0)
    assert (mid.correct, mid.wrong, mid.negatives) == (1, 1, 0)
    assert (loose.correct, loose.wrong, loose.negatives) == (1, 1, 1)
    assert tight.precision == 1.0 and tight.recall == 0.5
    assert loose.negative_accept_rate == 1.0


def test_the_hybrid_gate_accepts_a_row_named_after_the_entity() -> None:
    named = {"id": "x", "name": "rental_rate", "rrf": 1.0, "semantic_distance": 0.9}
    outcomes = [
        (_entity("rental rate"), [{**named, "query_entity": "rental rate"}], {"x"}),
        (_entity("film fee"), [{**named, "query_entity": "film fee"}], {"x"}),
    ]

    [named_gate] = gate_sweep(outcomes, "name-or-distance", [0.65])
    [distance_gate] = gate_sweep(outcomes, "vector-distance", [0.65])

    assert (named_gate.correct, named_gate.accepted) == (1, 1)
    assert distance_gate.accepted == 0


def test_an_rrf_gate_accepts_at_or_above_the_threshold() -> None:
    [loose, mid, tight] = gate_sweep(_outcomes(), "rrf", [0.4, 0.6, 0.9])

    assert loose.accepted == 3
    assert mid.accepted == 2
    assert (tight.correct, tight.accepted) == (1, 1)


def test_the_best_threshold_prefers_the_stricter_on_a_tie() -> None:
    points = gate_sweep(_outcomes(), "cosine", [0.25, 0.3, 0.45])

    # All three accept only "good": same F1, so the tightest wins.
    assert best_point(points, higher_is_better=False).threshold == 0.25
    rrf_points = gate_sweep(_outcomes(), "rrf", [0.7, 0.8, 0.9])
    assert best_point(rrf_points, higher_is_better=True).threshold == 0.9


def test_parity_of_identical_results_is_perfect() -> None:
    rows = {"e": [{"id": "a", "score": 0.1}, {"id": "b", "score": 0.3}]}

    assert parity(rows, rows) == {
        "top1_agreement": 1.0,
        "mean_overlap": 1.0,
        "max_distance_diff": 0.0,
    }


def test_parity_notices_a_different_top_hit_and_drifting_distances() -> None:
    a = {"e": [{"id": "a", "score": 0.1}, {"id": "b", "score": 0.3}]}
    b = {"e": [{"id": "b", "score": 0.2}, {"id": "a", "score": 0.15}]}

    result = parity(a, b)

    assert result["top1_agreement"] == 0.0
    assert result["max_distance_diff"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Labels and the end-to-end run (need the seeded fixture)
# ---------------------------------------------------------------------------


def _require_fixture() -> None:
    try:
        psycopg.connect(get_postgres_connection_string(), connect_timeout=3).close()
    except Exception as exc:  # noqa: BLE001 -- no database, nothing to assert
        pytest.skip(f"postgres not reachable: {exc}")


def test_every_label_names_something_in_the_seeded_catalog() -> None:
    _require_fixture()

    targets = resolve_targets(load_entities())

    assert all(targets[e.entity] for e in load_entities() if e.has_answer)


@pytest.fixture
def fake() -> Iterator[FakeEmbedder]:
    with FakeEmbedder(host="0.0.0.0") as server:
        yield server


def test_a_run_reports_every_mode_and_the_semantic_modes_agree(
    fake: FakeEmbedder,
) -> None:
    _require_fixture()
    from auto_ontology.infra.elasticsearch import (
        EmbedEndpoint,
        get_elasticsearch_client,
        get_elasticsearch_config,
    )
    from auto_ontology.utils.embedding import EmbedParams

    try:
        get_elasticsearch_client(get_elasticsearch_config()).info()
    except Exception as exc:  # noqa: BLE001 -- no cluster, nothing to assert
        pytest.skip(f"elasticsearch not reachable: {exc}")

    host = os.environ.get("ES_TEST_EMBEDDER_HOST", "host.docker.internal")
    run = Run(
        embed_params=EmbedParams(model_name="fake/app", embed_invoke_url=fake.url()),
        es_endpoint=EmbedEndpoint(
            inference_id=f"ao-calibration-test-{uuid.uuid4().hex[:6]}",
            model_id="fake/es",
            api_key="unused",
            url=f"{fake.url(host)}/embeddings",
        ),
        top_k=5,
    )
    entities = [
        e
        for e in load_entities()
        if e.entity in {"movie length", "rental_rate", "film budget", "renter email"}
    ]

    report = calibrate(run, entities)

    assert set(report["ranking"]) == {"postgres", "es-semantic", "es-hybrid"}
    assert report["parity"]["top1_agreement"] == 1.0
    assert report["parity"]["max_distance_diff"] < 0.02
    assert report["gates"]["es-hybrid"]["gate"] == "name-or-distance"
    assert {"es-hybrid, vector-distance", "es-hybrid, rrf"} <= set(report["gates"])
    assert "## Gates" in render(report)
