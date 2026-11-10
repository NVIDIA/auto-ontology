# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from auto_ontology.retrieval.interactive import entity_resolution as er


def _hit(entity, hit_id, score, text=""):
    return {"query_entity": entity, "id": hit_id, "score": score, "text": text}


def _run(col_hits, entities):
    path_state = {
        "entities": entities,
        "retrieved_column_attributes": col_hits,
        "final_response": {"uncovered_entities": [], "candidates": []},
    }
    still_ambiguous: set[str] = set()
    ambiguous_hits: dict = {}
    with (
        patch.object(er, "_run_entity_coverage_pipeline", return_value=path_state),
        patch.object(
            er,
            "_collect_ambiguous_hits",
            side_effect=lambda _r, _db, ents: {e: [] for e in ents},
        ),
    ):
        result = er._find_unresolvable_entities(
            "Show me each profile's growth speed.",
            semantic_retriever=object(),
            db_name="fake_account",
            ambiguous_hits_out=ambiguous_hits,
            still_ambiguous_out=still_ambiguous,
        )
    return result, still_ambiguous, ambiguous_hits


def test_ambiguous_term_without_collision_is_reported_as_still_ambiguous():
    (unresolvable, resolved_hits, *_), still_ambiguous, ambiguous_hits = _run(
        [
            _hit("profile growth speed", "foll", 0.30, "Follower growth rate"),
            _hit("profile growth speed", "fing", 0.32, "Following growth rate"),
        ],
        ["profile growth speed"],
    )

    assert still_ambiguous == {"profile growth speed"}
    assert "profile growth speed" in ambiguous_hits
    assert unresolvable == []
    assert [h[0] for h in resolved_hits] == ["profile growth speed"]


def test_clear_single_match_is_not_ambiguous():
    _, still_ambiguous, _ = _run(
        [_hit("downtime score", "mttr", 0.20)],
        ["downtime score"],
    )

    assert still_ambiguous == set()


def test_still_ambiguous_out_is_reset_on_each_call():
    still_ambiguous = {"stale term"}
    with patch.object(er, "_run_entity_coverage_pipeline", return_value={}):
        er._find_unresolvable_entities(
            "q", object(), "db", still_ambiguous_out=still_ambiguous
        )

    assert still_ambiguous == set()


def test_collided_entities_are_those_sharing_a_best_hit():
    best = {
        "follower growth": {"id": "foll"},
        "growth speed": {"id": "foll"},
        "profile score": {"id": "score"},
    }

    assert er._collided_entities(best) == {"follower growth", "growth speed"}


def test_still_ambiguous_skips_collided_and_kb_covered_terms():
    best = {
        "growth speed": {"id": "foll"},
        "risk level": {"id": "risk"},
        "reach": {"id": "reach"},
    }

    norm = lambda e: er._normalize_entity(e) or e  # noqa: E731

    still = er._still_ambiguous_norms(
        ambiguous_entities={"growth speed", "risk level", "reach"},
        best_hit_per_entity=best,
        collided_entities={"reach"},
        kb_covered_norms={norm("risk level")},
    )

    assert still == {norm("growth speed")}
