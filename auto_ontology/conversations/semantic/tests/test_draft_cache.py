# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The extraction draft cache, and the fingerprint that makes it safe to use."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from auto_ontology.conversations.semantic.draft_cache import (
    fingerprint,
    load_draft,
    save_draft,
)
from auto_ontology.conversations.semantic.extract import (
    EdgeDraft,
    GraphDraft,
    NodeDraft,
)
from auto_ontology.conversations.semantic.windows import build_windows


def _turns(count: int = 3) -> list[dict]:
    return [
        {
            "turn_index": i,
            "question": f"question {i}",
            "answer": f"answer {i}",
            "time_anchor": "March-15-2024",
        }
        for i in range(count)
    ]


def _draft() -> GraphDraft:
    return GraphDraft(
        nodes=[
            NodeDraft(
                name="Weather app",
                type="Application",
                kinds=["object"],
                description="a weather app",
                turn_indexes={0, 2},
            ),
            NodeDraft(
                name="Daily quota 1,200/day",
                type="Measurement",
                kinds=["state"],
                description="the quota is 1,200 calls a day",
                key="daily_quota",
                value="1200 calls/day",
                occurred_on=date(2024, 3, 15),
                turn_indexes={2},
                verified_turn_indexes={2},
                evidence="1,200 calls per day",
                time_anchor="March-15-2024",
            ),
        ],
        edges=[
            EdgeDraft(
                source="Daily quota 1,200/day",
                target="Weather app",
                role="about",
                turn_indexes={2},
                verified_turn_indexes={2},
                evidence="the app's quota",
            )
        ],
    )


class TestRoundTrip:
    def test_a_saved_draft_loads_back_unchanged(self, tmp_path: Path) -> None:
        path = tmp_path / "draft.json"
        save_draft(path, "abc", _draft())
        loaded = load_draft(path, "abc")

        assert loaded is not None
        assert [n.name for n in loaded.nodes] == [
            "Weather app",
            "Daily quota 1,200/day",
        ]
        assert [(e.source, e.role, e.target) for e in loaded.edges] == [
            ("Daily quota 1,200/day", "about", "Weather app")
        ]

    def test_the_fields_derivation_reads_survive(self, tmp_path: Path) -> None:
        """The key, the value and the date are what the revision chain is built
        from, so a cache that lost them would silently produce a smaller graph."""
        path = tmp_path / "draft.json"
        save_draft(path, "abc", _draft())
        node = load_draft(path, "abc").nodes[1]

        assert node.key == "daily_quota"
        assert node.value == "1200 calls/day"
        assert node.occurred_on == date(2024, 3, 15)
        assert node.turn_indexes == {2}
        assert node.verified_turn_indexes == {2}
        assert node.evidence == "1,200 calls per day"
        assert node.time_anchor == "March-15-2024"

    def test_an_edges_verified_turn_survives(self, tmp_path: Path) -> None:
        """The only evidence an edge carries that it is true, so a cache that
        dropped it would make every reused draft look unattributed."""
        path = tmp_path / "draft.json"
        save_draft(path, "abc", _draft())
        assert load_draft(path, "abc").edges[0].verified_turn_indexes == {2}

    def test_turn_indexes_come_back_as_a_set(self, tmp_path: Path) -> None:
        """Stored sorted, because JSON has no set; the drafts are mutated with
        set operations downstream."""
        path = tmp_path / "draft.json"
        save_draft(path, "abc", _draft())
        assert load_draft(path, "abc").nodes[0].turn_indexes == {0, 2}


class TestInvalidation:
    def test_a_different_fingerprint_does_not_load(self, tmp_path: Path) -> None:
        """The whole point: a draft from a prompt that has since been edited
        would make the edit look like it did nothing."""
        path = tmp_path / "draft.json"
        save_draft(path, "abc", _draft())
        assert load_draft(path, "def") is None

    def test_a_missing_file_does_not_load(self, tmp_path: Path) -> None:
        assert load_draft(tmp_path / "absent.json", "abc") is None

    def test_a_corrupt_file_does_not_load(self, tmp_path: Path) -> None:
        path = tmp_path / "draft.json"
        path.write_text("{not json")
        assert load_draft(path, "abc") is None

    def test_a_truncated_write_is_not_left_in_place(self, tmp_path: Path) -> None:
        """Written beside and moved in, so an interrupted save cannot leave a
        partial draft that the next run reads as complete."""
        path = tmp_path / "draft.json"
        save_draft(path, "abc", _draft())
        assert not list(tmp_path.glob("*.partial"))


class TestFingerprint:
    def test_the_same_transcript_fingerprints_the_same(self) -> None:
        assert fingerprint(build_windows(_turns())) == fingerprint(
            build_windows(_turns())
        )

    def test_changed_turn_text_changes_the_fingerprint(self) -> None:
        turns = _turns()
        turns[1]["answer"] = "something else entirely"
        assert fingerprint(build_windows(turns)) != fingerprint(build_windows(_turns()))

    def test_a_different_window_size_changes_the_fingerprint(self) -> None:
        """One turn per call and the packed default are two different extractions
        of one transcript, and both are run."""
        turns = _turns(6)
        assert fingerprint(build_windows(turns, max_chars=1)) != fingerprint(
            build_windows(turns, max_chars=12000)
        )

    def test_an_extra_turn_changes_the_fingerprint(self) -> None:
        assert fingerprint(build_windows(_turns(4))) != fingerprint(
            build_windows(_turns(3))
        )
