# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Row building for the hosted extractor, without a PDF or an endpoint.

The functions under test take pages, detections and transcripts as arguments, so
stand-ins exercise them without ``pypdfium2``, without a layout call, and without
a vision call. PDF parsing and the service's detection quality stay untested —
only a real document shows those.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from unstructured_ao.hosted_extract import (
    _MIN_FIGURE_CHARS,
    _collect_crops,
    _figure_rows,
    _page_text,
    _text_rows,
)
from unstructured_ao.page_elements import Box


@dataclass
class FakeImage:
    """A page raster that records the crops taken out of it."""

    size: tuple[int, int] = (1000, 1000)

    def crop(self, _box: tuple[int, int, int, int]) -> FakeImage:
        return self

    def save(self, buffer: Any, format: str) -> None:  # noqa: A002 - PIL's name
        buffer.write(b"png-bytes")


def _box(label: str, span: float = 0.5, confidence: float = 0.9) -> Box:
    return Box(label, 0.1, 0.1, 0.1 + span, 0.1 + span, confidence)


def test_text_rows_number_pages_from_one() -> None:
    rows = _text_rows(["first", "second"], "/corpus/a.pdf")
    assert [row["page_number"] for row in rows] == [1, 2]
    assert rows[0]["element_type"] == "text"
    assert rows[0]["source_id"] == "/corpus/a.pdf"


def test_text_rows_skip_empty_pages_without_renumbering() -> None:
    """A scanned page has no text layer, but the pages after it keep their numbers."""
    rows = _text_rows(["first", "", "third"], "x")
    assert [row["page_number"] for row in rows] == [1, 3]


def test_page_text_survives_a_broken_page() -> None:
    class Broken:
        def get_textpage(self) -> Any:
            raise RuntimeError("no text layer")

    assert _page_text(Broken()) == ""


def test_collect_crops_only_takes_figures() -> None:
    """Paragraph and title text is already in the PDF's own text layer."""
    detections = [[_box("chart"), _box("paragraph"), _box("title"), _box("table")]]
    crops, meta = _collect_crops([FakeImage()], detections)
    assert len(crops) == 2
    assert [label for _, label in meta] == ["chart", "table"]


def test_collect_crops_tags_each_crop_with_its_page() -> None:
    detections = [[_box("chart")], [], [_box("table"), _box("chart")]]
    _, meta = _collect_crops([FakeImage(), FakeImage(), FakeImage()], detections)
    assert meta == [(1, "chart"), (3, "table"), (3, "chart")]


def test_collect_crops_drop_slivers() -> None:
    """A box a few pixels across is a rule or an icon, not a figure."""
    tiny = Box("chart", 0.0, 0.0, 0.01, 0.01, 0.9)
    crops, _ = _collect_crops([FakeImage(size=(1000, 1000))], [[tiny]])
    assert crops == []


def test_figure_rows_map_labels_to_element_types() -> None:
    long_enough = "Bar chart of employment by sector, 2019 to 2024, EU27."
    assert len(long_enough) >= _MIN_FIGURE_CHARS
    rows = _figure_rows(
        [(3, "chart"), (4, "table"), (5, "infographic")],
        [long_enough] * 3,
        "x",
    )
    assert [row["element_type"] for row in rows] == ["chart", "table", "infographic"]
    assert [row["page_number"] for row in rows] == [3, 4, 5]


def test_figure_rows_drop_unreadable_crops() -> None:
    """An empty transcript means the gateway could not read it; keep the rest."""
    long_enough = "Line chart of unemployment, by member state, 2015 to 2024."
    rows = _figure_rows([(1, "chart"), (2, "chart")], ["", long_enough], "x")
    assert [row["page_number"] for row in rows] == [2]


def test_figure_rows_drop_transcripts_too_short_to_matter() -> None:
    rows = _figure_rows([(1, "chart")], ["Figure 1"], "x")
    assert rows == []


def test_figure_rows_label_unknown_regions_as_infographic() -> None:
    long_enough = "A diagram of the policy pipeline from proposal to adoption."
    rows = _figure_rows([(1, "something_new")], [long_enough], "x")
    assert rows[0]["element_type"] == "infographic"
