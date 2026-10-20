# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Response parsing and batching for the hosted layout model, without a network.

Clients are built with ``object.__new__`` so no endpoint has to resolve, and the
HTTP layer is replaced by a function recording what it was asked for. That
leaves the service's own detection quality untested — only a real page can show
that — but covers the parts that have actually broken: the 0-1 box scale, the
response ``index``, and keeping a document alive when a batch fails.
"""

from __future__ import annotations

from typing import Any

from unstructured_ao.page_elements import (
    FIGURE_LABELS,
    Box,
    PageElementClient,
    _boxes_of,
)


def _client(batches: Any) -> PageElementClient:
    """A client whose per-batch call is *batches*, with no endpoint resolved."""
    client = object.__new__(PageElementClient)
    client._detect_batch = batches  # type: ignore[method-assign]
    return client


def test_boxes_scale_to_pixels() -> None:
    """The service returns fractions of the page; crops need pixels."""
    box = Box("chart", 0.1, 0.2, 0.6, 0.8, 0.9)
    assert box.to_pixels(1000, 500) == (100, 100, 600, 400)


def test_boxes_of_reads_every_label() -> None:
    boxes = _boxes_of(
        {
            "chart": [
                {
                    "x_min": 0.0,
                    "y_min": 0.0,
                    "x_max": 0.5,
                    "y_max": 0.5,
                    "confidence": 0.9,
                }
            ],
            "title": [
                {
                    "x_min": 0.1,
                    "y_min": 0.1,
                    "x_max": 0.9,
                    "y_max": 0.2,
                    "confidence": 0.8,
                }
            ],
        }
    )
    assert sorted(box.label for box in boxes) == ["chart", "title"]


def test_boxes_of_drops_low_confidence() -> None:
    """Below the floor a detection is more likely noise than a figure."""
    boxes = _boxes_of(
        {
            "chart": [
                {"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1, "confidence": 0.05},
                {"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1, "confidence": 0.95},
            ]
        }
    )
    assert [box.confidence for box in boxes] == [0.95]


def test_boxes_of_tolerates_an_empty_label() -> None:
    assert _boxes_of({"chart": None, "table": []}) == []


def test_detect_returns_one_entry_per_page() -> None:
    seen: list[list[str]] = []

    def batches(images: list[str]) -> list[list[Box]]:
        seen.append(images)
        return [[Box("chart", 0, 0, 1, 1, 1.0)] for _ in images]

    pages = _client(batches).detect(["a", "b", "c"])
    assert len(pages) == 3
    assert seen == [["a", "b", "c"]]


def test_detect_splits_long_documents_into_batches() -> None:
    from unstructured_ao.page_elements import _BATCH

    sizes: list[int] = []

    def batches(images: list[str]) -> list[list[Box]]:
        sizes.append(len(images))
        return [[] for _ in images]

    pages = _client(batches).detect(["p"] * (_BATCH + 3))
    assert len(pages) == _BATCH + 3
    assert sorted(sizes) == [3, _BATCH]


def test_detect_keeps_pages_aligned_when_a_batch_fails() -> None:
    """A failed layout call loses the figures, not the document."""

    def batches(_images: list[str]) -> list[list[Box]]:
        raise RuntimeError("service down")

    assert _client(batches).detect(["a", "b"]) == [[], []]


def test_detect_is_empty_for_no_pages() -> None:
    def batches(_images: list[str]) -> list[list[Box]]:
        raise AssertionError("must not be called")

    assert _client(batches).detect([]) == []


def test_figure_labels_exclude_prose() -> None:
    """Paragraph and title text already comes from the PDF's text layer."""
    assert "chart" in FIGURE_LABELS
    assert not FIGURE_LABELS & {"paragraph", "title"}
