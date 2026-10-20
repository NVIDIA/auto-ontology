# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Row building from a converted document, without Docling or an endpoint.

The functions under test take the converted document as an argument and read it
through duck typing, so a handful of stand-ins exercise them without the
``unstructured`` extra installed, without torch, and without a model download.
What is left untested here is Docling's own conversion, which no unit test can
stand in for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from unstructured_ao.docling_extract import (
    _MIN_FIGURE_CHARS,
    _figure_rows,
    _page_rows,
    _table_rows,
)


@dataclass
class FakeLabel:
    value: str


@dataclass
class FakeProv:
    page_no: int


@dataclass
class FakeItem:
    text: str = ""
    label_value: str = "text"
    page: int | None = 1

    @property
    def label(self) -> FakeLabel:
        return FakeLabel(self.label_value)

    @property
    def prov(self) -> list[FakeProv]:
        return [FakeProv(self.page)] if self.page is not None else []


@dataclass
class FakePicture(FakeItem):
    caption: str = ""
    image: Any = object()

    def caption_text(self, _doc: Any) -> str:
        return self.caption

    def get_image(self, _doc: Any) -> Any:
        return self.image


@dataclass
class FakeTable(FakeItem):
    markdown: str = ""

    def export_to_markdown(self, _doc: Any) -> str:
        return self.markdown


@dataclass
class FakeDoc:
    items: list[Any] = field(default_factory=list)
    tables: list[Any] = field(default_factory=list)
    pictures: list[Any] = field(default_factory=list)

    def iterate_items(self):
        return ((item, 0) for item in self.items)


class FakeClient:
    """Stands in for the gateway; records what it was asked to read.

    ``fail`` returns empty strings rather than raising, which is what the real
    ``describe_images`` does — it swallows a per-crop failure so one unreadable
    figure cannot cost the document its other rows.
    """

    def __init__(self, reply: str = "", fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.calls = 0

    def describe_images(self, _prompt: str, images_b64: list[str]) -> list[str]:
        self.calls += len(images_b64)
        return ["" if self.fail else self.reply for _ in images_b64]


class FakeImage:
    def save(self, buffer: Any, format: str) -> None:  # noqa: A002 - PIL's name
        buffer.write(b"png-bytes")


def test_page_rows_group_text_by_page_in_order() -> None:
    doc = FakeDoc(
        items=[
            FakeItem("Heading", "section_header", 1),
            FakeItem("Body of page one.", "text", 1),
            FakeItem("Page two.", "text", 2),
        ]
    )
    rows = _page_rows(doc, "/corpus/a.pdf")
    assert [row["page_number"] for row in rows] == [1, 2]
    assert rows[0]["text"] == "Heading\nBody of page one."
    assert rows[0]["element_type"] == "text"
    assert rows[0]["source_id"] == "/corpus/a.pdf"


def test_page_rows_exclude_figures_and_tables() -> None:
    """They get their own rows; counting them twice would double the text."""
    doc = FakeDoc(
        items=[
            FakeItem("Prose.", "text", 1),
            FakeItem("A table leaked through", "table", 1),
            FakeItem("A picture leaked through", "picture", 1),
        ]
    )
    assert _page_rows(doc, "x")[0]["text"] == "Prose."


def test_page_rows_skip_items_with_no_page() -> None:
    """A row with no page number cannot be scored against page judgements."""
    doc = FakeDoc(items=[FakeItem("Floating.", "text", None)])
    assert _page_rows(doc, "x") == []


def test_table_rows_use_markdown() -> None:
    doc = FakeDoc(tables=[FakeTable(markdown="| a | b |\n| - | - |", page=4)])
    rows = _table_rows(doc, "x")
    assert rows == [
        {
            "text": "| a | b |\n| - | - |",
            "source_id": "x",
            "element_type": "table",
            "page_number": 4,
        }
    ]


def test_table_rows_survive_a_failed_export() -> None:
    class Broken(FakeTable):
        def export_to_markdown(self, _doc: Any) -> str:
            raise RuntimeError("malformed grid")

    doc = FakeDoc(tables=[Broken(page=1), FakeTable(markdown="| ok |", page=2)])
    assert [row["page_number"] for row in _table_rows(doc, "x")] == [2]


def test_figure_rows_join_caption_and_transcript() -> None:
    doc = FakeDoc(
        pictures=[FakePicture(caption="Figure 3: wages", image=FakeImage(), page=7)]
    )
    client = FakeClient(reply="Bar chart. DE 41. FR 38. IT 29.")
    rows = _figure_rows(doc, "x", client)
    assert rows[0]["text"] == "Figure 3: wages\nBar chart. DE 41. FR 38. IT 29."
    assert rows[0]["element_type"] == "chart"
    assert client.calls == 1


def test_figure_rows_drop_caption_only_figures() -> None:
    """A bare caption adds nothing the page text does not already carry."""
    doc = FakeDoc(pictures=[FakePicture(caption="Figure 1", image=FakeImage())])
    assert _figure_rows(doc, "x", FakeClient(reply="")) == []


def test_figure_rows_survive_a_gateway_failure() -> None:
    """One unreadable figure must not cost the document its other rows."""
    long_caption = "Figure 2: employment by sector, by country, 2019-2024"
    assert len(long_caption) >= _MIN_FIGURE_CHARS
    doc = FakeDoc(pictures=[FakePicture(caption=long_caption, image=FakeImage())])
    rows = _figure_rows(doc, "x", FakeClient(fail=True))
    assert rows[0]["text"] == long_caption


def test_figure_rows_skip_pictures_with_no_crop() -> None:
    doc = FakeDoc(pictures=[FakePicture(caption="Figure 4: a chart", image=None)])
    client = FakeClient(reply="never asked")
    assert _figure_rows(doc, "x", client) == []
    assert client.calls == 0
