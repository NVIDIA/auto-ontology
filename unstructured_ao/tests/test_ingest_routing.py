# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

import unstructured_ao.config as config
from unstructured_ao.ingest import _extractor_for_file
from unstructured_ao.text_extract import extract_chunks


@pytest.mark.parametrize(
    ("path", "extractor"),
    [
        ("document.pdf", "hosted"),
        ("document.docx", "docling"),
        ("slides.pptx", "docling"),
        ("page.html", "docling"),
        ("notes.txt", "text"),
    ],
)
def test_auto_routes_by_extension(
    monkeypatch: pytest.MonkeyPatch, path: str, extractor: str
) -> None:
    monkeypatch.setattr(config, "EXTRACTOR", "auto")
    assert _extractor_for_file(path) == extractor


def test_hosted_rejects_non_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "EXTRACTOR", "hosted")
    with pytest.raises(ValueError, match="does not support"):
        _extractor_for_file("document.docx")


def test_unknown_extractor_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "EXTRACTOR", "unknown")
    with pytest.raises(ValueError, match="Unknown EXTRACTOR"):
        _extractor_for_file("document.pdf")


def test_plain_text_extraction(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("  first line\nsecond line  ", encoding="utf-8")

    assert extract_chunks(str(path)) == [
        {
            "text": "first line\nsecond line",
            "source_id": str(path),
            "element_type": "text",
            "page_number": None,
        }
    ]
