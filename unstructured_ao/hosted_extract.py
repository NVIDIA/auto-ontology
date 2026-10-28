# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extraction with nothing running locally: hosted layout, hosted vision.

It returns the same record shape as Docling (``text``, ``source_id``,
``element_type``, ``page_number``), so
:func:`unstructured_ao.ingest.ingest_file` does not know which one ran.

Why it exists
-------------
Docling runs page layout on the local CPU and crashes on Apple's MPS backend,
so it is pinned to CPU. The hosted route avoids that local model.

Here the PDF is read with ``pypdfium2`` (text layer and page rasterisation, no
model), layout comes from NVIDIA's hosted page-elements model at ~0.8s per page,
and figures are read by the ``UNSTRUCTURED_VISION_*`` model. No torch, no cluster, and
the only local work is parsing and cropping.

What it gives up
----------------
Reading order is the PDF's own, not a model's: text comes out in the order the
file stores it, which is usually but not always visual order. Docling's layout
model reorders multi-column pages properly. For page-level retrieval that
distinction does not matter — the whole page is one row either way — but it
would matter if rows were ever cut finer than a page.

Tables are transcribed by the vision model rather than by a table-structure
model, so they arrive as readable text rather than as a faithful cell grid.
``nemotron-table-structure-v1`` is the better tool if the grid is ever needed.
"""

from __future__ import annotations

import base64
import io
import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: PDFium is not thread-safe, and uploads are extracted on a thread pool. Every
#: call into ``pypdfium2`` — open, text, render, close — happens under this.
_PDFIUM_LOCK = threading.Lock()

#: Pages rendered at once. A page at :data:`_SCALE` is about 6 MB in memory,
#: so holding a whole document's pages would grow without bound.
_PAGE_BATCH = 24

#: Detected label → the ``element_type`` to report, matching the vocabulary
#: ``_CONTENT_TYPE_MAP`` in :mod:`unstructured_ao.ingest` already resolves.
_ELEMENT_TYPES: dict[str, str] = {
    "chart": "chart",
    "table": "table",
    "infographic": "infographic",
}

#: Rasterisation scale.  2.0 is what the figure-reading measurements were taken
#: at; lower starts losing axis labels, higher mostly costs upload bytes.
_SCALE = 2.0

#: Asks for flat text, because this is embedded and searched rather than parsed.
_FIGURE_PROMPT = (
    "Transcribe this figure as text for a search index. Include the title, axis "
    "labels, legend entries, every category name, and every data value you can "
    "read. Output plain text only, no commentary and no markdown fences."
)

#: A crop smaller than this is a rule, bullet or icon rather than a figure.
_MIN_CROP_PIXELS = 64

#: Below this a figure row carries nothing the page text does not already say.
_MIN_FIGURE_CHARS = 40


def _png_b64(image: Any) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _page_text(page: Any) -> str:
    """The page's own text layer.

    Empty for a scanned page — those need OCR, which this extractor does not do.
    Twenty pages of the evaluation corpus come back empty here.
    """
    try:
        textpage = page.get_textpage()
        try:
            return (textpage.get_text_bounded() or "").strip()
        finally:
            textpage.close()
    except Exception:
        logger.exception("Text extraction failed for a page.")
        return ""


def _read_pages(document: Any, start: int, stop: int) -> tuple[list[str], list[Any]]:
    """Text and a rendered image for pages ``start`` to ``stop - 1``.

    The image is copied out of PDFium's bitmap so the PDFium objects can be
    closed here, under the lock, rather than by the garbage collector on
    whichever thread it next runs. The lock is taken per page, so another
    upload's extraction can run between this one's pages.
    """
    texts: list[str] = []
    images: list[Any] = []
    for index in range(start, stop):
        with _PDFIUM_LOCK:
            page = document[index]
            try:
                texts.append(_page_text(page))
                bitmap = page.render(scale=_SCALE)
                try:
                    images.append(bitmap.to_pil().copy())
                finally:
                    bitmap.close()
            finally:
                page.close()
    return texts, images


def _text_rows(page_texts: list[str], source_id: str) -> list[dict[str, Any]]:
    """One row per page that has text, so a hit maps onto a page.

    The relevance judgements are per page, and a row cut finer than a page
    cannot be scored against them.
    """
    return [
        {
            "text": text,
            "source_id": source_id,
            "element_type": "text",
            "page_number": number,
        }
        for number, text in enumerate(page_texts, start=1)
        if text
    ]


def _collect_crops(
    images: list[Any], detections: list[list[Any]], *, first_page: int = 1
) -> tuple[list[str], list[tuple[int, str]]]:
    """Every figure crop on *images*, with the page and label it came from.

    *first_page* is the page number of ``images[0]``. Crops are gathered across
    a whole batch of pages before any is sent, because the vision calls are
    latency-bound: eight in flight measured 5.4x faster than one at a time on
    identical output.
    """
    from unstructured_ao.page_elements import FIGURE_LABELS

    crops: list[str] = []
    meta: list[tuple[int, str]] = []
    for index, (image, boxes) in enumerate(zip(images, detections)):
        width, height = image.size
        for box in boxes:
            if box.label not in FIGURE_LABELS:
                continue
            left, top, right, bottom = box.to_pixels(width, height)
            if right - left < _MIN_CROP_PIXELS or bottom - top < _MIN_CROP_PIXELS:
                continue
            crops.append(_png_b64(image.crop((left, top, right, bottom))))
            meta.append((first_page + index, box.label))
    return crops, meta


def _figure_rows(
    meta: list[tuple[int, str]], transcripts: list[str], source_id: str
) -> list[dict[str, Any]]:
    """One row per figure the vision model managed to read."""
    rows = []
    for (page_number, label), transcript in zip(meta, transcripts):
        text = (transcript or "").strip()
        if len(text) < _MIN_FIGURE_CHARS:
            continue
        rows.append(
            {
                "text": text,
                "source_id": source_id,
                "element_type": _ELEMENT_TYPES.get(label, "infographic"),
                "page_number": page_number,
            }
        )
    return rows


def extract_chunks(file_path: str, client: Any | None = None) -> list[dict[str, Any]]:
    """Extract *file_path* into records, without embeddings.

    Embedding happens afterwards in :mod:`unstructured_ao.ingest`, so a re-embed
    does not re-extract and extraction stays testable without an endpoint.
    """
    import pypdfium2

    from unstructured_ao.embed_client import get_embed_client
    from unstructured_ao.page_elements import get_page_element_client

    client = client or get_embed_client()
    detector = get_page_element_client()
    source_id = str(Path(file_path).resolve())

    with _PDFIUM_LOCK:
        document = pypdfium2.PdfDocument(file_path)
        page_count = len(document)
    page_texts: list[str] = []
    figure_rows: list[dict[str, Any]] = []
    figures = 0
    try:
        for start in range(0, page_count, _PAGE_BATCH):
            texts, images = _read_pages(
                document, start, min(start + _PAGE_BATCH, page_count)
            )
            page_texts += texts
            detections = detector.detect([_png_b64(image) for image in images])
            crops, meta = _collect_crops(images, detections, first_page=start + 1)
            # Read now, so a batch's crops are released before the next renders.
            transcripts = client.describe_images(_FIGURE_PROMPT, crops)
            figure_rows += _figure_rows(meta, transcripts, source_id)
            figures += len(crops)
    finally:
        with _PDFIUM_LOCK:
            document.close()

    rows = _text_rows(page_texts, source_id) + figure_rows
    rows.sort(key=lambda row: row["page_number"] or 0)
    logger.info(
        "Hosted extraction: %d row(s) from %s — %d page(s), %d figure(s) read.",
        len(rows),
        Path(file_path).name,
        page_count,
        figures,
    )
    return rows
