# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docling extraction for the unstructured ingest pipeline.

A drop-in alternative to the retriever's extract stage for
:func:`unstructured_ao.ingest.ingest_file`: same dict shape out (``text``,
``source_id``, ``element_type``, ``page_number``), so everything downstream —
``content_type`` mapping, chunk rows, vector upload — is untouched.

PDF rows are page-based, not chunk-based
----------------------------------------
Docling's ``HybridChunker`` splits on document structure, which cuts across
pages: on a 52-page document it yielded 367 chunks averaging 511 characters
where the retriever yielded 52 page rows. Page rows are kept instead because
relevance here is judged per page, and because a chunk spanning two pages has no
single page number to record. Formats without page provenance, such as DOCX and
HTML, produce one text row per heading-delimited section with no page number.

Charts need a second pass
-------------------------
Docling finds every chart and reads nothing inside it. Measured against the
retriever on a 20-chart document: identical detection, 26,463 characters of
chart text for the retriever against zero for Docling. The interiors are vector
graphics it leaves as pixels, and the numbers are absent from the page's text
stream entirely.

Docling ships a chart-understanding stage for exactly this, but it only parses
Granite-Vision's output contract (both ``ChartExtractionOutputFormat`` members
are ``granite_vision_*``), so pointing it at any other model yields empty
annotations. Rather than ship an 8GB local model for it, the crop goes to the
same gateway the embeddings already use. On three sample charts that recovered
6/6, 2/3 and the full series of the retriever's values.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Docling item labels that get their own row, and the ``element_type`` to
#: report for each.  Names match the retriever's vocabulary so
#: ``_CONTENT_TYPE_MAP`` in ``ingest`` resolves both extractors' output.
_FIGURE_LABELS: dict[str, str] = {
    "picture": "chart",
    "chart": "chart",
}

#: Asks for the chart's content as flat text.  Deliberately not JSON or CSV:
#: this is embedded and searched, never parsed, and a model that cannot read an
#: axis should drop that series rather than fail a schema.
_CHART_PROMPT = (
    "Transcribe this chart as text for a search index. Include the title, axis "
    "labels, legend entries, every category name, and every data value you can "
    "read. Output plain text only, no commentary and no markdown fences."
)

#: Below this many characters a figure row is treated as not worth embedding.
#: A caption alone ("Figure 1: Classification of care work", 37 characters) adds
#: nothing the page text does not already say.
_MIN_FIGURE_CHARS = 40


def _pipeline_options() -> Any:
    """Docling PDF options tuned for this corpus.

    OCR is off: it defaults on, downloads ~30MB of RapidOCR models from
    modelscope.cn, and on born-digital PDFs adds minutes per document for
    nothing.  Turn it back on for scanned input.

    Picture images are on because the chart pass needs the crop; page images are
    off because rows are embedded as text — measured better than ``text_image``
    on this corpus, see ``_embed_params`` in :mod:`unstructured_ao.ingest`.

    The accelerator is pinned to CPU.  Left on ``auto`` it picks MPS on Apple
    silicon, where the layout model dies with "Cannot convert a MPS Tensor to
    float64 dtype" on every page.  CUDA hosts should set
    ``DOCLING_DEVICE=cuda``; this only needs to not be MPS.
    """
    import os

    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        PdfPipelineOptions,
    )

    options = PdfPipelineOptions()
    options.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice(os.environ.get("DOCLING_DEVICE", "cpu"))
    )
    options.do_ocr = False
    options.generate_page_images = False
    options.generate_picture_images = True
    options.images_scale = 2.0
    return options


def _converter() -> Any:
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=_pipeline_options())
        }
    )


def _page_of(item: Any) -> int | None:
    """The 1-based page an item sits on, or ``None`` if Docling did not say.

    Page numbers are 1-based in both Docling and the retriever, which is also
    what page-level relevance judgements use, so no adjustment is needed.
    """
    prov = getattr(item, "prov", None)
    return prov[0].page_no if prov else None


def _crop_b64(image: Any) -> str:
    """One figure crop as base64 PNG, ready for the vision call."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _page_rows(document: Any, source_id: str) -> list[dict[str, Any]]:
    """Text grouped by page or section, depending on available provenance."""
    texts: dict[int, list[str]] = {}
    unpaged: list[tuple[Any, int]] = []
    for item, level in document.iterate_items():
        label = getattr(getattr(item, "label", None), "value", None)
        if label in _FIGURE_LABELS or label == "table":
            continue  # these get their own rows
        text = (getattr(item, "text", "") or "").strip()
        page = _page_of(item)
        if not text:
            continue
        if page is None:
            unpaged.append((item, level))
        else:
            texts.setdefault(page, []).append(text)
    rows = [
        {
            "text": "\n".join(lines),
            "source_id": source_id,
            "element_type": "text",
            "page_number": page,
        }
        for page, lines in sorted(texts.items())
    ]
    if unpaged:
        rows.extend(_section_rows(unpaged, source_id))
    return rows


def _section_rows(items: list[tuple[Any, int]], source_id: str) -> list[dict[str, Any]]:
    """One row per heading-delimited section, with ancestor headings included.

    A section longer than :data:`MAX_CHUNK_CHARS` becomes several rows, each
    carrying the same headings so it still says where it came from.
    """
    from unstructured_ao.text_extract import MAX_CHUNK_CHARS, split_text

    rows: list[dict[str, Any]] = []
    headings: dict[int, str] = {}
    body: list[str] = []

    def flush() -> None:
        if not body:
            return
        prefix = "\n".join(headings[level] for level in sorted(headings))
        # At most half a row, keeping the end: the deepest heading names the
        # section, and the body keeps room to say something.
        prefix = prefix[-(MAX_CHUNK_CHARS // 2) :].lstrip()
        budget = MAX_CHUNK_CHARS - len(prefix) - 1
        for piece in split_text("\n".join(body), budget):
            rows.append(
                {
                    "text": f"{prefix}\n{piece}" if prefix else piece,
                    "source_id": source_id,
                    "element_type": "text",
                    "page_number": None,
                }
            )
        body.clear()

    for item, level in items:
        text = (getattr(item, "text", "") or "").strip()
        label = getattr(getattr(item, "label", None), "value", None)
        if label in {"title", "section_header"}:
            flush()
            headings = {
                heading_level: heading
                for heading_level, heading in headings.items()
                if heading_level < level
            }
            headings[level] = text
        else:
            body.append(text)

    flush()
    return rows


def _table_rows(document: Any, source_id: str) -> list[dict[str, Any]]:
    """One markdown row per table, never split.

    Markdown rather than the cell grid because that is what the retriever
    emitted and what the embedder was measured on.  A long table stays one row;
    the embedder's length limit decides any truncation.
    """
    rows = []
    for table in getattr(document, "tables", []):
        try:
            text = table.export_to_markdown(document).strip()
        except Exception:
            logger.exception("Table export failed — skipping the row.")
            continue
        if text:
            rows.append(
                {
                    "text": text,
                    "source_id": source_id,
                    "element_type": "table",
                    "page_number": _page_of(table),
                }
            )
    return rows


def _figure_rows(document: Any, source_id: str, client: Any) -> list[dict[str, Any]]:
    """One row per figure: its caption plus whatever the VLM reads in it.

    Every crop is collected before any is sent, so the vision calls go out
    together: they are latency-bound, and eight in flight measured 5.4x faster
    than one at a time on identical output.
    """
    pending: list[tuple[Any, str, str]] = []
    for picture in getattr(document, "pictures", []):
        label = getattr(getattr(picture, "label", None), "value", None)
        element_type = _FIGURE_LABELS.get(label or "picture", "chart")
        try:
            caption = (picture.caption_text(document) or "").strip()
        except Exception:
            caption = ""
        image = picture.get_image(document)
        if image is None:
            continue
        pending.append((picture, element_type, caption))

    crops = [_crop_b64(picture.get_image(document)) for picture, _, _ in pending]
    transcripts = client.describe_images(_CHART_PROMPT, crops)

    rows = []
    for (picture, element_type, caption), transcript in zip(pending, transcripts):
        text = "\n".join(part for part in (caption, transcript.strip()) if part).strip()
        if len(text) < _MIN_FIGURE_CHARS:
            continue
        rows.append(
            {
                "text": text,
                "source_id": source_id,
                "element_type": element_type,
                "page_number": _page_of(picture),
            }
        )
    return rows


def extract_chunks(file_path: str, client: Any | None = None) -> list[dict[str, Any]]:
    """Extract *file_path* into retriever-shaped records, without embeddings.

    Unlike the retriever's graph this does not embed: *client* is used only to
    read figures.  :func:`unstructured_ao.ingest.ingest_file` embeds the rows
    afterwards, which keeps extraction testable without an endpoint and means a
    re-embed does not re-extract.
    """
    from unstructured_ao.embed_client import get_embed_client

    client = client or get_embed_client()
    document = _converter().convert(file_path).document
    source_id = str(Path(file_path).resolve())

    rows = (
        _page_rows(document, source_id)
        + _table_rows(document, source_id)
        + _figure_rows(document, source_id, client)
    )
    rows.sort(key=lambda row: row["page_number"] or 0)
    logger.info(
        "Docling extracted %d row(s) from %s across %d page(s).",
        len(rows),
        Path(file_path).name,
        len(getattr(document, "pages", {}) or {}),
    )
    return rows
