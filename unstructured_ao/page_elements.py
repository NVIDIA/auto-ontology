# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Page-layout detection through NVIDIA's hosted CV model.

Answers "where on this page are the paragraphs, titles, tables and charts" —
the job Docling does with a local torch model, and NeMo Retriever does with this
same service. Measured at **0.8s per page** against roughly two seconds of local
CPU, which is what makes it worth a second client.

Host and key
------------
The hosted model is available through the inference gateway with the same
``sk-`` key family as the other hosted models. Unlike the OpenAI-compatible
chat and embedding routes, its model-specific URL ends in
``/v1/page-elements``.

Wire format
-----------
``{"input": [{"type": "image_url", "url": "data:image/png;base64,..."}]}`` — the
shape NeMo builds in ``yolox.py``, not the chat-completions contract. Boxes come
back **normalised to 0-1**, so they must be scaled by the image's pixel size
before cropping.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: Model-config triplet prefix, so the vars sit in the same ``UNSTRUCTURED_*``
#: namespace as the rest of this package's settings.
_PREFIX = "UNSTRUCTURED_PAGE_ELEMENTS"

#: Labels this model emits that are figures — a region whose meaning is in the
#: pixels, so it needs reading separately.  ``paragraph`` and ``title`` are
#: deliberately absent: their text is already in the PDF's text layer.
FIGURE_LABELS: frozenset[str] = frozenset({"chart", "table", "infographic"})

#: Pages per request.  Matches NeMo's ``page_elements_batch_size`` default.
_BATCH = 24

#: Requests in flight.  Detection is fast enough that this mostly hides
#: round-trip latency on long documents.
_WORKERS = 4

_TIMEOUT = 180.0

#: Below this the detection is more likely noise than a figure.  NeMo applies
#: per-label thresholds in the same spirit (``YOLOX_PAGE_V2_FINAL_SCORE``); a
#: single conservative floor is enough here because a missed figure costs only
#: its own row.
_MIN_CONFIDENCE = 0.30


@dataclass(frozen=True)
class Box:
    """One detected region, in fractions of the page."""

    label: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """``(left, top, right, bottom)`` for ``PIL.Image.crop``."""
        return (
            int(self.x_min * width),
            int(self.y_min * height),
            int(self.x_max * width),
            int(self.y_max * height),
        )


def _resolved() -> tuple[str, str, str]:
    from auto_ontology.utils.model_config import resolve

    endpoint = resolve(_PREFIX, "ENDPOINT", use_shared_default=False).rstrip("/")
    model = resolve(_PREFIX, "MODEL", use_shared_default=False).strip("/")
    key = resolve(_PREFIX, "API_KEY")
    return endpoint, key, model


def _inference_url(endpoint: str, model: str) -> str:
    """Build the route for the selected hosted Page Elements gateway."""
    base = f"{endpoint}/{model}"
    if endpoint.endswith("/v1/ocr"):
        return f"{base}/v1/page-elements"
    return base


class PageElementClient:
    """Detects page elements, a batch of page images at a time."""

    def __init__(self) -> None:
        import httpx

        endpoint, self._api_key, model = _resolved()
        self._url = _inference_url(endpoint, model)
        self._http = httpx.Client(timeout=_TIMEOUT)

    def _detect_batch(self, images_b64: list[str]) -> list[list[Box]]:
        payload = {
            "input": [
                {"type": "image_url", "url": f"data:image/png;base64,{b64}"}
                for b64 in images_b64
            ]
        }
        response = self._http.post(
            self._url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        # Results carry their own ``index``; trust it rather than arrival order.
        found: list[list[Box]] = [[] for _ in images_b64]
        for item in response.json().get("data", []):
            index = item.get("index", 0)
            if not 0 <= index < len(found):
                continue
            found[index] = _boxes_of(item.get("bounding_boxes") or {})
        return found

    def detect(self, images_b64: list[str]) -> list[list[Box]]:
        """Boxes per page, in input order.

        A batch the service rejects yields empty lists for its pages rather than
        raising: a document whose layout call failed still has its text layer,
        and losing the figures beats losing the document.
        """
        if not images_b64:
            return []
        batches = [
            images_b64[start : start + _BATCH]
            for start in range(0, len(images_b64), _BATCH)
        ]

        def run(batch: list[str]) -> list[list[Box]]:
            try:
                return self._detect_batch(batch)
            except Exception:
                logger.exception("Page-element detection failed for a batch.")
                return [[] for _ in batch]

        with ThreadPoolExecutor(max_workers=min(_WORKERS, len(batches))) as pool:
            results = list(pool.map(run, batches))
        return [page for batch in results for page in batch]


def _boxes_of(bounding_boxes: dict[str, Any]) -> list[Box]:
    boxes = []
    for label, entries in bounding_boxes.items():
        for entry in entries or []:
            confidence = float(entry.get("confidence", 0.0))
            if confidence < _MIN_CONFIDENCE:
                continue
            boxes.append(
                Box(
                    label=label,
                    x_min=float(entry["x_min"]),
                    y_min=float(entry["y_min"]),
                    x_max=float(entry["x_max"]),
                    y_max=float(entry["y_max"]),
                    confidence=confidence,
                )
            )
    return boxes


_client: PageElementClient | None = None


def get_page_element_client() -> PageElementClient:
    """The process-wide client, built on first use."""
    global _client
    if _client is None:
        _client = PageElementClient()
    return _client
