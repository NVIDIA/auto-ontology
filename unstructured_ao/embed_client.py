# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embeddings and figure reading over plain HTTP, with no Ray cluster behind it.

What :mod:`unstructured_ao.ingest` needs from NeMo Retriever's 3,500-line embed
stack is one POST.  This is it, for the Docling path.

Verified against the library rather than guessed: embedding the same text and
image through here and through the retriever's own embedder agrees to cosine
0.9944-0.9961, where a text-only vector against the same reference scores
0.8748.  The payload shape is the retriever's — see
``_format_text_image_pair_input_string`` in
``nemo_retriever/models/inference/main_text_embed.py`` — which matters because
vectors written by either path have to share a space.

Modality
--------
``text`` by default, and that is a measured choice, not a simplification.  On
ViDoRe v3 HR, embedding the same chunks text-only beat ``text_image`` on nDCG@5
(0.2317 against 0.2131) and recall@5 (0.2215 against 0.1929) with a
paired-bootstrap interval excluding zero.  ``text_image`` is still reachable
through :meth:`EmbedClient.embed` for a re-measurement; the wire format for it
is one string, text and data URL joined by a newline, plus a top-level
``modality`` field.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

logger = logging.getLogger(__name__)

Modality = Literal["text", "image", "text_image"]

#: Rows per embed request.  The gateway accepts more, but a failure costs the
#: whole batch and ingest re-runs are not free.
_BATCH = 16

#: Vision calls in flight at once.  Eight measured 5.4x faster than serial on
#: the same crops; higher is untested and risks the gateway's rate limit, which
#: would turn a slow ingest into a failed one.
_VISION_WORKERS = 8

#: Seconds.  Generous because a cold model behind the gateway can take a while,
#: and because an ingest that dies at document 11 of 14 wastes more than it
#: saves.
_TIMEOUT = 300.0


def _resolved() -> tuple[str, str, str]:
    """``(endpoint, api_key, model)`` from the shared ``EMBED_*`` triplet."""
    from auto_ontology.utils.model_config import resolve

    return (
        resolve("EMBED", "ENDPOINT").rstrip("/"),
        resolve("EMBED", "API_KEY"),
        resolve("EMBED", "MODEL"),
    )


class EmbedClient:
    """Embeds text, and reads figures, through the configured gateway."""

    def __init__(self) -> None:
        import httpx

        self._endpoint, self._api_key, self._model = _resolved()
        self._http = httpx.Client(timeout=_TIMEOUT)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._http.post(
            f"{self._endpoint}/{path}",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    def embed(
        self,
        texts: list[str],
        *,
        images_b64: list[str | None] | None = None,
        modality: Modality = "text",
        input_type: str = "passage",
    ) -> list[list[float] | None]:
        """Embed *texts*, returning one vector per input, in input order.

        ``None`` marks a row the gateway would reject — an empty string has no
        vector — so the caller can keep rows and vectors aligned by position.
        The retriever's own ``embed_text_image`` drops image-less rows instead,
        which silently shifts every later vector onto the wrong row.
        """
        images = images_b64 or [None] * len(texts)
        if len(images) != len(texts):
            raise ValueError(
                f"{len(texts)} text(s) but {len(images)} image(s); they must "
                "correspond one-to-one for the vectors to map back to rows."
            )

        vectors: list[list[float] | None] = [None] * len(texts)
        pending = [
            i for i, text in enumerate(texts) if (text or "").strip() or images[i]
        ]
        for start in range(0, len(pending), _BATCH):
            batch = pending[start : start + _BATCH]
            payload = {
                "model": self._model,
                "input": [self._one_input(texts[i], images[i]) for i in batch],
                "encoding_format": "float",
                "input_type": input_type,
                # The model's ceiling is 4096 tokens and a page can exceed it —
                # one document in the evaluation corpus has a 5683-token page.
                # NeMo can send "NONE" because its chunker never builds a row
                # that long; a page-grained row sometimes does, and the gateway
                # answers 422 for the whole batch rather than that one row.
                # Losing the tail of a long page beats losing the document.
                "truncate": "END",
                "modality": modality,
            }
            data = self._post("embeddings", payload)["data"]
            # The gateway may return out of order, so trust ``index``.
            for position, item in enumerate(data):
                vectors[batch[item.get("index", position)]] = item["embedding"]
        return vectors

    def _one_input(self, text: str, image_b64: str | None) -> str:
        """One request item: text, a data URL, or both newline-joined."""
        url = f"data:image/png;base64,{image_b64}" if image_b64 else ""
        if url and text:
            return f"{text}\n{url}"
        return url or text

    def embed_query(self, text: str) -> list[float] | None:
        """Embed a search query.

        ``input_type="query"`` rather than ``"passage"``: the model encodes the
        two asymmetrically, and mixing them degrades retrieval quietly.
        """
        return self.embed([text], input_type="query")[0]

    def describe_image(self, prompt: str, image_b64: str) -> str:
        """Ask the chat model about an image, for figures the extractor cannot read.

        Uses the ``NON_REASONING`` triplet, not ``EMBED``: this is a
        vision-language chat call and the embed model cannot serve it.  Reading
        a chart is transcription rather than reasoning, so it belongs on the
        cheaper tier — and both triplets fall back to ``BASE_URL`` /
        ``MODEL_NAME`` when only those are set.
        """
        from auto_ontology.utils.model_config import resolve

        response = self._http.post(
            f"{resolve('NON_REASONING', 'ENDPOINT').rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {resolve('NON_REASONING', 'API_KEY')}"},
            json={
                "model": resolve("NON_REASONING", "MODEL"),
                "max_tokens": 2048,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"] or ""

    def describe_images(
        self, prompt: str, images_b64: list[str], *, workers: int = _VISION_WORKERS
    ) -> list[str]:
        """:meth:`describe_image` over many crops at once, in input order.

        These calls are latency-bound, not throughput-bound — measured at 14.5s
        each serially against 2.7s amortised at eight in flight, a 5.4x
        difference on identical output.  A document with a hundred figures is
        the difference between twenty minutes and four.

        A crop the gateway could not read comes back as ``""`` rather than
        raising, so one bad figure costs its own row and nothing else.
        """
        if not images_b64:
            return []

        def one(b64: str) -> str:
            try:
                return self.describe_image(prompt, b64)
            except Exception:
                logger.exception("Vision call failed — leaving this crop empty.")
                return ""

        with ThreadPoolExecutor(max_workers=min(workers, len(images_b64))) as pool:
            return list(pool.map(one, images_b64))


_client: EmbedClient | None = None


def get_embed_client() -> EmbedClient:
    """The process-wide client, built on first use.

    Cached so one ``httpx.Client`` and its connection pool serve a whole ingest
    rather than one per document.
    """
    global _client
    if _client is None:
        _client = EmbedClient()
    return _client
