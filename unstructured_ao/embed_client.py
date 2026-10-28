# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embeddings and figure reading through the hosted inference API.

The ingest pipeline uses this client to embed extracted text and to transcribe
figures. Keeping request shaping here gives every extractor the same embedding
space.

Modality
--------
``text`` by default.  ``text_image`` is also available through
:meth:`EmbedClient.embed`; its wire format is one string, text and data URL
joined by a newline, plus a top-level ``modality`` field.
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

#: The model-config triplet figure reading uses; see :meth:`EmbedClient.describe_image`.
_VISION_PREFIX = "UNSTRUCTURED_VISION"

#: Seconds.  Generous because a cold model behind the gateway can take a while,
#: and because an ingest that dies at document 11 of 14 wastes more than it
#: saves.
_TIMEOUT = 300.0


#: Longest ``Retry-After`` honoured, in seconds, so one header cannot park a
#: worker thread indefinitely.
_MAX_RETRY_AFTER = 60.0


def _retryable(exc: Exception) -> bool:
    """Whether *exc* is safe to retry without paying for the call twice.

    A read timeout is not: the server most likely did the work, and with
    :data:`_TIMEOUT` each repeat can cost another five minutes.
    """
    import httpx

    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return isinstance(
        exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
    )


def _retry_after(exc: Exception) -> float | None:
    """The server's requested wait in seconds, when it sent one as a number."""
    response = getattr(exc, "response", None)
    value = response.headers.get("Retry-After") if response is not None else None
    try:
        return min(max(float(value), 0.0), _MAX_RETRY_AFTER) if value else None
    except ValueError:
        return None


def post_with_retry(http: Any, url: str, **kwargs: Any) -> dict[str, Any]:
    """POST *url* and return its JSON, backing off on 429, 5xx and failed connects.

    A single rate-limited request would otherwise fail a whole document after
    every paid call before it had already run.
    """
    import random
    import time

    from auto_ontology.utils.llm_invoke import RETRY_MAX_ATTEMPTS

    for attempt in range(RETRY_MAX_ATTEMPTS):
        try:
            response = http.post(url, **kwargs)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            if not _retryable(exc) or attempt == RETRY_MAX_ATTEMPTS - 1:
                raise
            wait = _retry_after(exc)
            if wait is None:
                wait = 2 ** (attempt + 1) + random.uniform(0, 1)
            logger.warning(
                "Retryable error from %s on attempt %d/%d — retrying in %.1fs: %s",
                url,
                attempt + 1,
                RETRY_MAX_ATTEMPTS,
                wait,
                str(exc)[:120],
            )
            time.sleep(wait)

    raise RuntimeError("unreachable")  # pragma: no cover


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
        return post_with_retry(
            self._http,
            f"{self._endpoint}/{path}",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=payload,
        )

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
                # A page-grained row can exceed that limit, and the gateway
                # answers 422 for the whole batch rather than only that row.
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
        """Ask a vision model about an image, for figures the extractor cannot read.

        Uses the ``UNSTRUCTURED_VISION`` triplet. Its endpoint and key fall
        back like any other, but its model never inherits ``MODEL_NAME`` or
        ``DEFAULT_MODELS_MODEL``: those are usually text-only, which the
        endpoint rejects with a 400 and every figure is silently dropped.
        Unset, the model follows the key — a vision model on either endpoint.
        """
        from auto_ontology.utils.model_config import resolve

        endpoint = resolve(_VISION_PREFIX, "ENDPOINT").rstrip("/")
        data = post_with_retry(
            self._http,
            f"{endpoint}/chat/completions",
            headers={"Authorization": f"Bearer {resolve(_VISION_PREFIX, 'API_KEY')}"},
            json={
                "model": resolve(_VISION_PREFIX, "MODEL", use_shared_default=False),
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
        return data["choices"][0]["message"]["content"] or ""

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
            transcripts = list(pool.map(one, images_b64))
        empty = sum(1 for text in transcripts if not text)
        if empty:
            logger.warning(
                "%d of %d figure(s) came back unread; their rows are dropped.",
                empty,
                len(transcripts),
            )
        return transcripts


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
