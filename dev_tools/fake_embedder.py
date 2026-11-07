# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A deterministic stand-in for NVIDIA's ``/v1/embeddings`` endpoint.

Both vector-store backends can be pointed at it: the Postgres backend through
``EMBED_ENDPOINT`` (the app embeds), the Elasticsearch backend through the
``url`` of its ``nvidia`` inference endpoint (Elasticsearch embeds). Because
both then embed with the same function, the two backends must return the same
neighbours for the same query -- which is what makes it a parity check, not
just a smoke test.

Each word and character trigram gets a fixed pseudo-random direction, and a
text's vector is the normalised sum of its features'. That is crude, but it is
not random: texts that share words land near each other, so ranking assertions
mean something.

The vectors are *dense*, like a real model's, and that matters. An earlier
version hashed each feature into one of 2048 buckets, leaving a handful of
non-zero dimensions; Elasticsearch's default index quantises vectors to one
bit per dimension (BBQ), which suits dense embeddings and loses a third of the
true neighbours on sparse ones -- a recall gap the real NVIDIA embeddings do
not have.

Every request is recorded with its ``input_type``. NVIDIA's models are
asymmetric, and a store that embeds a row as a ``query`` raises no error -- it
just ranks worse -- so the tests assert on the recorded types rather than
trusting that the right one was sent.

Run standalone for manual checks::

    uv run python -m dev_tools.fake_embedder --port 8765
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

DEFAULT_DIMENSIONS = 2048

_WORD = re.compile(r"[a-z0-9]+")


@functools.lru_cache(maxsize=65536)
def _direction(feature: str, dimensions: int) -> np.ndarray:
    """*feature*'s fixed pseudo-random direction, the same in every process."""
    seed = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest())
    return np.random.default_rng(seed).standard_normal(dimensions)


def embed(text: str, dimensions: int = DEFAULT_DIMENSIONS) -> list[float]:
    """The fake embedding of *text*: its words and trigrams' directions, unit length."""
    features: list[str] = []
    for word in _WORD.findall(text.lower()):
        # Words count double: sharing a word should outweigh sharing a trigram.
        features.extend([f"w:{word}"] * 2)
        padded = f"#{word}#"
        features.extend(f"t:{padded[i : i + 3]}" for i in range(len(padded) - 2))
    vector = sum(_direction(f, dimensions) for f in features or ["<empty>"])
    norm = float(np.linalg.norm(vector)) or 1.0
    return (vector / norm).tolist()


@dataclass
class Call:
    """One request the fake served."""

    input_type: str | None
    inputs: list[str]
    model: str | None
    authorized: bool


@dataclass
class FakeEmbedder:
    """An embeddings server on a background thread; use as a context manager.

    Bind to ``0.0.0.0`` when an Elasticsearch container has to reach it (it
    then calls back through ``host.docker.internal``); the default loopback is
    enough for the Postgres backend, which embeds in-process.
    """

    host: str = "127.0.0.1"
    port: int = 0
    dimensions: int = DEFAULT_DIMENSIONS
    calls: list[Call] = field(default_factory=list)
    _failures: list[int] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _server: ThreadingHTTPServer | None = None

    def fail_next(self, status: int, times: int = 1) -> None:
        """Answer the next *times* requests with HTTP *status*."""
        with self._lock:
            self._failures.extend([status] * times)

    def input_types(self) -> list[str | None]:
        with self._lock:
            return [call.input_type for call in self.calls]

    def reset(self) -> None:
        with self._lock:
            self.calls.clear()
            self._failures.clear()

    @property
    def bound_port(self) -> int:
        assert self._server is not None, "server not started"
        return self._server.server_address[1]

    def url(self, host: str | None = None) -> str:
        """The ``/v1`` base URL, as the app's ``EMBED_ENDPOINT`` spells it."""
        return f"http://{host or '127.0.0.1'}:{self.bound_port}/v1"

    def __enter__(self) -> "FakeEmbedder":
        self._server = ThreadingHTTPServer((self.host, self.port), _handler(self))
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def _take_failure(self) -> int | None:
        with self._lock:
            return self._failures.pop(0) if self._failures else None

    def _record(self, call: Call) -> None:
        with self._lock:
            self.calls.append(call)


def _handler(fake: FakeEmbedder) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 -- http.server's naming
            if not self.path.rstrip("/").endswith("/embeddings"):
                self._send(404, {"error": f"no route {self.path}"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            body: dict[str, Any] = json.loads(self.rfile.read(length) or b"{}")
            raw = body.get("input")
            inputs = [raw] if isinstance(raw, str) else list(raw or [])
            fake._record(
                Call(
                    input_type=body.get("input_type"),
                    inputs=inputs,
                    model=body.get("model"),
                    authorized=bool(self.headers.get("Authorization")),
                )
            )
            status = fake._take_failure()
            if status is not None:
                self._send(status, {"error": f"injected {status}"})
                return
            self._send(
                200,
                {
                    "object": "list",
                    "model": body.get("model"),
                    "data": [
                        {
                            "object": "embedding",
                            "index": i,
                            "embedding": embed(text, fake.dimensions),
                        }
                        for i, text in enumerate(inputs)
                    ],
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                },
            )

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    args = parser.parse_args()
    with FakeEmbedder(args.host, args.port, args.dimensions) as fake:
        print(f"fake embedder on {fake.url(args.host)}", flush=True)
        threading.Event().wait()


if __name__ == "__main__":
    main()
