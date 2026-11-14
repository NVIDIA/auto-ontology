# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``INGEST_INTERVAL_HOURS`` sets the ingest cadence.

``.env.example`` documents the variable; the scheduler has to read it, and a
bad value has to fall back to the 24h default rather than stop the service.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import pytest

from auto_ontology.ingestion_service.data_scheduler import (
    DEFAULT_INGEST_INTERVAL,
    INGEST_INTERVAL_ENV,
    DataScheduler,
    resolve_ingest_interval,
)


def test_unset_means_24h(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(INGEST_INTERVAL_ENV, raising=False)
    assert resolve_ingest_interval() == timedelta(hours=24)
    assert DataScheduler()._interval == DEFAULT_INGEST_INTERVAL


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("6", timedelta(hours=6)),
        ("0.5", timedelta(minutes=30)),
        (" 12 ", timedelta(hours=12)),
        ("1e-9", timedelta(microseconds=4)),
    ],
)
def test_positive_hours_are_honoured(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: timedelta
) -> None:
    monkeypatch.setenv(INGEST_INTERVAL_ENV, raw)
    assert resolve_ingest_interval() == expected
    assert DataScheduler()._interval == expected


@pytest.mark.parametrize(
    "raw", ["0", "-3", "daily", "nan", "inf", "1e-12", "1e11", "1e8"]
)
def test_invalid_value_warns_and_defaults(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, raw: str
) -> None:
    monkeypatch.setenv(INGEST_INTERVAL_ENV, raw)
    with caplog.at_level(logging.WARNING):
        assert resolve_ingest_interval() == DEFAULT_INGEST_INTERVAL
        assert DataScheduler()._interval == DEFAULT_INGEST_INTERVAL
    assert INGEST_INTERVAL_ENV in caplog.text


# timedelta(0) is falsy: an explicit argument must win on presence, not truthiness.
@pytest.mark.parametrize("interval", [timedelta(hours=1), timedelta(0)])
def test_explicit_interval_wins_over_env(
    monkeypatch: pytest.MonkeyPatch, interval: timedelta
) -> None:
    monkeypatch.setenv(INGEST_INTERVAL_ENV, "6")
    assert DataScheduler(interval)._interval == interval
