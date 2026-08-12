# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keying a table the catalog and inference both left without an identity.

Every "is not a primary key" in the evaluation sweep traces to the same shape: the
catalog recorded no key, inference declined to pick one, and the table then reached
KumoRFM with no identity at all -- unlinkable, and unusable as a prediction entity.
"""

import pandas as pd
import pytest

from gsf.retrieval.kumo.kumo_model import key_columns
from gsf.retrieval.kumo.predictor import _adopt_unique_keys

rfm = pytest.importorskip("kumorfm.rfm", reason="KumoRFM SDK is not installed")


def _graph(frames: dict[str, pd.DataFrame]):
    return rfm.Graph.from_data(frames, edges=[], infer_metadata=True, verbose=False)


def test_a_table_inference_left_unkeyed_is_keyed_on_its_unique_column() -> None:
    """Several ``ID`` columns and none resembling the table is what inference declines."""
    frames = {
        "playstore": pd.DataFrame(
            {
                "App": [f"com.app.{i}" for i in range(6)],
                "Category": ["GAME", "TOOL"] * 3,
                "Rating": [4.0, 3.5, 4.2, 2.9, 4.8, 3.1],
            }
        )
    }
    graph = _graph(frames)
    assert key_columns(graph["playstore"]) == []

    assert _adopt_unique_keys(graph, frames) == 1
    assert key_columns(graph["playstore"]) == ["App"]


def test_a_column_that_does_not_identify_rows_is_not_adopted() -> None:
    """A duplicated or null column would key the table on the wrong thing."""
    frames = {
        "events": pd.DataFrame(
            {"Status": ["open", "open", "shut"], "Owner": ["a", None, "c"]}
        )
    }
    graph = _graph(frames)

    assert _adopt_unique_keys(graph, frames) == 0
    assert key_columns(graph["events"]) == []


def test_an_existing_key_is_never_replaced() -> None:
    """A catalog key and an inferred key both outrank this fallback."""
    frames = {
        "orders": pd.DataFrame({"order_id": [1, 2, 3], "ref": ["r1", "r2", "r3"]})
    }
    graph = _graph(frames)
    graph["orders"].primary_key = "ref"

    assert _adopt_unique_keys(graph, frames) == 0
    assert key_columns(graph["orders"]) == ["ref"]


def test_the_column_a_schema_would_have_declared_wins() -> None:
    """Two columns identify the rows; the one named for the table is the key."""
    frames = {
        "series": pd.DataFrame(
            {"Notes": ["a", "b", "c"], "SeriesCode": ["S1", "S2", "S3"]}
        )
    }
    graph = _graph(frames)

    _adopt_unique_keys(graph, frames)

    assert key_columns(graph["series"]) == ["SeriesCode"]


def test_a_float_column_is_never_an_identity() -> None:
    """Unique measurements are not identifiers, however unique the sample happens to be."""
    frames = {
        "readings": pd.DataFrame(
            {"Temperature": [1.5, 2.5, 3.5], "Humidity": [10.1, 20.2, 30.3]}
        )
    }
    graph = _graph(frames)

    assert _adopt_unique_keys(graph, frames) == 0


def test_a_table_with_no_loaded_frame_is_left_alone() -> None:
    """Without the rows there is nothing to judge uniqueness over."""
    frames = {"orders": pd.DataFrame({"ref": ["r1", "r2", "r3"]})}
    graph = _graph(frames)

    assert _adopt_unique_keys(graph, {}) == 0
    assert key_columns(graph["orders"]) == []


def test_a_composite_catalog_key_is_left_intact() -> None:
    """The tuple #168 declares must not be replaced by a single unique column."""
    frames = {
        "people": pd.DataFrame(
            {
                "Customer ID": ["A", "B", "A"],
                "REGION": ["West", "East", "East"],
                "Row": [1, 2, 3],
            }
        )
    }
    graph = _graph(frames)
    graph["people"].primary_key = ("Customer ID", "REGION")

    assert _adopt_unique_keys(graph, frames) == 0
    assert key_columns(graph["people"]) == ["Customer ID", "REGION"]
