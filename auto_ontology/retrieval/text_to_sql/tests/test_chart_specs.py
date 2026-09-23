# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chart spec construction: which results get a chart, and which fall back."""

from __future__ import annotations

import pandas as pd

from auto_ontology.retrieval.text_to_sql.visualization.chart_specs import (
    build_chart_spec,
)
from auto_ontology.retrieval.text_to_sql.visualization.prompts import PlotRecommendation


def _plot(**kwargs: object) -> PlotRecommendation:
    defaults: dict[str, object] = {"plot_type": "bar", "title": "T"}
    defaults.update(kwargs)
    return PlotRecommendation(**defaults)  # type: ignore[arg-type]


def _wide(x_values: list[str], y_values: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"x": x_values, "sales": y_values})


def test_numeric_y_is_summed_per_category() -> None:
    df = pd.DataFrame(
        {
            "OrderType": ["Promotional", "Non-Promotional"],
            "AverageOrderValue": [3100.5, 2432.75],
        }
    )
    spec = build_chart_spec(
        df, _plot(x="OrderType", y="AverageOrderValue", y_format="currency")
    )

    assert spec is not None
    assert spec["type"] == "bar"
    assert spec["x"]["key"] == "OrderType"
    assert spec["y"] == {"label": "AverageOrderValue", "format": "currency"}
    assert "subtitle" not in spec
    assert spec["series"][0]["key"] == "AverageOrderValue"
    assert spec["data"] == [
        {"OrderType": "Non-Promotional", "AverageOrderValue": 2432.75},
        {"OrderType": "Promotional", "AverageOrderValue": 3100.5},
    ]


# --- results with no numeric measure are listings, not charts ----------------


def test_no_chart_when_y_holds_no_numeric_measure() -> None:
    """ "The phone number of each customer" is a listing; charting it would mean
    inventing a count nobody asked for."""
    df = pd.DataFrame(
        {
            "customername": ["Aakriti", "Bhaamini", "Chandra", "Dana"],
            "phonenumber": ["(216) 555-0100", "(216) 555-0100", "(215) 555-0100", "x"],
        }
    )
    assert build_chart_spec(df, _plot(x="customername", y="phonenumber")) is None


def test_no_chart_for_a_benchmark_to_winner_mapping() -> None:
    df = pd.DataFrame(
        {
            "benchmark_name": ["b1", "b2", "b3", "b4"],
            "gpu_model": ["A100", "A100", "H100", "L40S"],
        }
    )
    assert build_chart_spec(df, _plot(x="benchmark_name", y="gpu_model")) is None


def test_no_chart_when_the_category_axis_already_varies() -> None:
    """The x column repeating does not make a listing chartable.

    "Which category does each customer belong to" named a category on x and a
    customer on y; counting rows per category answers a different question.
    """
    df = pd.DataFrame(
        {
            "customercategoryname": ["Novelty Shop", "Novelty Shop", "Corporate"],
            "customername": ["c1", "c2", "c3"],
        }
    )
    plot = _plot(x="customercategoryname", y="customername")

    assert build_chart_spec(df, plot) is None


def test_no_chart_for_a_one_to_one_mapping() -> None:
    df = pd.DataFrame(
        {"CustomerName": ["c1", "c2", "c3"], "PhoneNumber": ["p1", "p2", "p3"]}
    )
    assert build_chart_spec(df, _plot(x="CustomerName", y="PhoneNumber")) is None


# --- constant measures compare nothing --------------------------------------


def test_no_chart_when_a_real_measure_is_constant() -> None:
    """ "How many phone numbers does each customer have" is 1 for everyone."""
    df = pd.DataFrame(
        {"customername": ["c1", "c2", "c3", "c4"], "phone_count": [1, 1, 1, 1]}
    )
    assert build_chart_spec(df, _plot(x="customername", y="phone_count")) is None


def test_constant_measure_still_charts_as_a_time_series() -> None:
    """A flat line says the metric never moved, which is worth drawing."""
    df = pd.DataFrame(
        {"day": ["2026-01-01", "2026-01-02", "2026-01-03"], "orders": [7, 7, 7]}
    )
    spec = build_chart_spec(df, _plot(plot_type="line", x="day", y="orders"))

    assert spec is not None
    assert spec["type"] == "line"
    assert len(spec["data"]) == 3


def test_varying_measure_is_unaffected() -> None:
    spec = build_chart_spec(_wide(["a", "b", "c"], [1, 1, 2]), _plot(x="x", y="sales"))

    assert spec is not None
    assert len(spec["data"]) == 3


# --- truncation must not misrepresent the x axis -----------------------------


def test_truncated_rows_are_disclosed_in_the_subtitle() -> None:
    spec = build_chart_spec(
        _wide([f"c{i:02d}" for i in range(70)], list(range(70))),
        _plot(x="x", y="sales"),
    )

    assert spec is not None
    assert len(spec["data"]) == 60
    assert spec["subtitle"] == "Top 60 of 70 rows"


def test_bar_truncation_keeps_the_largest_rows_in_their_original_order() -> None:
    """A categorical axis is not a ranking, so it must not be re-sorted."""
    labels = [f"c{i:02d}" for i in range(70)]
    spec = build_chart_spec(
        _wide(labels, list(range(70))), _plot(plot_type="bar", x="x", y="sales")
    )

    assert spec is not None
    kept = [row["x"] for row in spec["data"]]
    assert kept == sorted(kept)
    # The 10 smallest values are the ones dropped.
    assert kept == labels[10:]


def test_ranking_truncation_sorts_by_magnitude() -> None:
    labels = [f"c{i:02d}" for i in range(70)]
    spec = build_chart_spec(
        _wide(labels, list(range(70))), _plot(plot_type="hbar", x="x", y="sales")
    )

    assert spec is not None
    values = [row["sales"] for row in spec["data"]]
    assert values == sorted(values, reverse=True)
    assert values[0] == 69


def test_time_series_truncation_keeps_a_contiguous_recent_window() -> None:
    """Reordering or holing a time axis would draw a line that never happened."""
    days = [f"2026-01-{i + 1:02d}" for i in range(70)]
    # Largest values sit at the start, so magnitude ranking would reorder badly.
    spec = build_chart_spec(
        _wide(days, list(range(70, 0, -1))), _plot(plot_type="line", x="x", y="sales")
    )

    assert spec is not None
    assert [row["x"] for row in spec["data"]] == days[10:]
    assert spec["subtitle"] == "Last 60 of 70 rows"


# --- kpi cards tile exactly one row -----------------------------------------


def test_kpi_tiles_a_single_row_result() -> None:
    df = pd.DataFrame({"TotalOrders": [1234]})
    spec = build_chart_spec(df, _plot(plot_type="kpi", title="Total"))

    assert spec == {
        "title": "Total",
        "kpis": [{"label": "TotalOrders", "value": "1,234"}],
    }


def test_kpi_captions_the_grouping_column_instead_of_tiling_it() -> None:
    """A one-sided comparison must not present its group label as a metric.

    The model asks to compare promotion_status, only one group comes back, and
    the card has to say which one rather than tiling it like a number.
    """
    df = pd.DataFrame(
        {
            "promotion_status": ["Without Promotion"],
            "average_order_value": [2432.7537433711],
        }
    )
    plot = _plot(
        plot_type="kpi",
        x="promotion_status",
        y="average_order_value",
        title="Average Order Value: Promotion vs. No Promotion (H1 2016)",
    )
    spec = build_chart_spec(df, plot)

    assert spec == {
        "title": "Average Order Value: Promotion vs. No Promotion (H1 2016)",
        "subtitle": "promotion_status: Without Promotion",
        "kpis": [{"label": "average_order_value", "value": "2,432.75"}],
    }


def test_kpi_tiles_text_when_the_result_has_no_measure() -> None:
    """A purely textual one-row answer still deserves its value on the card."""
    df = pd.DataFrame({"top_salesperson": ["Hudson Onslow"]})
    spec = build_chart_spec(df, _plot(plot_type="kpi", title="Top Salesperson"))

    assert spec == {
        "title": "Top Salesperson",
        "kpis": [{"label": "top_salesperson", "value": "Hudson Onslow"}],
    }


def test_kpi_is_refused_for_a_multi_row_directory() -> None:
    """Row zero is not the answer to "the phone number of each customer"."""
    df = pd.DataFrame(
        {
            "customername": ["Aakriti Byrraju", "Bhaamini Beniwal", "Chandra Bhat"],
            "phonenumber": ["(216) 555-0100", "(216) 555-0101", "(216) 555-0102"],
        }
    )
    plot = _plot(plot_type="kpi", title="Customer Phone Directory")

    assert build_chart_spec(df, plot) is None


def test_kpi_on_multiple_rows_falls_through_to_a_real_chart() -> None:
    df = pd.DataFrame({"category": ["Gift", "Novelty"], "customers": [120, 402]})
    spec = build_chart_spec(
        df, _plot(plot_type="kpi", x="category", y="customers", title="Customers")
    )

    assert spec is not None
    assert spec["type"] == "bar"
    assert spec["data"] == [
        {"category": "Gift", "customers": 120},
        {"category": "Novelty", "customers": 402},
    ]


def test_missing_columns_skip_the_chart() -> None:
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert build_chart_spec(df, _plot(x="nope", y="b")) is None
