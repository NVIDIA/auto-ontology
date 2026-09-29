# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build ResultChart ChartSpec dicts from a DataFrame + LLM plot recommendations."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import pandas as pd

from auto_ontology.retrieval.text_to_sql.visualization.helpers import map_column
from auto_ontology.retrieval.text_to_sql.visualization.prompts import PlotRecommendation

logger = logging.getLogger(__name__)

MAX_ROWS = 60
MAX_SERIES = 6
CHART_COLORS = ("green", "blue", "amber", "red", "neutral")
# Types whose x axis is a ranking, so sorting rows by magnitude and cutting the
# tail is exactly what the chart means.
RANKING_TYPES = frozenset({"hbar", "delta"})
# Types whose x axis is a sequence. Dropping rows out of the middle would draw a
# line between points that are not adjacent, so these keep a contiguous window.
ORDERED_TYPES = frozenset({"line", "area"})


def _to_cell(value: Any) -> str | int | float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()  # type: ignore[no-any-return]
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def _format_kpi_value(value: Any) -> str:
    cell = _to_cell(value)
    if cell is None:
        return "—"
    if isinstance(cell, float):
        if cell == int(cell):
            return f"{int(cell):,}"
        return f"{cell:,.2f}"
    if isinstance(cell, int):
        return f"{cell:,}"
    return str(cell)


def _truncate_rows(
    rows: list[dict[str, Any]], y_keys: list[str], plot_type: str
) -> list[dict[str, Any]]:
    """Cap ``rows`` at ``MAX_ROWS`` without misrepresenting the x axis.

    Which rows may be dropped depends on what the axis means. A ranking keeps
    its largest rows in rank order. A sequence keeps its most recent rows, as a
    contiguous window, because reordering a time axis or punching holes in it
    would draw a line that never happened. Anything else keeps the largest rows
    but in the order they arrived, since a categorical axis is not a ranking and
    must not be silently re-sorted.
    """
    if len(rows) <= MAX_ROWS:
        return rows

    if plot_type in ORDERED_TYPES:
        return rows[-MAX_ROWS:]

    def magnitude(index: int) -> float:
        total = 0.0
        for key in y_keys:
            val = rows[index].get(key)
            if isinstance(val, (int, float)):
                total += abs(float(val))
        return total

    ranked = sorted(range(len(rows)), key=magnitude, reverse=True)[:MAX_ROWS]
    if plot_type not in RANKING_TYPES:
        ranked.sort()
    return [rows[i] for i in ranked]


def _build_kpi_spec(
    df: pd.DataFrame, plot: PlotRecommendation
) -> dict[str, Any] | None:
    """Tile the columns of a one-row result. Callers must check the row count:
    this reads row zero only, so a longer frame would lose every other row."""
    if df.empty:
        return None
    row = df.iloc[0]
    measures = [
        col
        for col in df.columns
        if bool(pd.to_numeric(df[col], errors="coerce").notna().all())
    ]
    # A grouping column says which group the number describes — it is context,
    # not a metric, so it captions the card instead of claiming a tile. With no
    # measure at all the result is itself textual, so tile what there is.
    context = (
        " · ".join(
            f"{col}: {_format_kpi_value(row[col])}"
            for col in df.columns
            if col not in measures
        )
        if measures
        else ""
    )

    kpis = [
        {"label": str(col), "value": _format_kpi_value(row[col])}
        for col in (measures or list(df.columns))[:4]
    ]
    if not kpis:
        return None
    if context:
        return {
            "title": plot.title or "Result",
            "subtitle": context,
            "kpis": kpis,
        }
    return {
        "title": plot.title or "Result",
        "kpis": kpis,
    }


def _is_constant(data: list[dict[str, Any]], key: str) -> bool:
    """True when ``key`` holds the same number on every one of several rows.

    Such a chart compares nothing — every bar is the same height.
    """
    values = [row[key] for row in data if isinstance(row.get(key), (int, float))]
    return len(values) > 1 and len(set(values)) == 1


def _aggregate(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate the result set down to the (rows, series) a chart draws.

    Returns empty lists when there is nothing to plot — including when ``y_col``
    holds no numeric measure. Such a result is a listing, and counting its rows
    to have something to draw would chart a question nobody asked.
    """
    work = df[[c for c in {x_col, y_col, hue_col} if c]].dropna(how="any").copy()
    if work.empty:
        return [], []

    y_numeric = pd.to_numeric(work[y_col], errors="coerce")
    if not bool(y_numeric.notna().all()):
        logger.info("aggregate: %r holds no numeric measure — skip", y_col)
        return [], []
    work[y_col] = y_numeric

    if hue_col:
        pivot = (
            work.groupby([x_col, hue_col], dropna=False)[y_col]
            .sum()
            .unstack(fill_value=0)
        )
        pivot = pivot.fillna(0)
        # Cap series count
        if pivot.shape[1] > MAX_SERIES:
            totals = pivot.abs().sum().sort_values(ascending=False)
            keep = list(totals.index[:MAX_SERIES])
            pivot = pivot[keep]

        series_keys = [str(_to_cell(c) or c) for c in pivot.columns]
        # Sanitize keys for JSON object fields
        safe_keys: list[str] = []
        for i, key in enumerate(series_keys):
            safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in key)
            if not safe or safe in safe_keys:
                safe = f"series_{i}"
            safe_keys.append(safe)

        data: list[dict[str, Any]] = []
        for idx, row in pivot.iterrows():
            item: dict[str, Any] = {x_col: _to_cell(idx)}
            for safe, col in zip(safe_keys, pivot.columns):
                item[safe] = _to_cell(row[col])
            data.append(item)

        series = [
            {
                "key": safe,
                "label": label,
                "color": CHART_COLORS[i % len(CHART_COLORS)],
            }
            for i, (safe, label) in enumerate(zip(safe_keys, series_keys))
        ]
        return data, series

    grouped = work.groupby(x_col, dropna=False)[y_col].sum()
    data = [
        {x_col: _to_cell(idx), y_col: _to_cell(val)} for idx, val in grouped.items()
    ]
    series = [
        {
            "key": y_col,
            "label": str(y_col),
            "color": CHART_COLORS[0],
        }
    ]
    return data, series


def build_chart_spec(
    df: pd.DataFrame,
    plot: PlotRecommendation,
) -> dict[str, Any] | None:
    """Convert one plot recommendation + dataframe into a ResultChart JSON dict."""
    plot_type = plot.plot_type

    if plot_type == "kpi":
        if len(df) == 1:
            return _build_kpi_spec(df, plot)
        # A KPI card tiles a single row. The model picks it for whole result
        # sets anyway, which would publish row zero as though it were the whole
        # answer and drop every other row without a trace. Chart it instead —
        # and if there is nothing chartable, the caller falls back to the table,
        # which is the honest rendering of a long list.
        logger.info(
            "build_chart_spec: ignoring kpi recommendation for %d rows (%s)",
            len(df),
            plot.title,
        )
        plot_type = "bar"

    columns = list(df.columns)
    x_col = map_column(plot.x, columns)
    y_col = map_column(plot.y, columns)
    hue_col = map_column(plot.hue, columns) if plot.hue else None

    if not x_col or not y_col:
        logger.info(
            "build_chart_spec: missing x/y (got x=%r y=%r) for %s — skip",
            plot.x,
            plot.y,
            plot_type,
        )
        return None

    # Force grouped-bar when hue is present and type is plain bar.
    if hue_col and plot_type == "bar":
        plot_type = "grouped-bar"
    if hue_col and plot_type == "delta":
        # delta only supports one series
        hue_col = None

    rows, series = _aggregate(df, x_col, y_col, hue_col)
    if not rows or not series:
        logger.info("build_chart_spec: empty aggregation for %s — skip", plot.title)
        return None

    if plot_type == "delta" and len(series) > 1:
        series = series[:1]

    # A flat line still says something useful — that the metric never moved —
    # so only the types that compare categories side by side are dropped.
    if (
        len(series) == 1
        and plot_type not in ORDERED_TYPES
        and _is_constant(rows, str(series[0]["key"]))
    ):
        logger.info(
            "build_chart_spec: %r is constant across %d rows — skip (%s)",
            series[0]["key"],
            len(rows),
            plot.title,
        )
        return None

    notes: list[str] = []
    data = _truncate_rows(rows, [str(s["key"]) for s in series], plot_type)
    if len(data) < len(rows):
        # Say which rows were dropped, otherwise the chart reads as the whole
        # result and any caption counting its points understates the total.
        kept = "Last" if plot_type in ORDERED_TYPES else "Top"
        notes.append(f"{kept} {len(data)} of {len(rows)} rows")

    spec: dict[str, Any] = {
        "type": plot_type,
        "title": plot.title,
        "x": {"key": x_col, "label": x_col},
        "y": {"label": plot.y or y_col, "format": plot.y_format or "compact"},
        "series": series,
        "data": data,
    }
    if notes:
        spec["subtitle"] = " · ".join(notes)
    return spec


def generate_chart_specs(
    df: pd.DataFrame,
    plots: list[PlotRecommendation],
) -> list[dict[str, Any]]:
    """Build all valid ChartSpec / KPI dicts from LLM recommendations."""
    specs: list[dict[str, Any]] = []
    for plot in plots:
        try:
            spec = build_chart_spec(df, plot)
            if spec:
                specs.append(spec)
        except Exception as exc:  # noqa: BLE001 — soft-fail per plot
            logger.error("generate_chart_specs: failed for %s: %s", plot, exc)
    return specs
