# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build ResultChart ChartSpec dicts from a DataFrame + LLM plot recommendations."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import pandas as pd

from gsf.retrieval.text_to_sql.visualization.helpers import map_column
from gsf.retrieval.text_to_sql.visualization.prompts import PlotRecommendation

logger = logging.getLogger(__name__)

MAX_ROWS = 60
MAX_SERIES = 6
CHART_COLORS = ("green", "blue", "amber", "red", "neutral")


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
    rows: list[dict[str, Any]], y_keys: list[str]
) -> list[dict[str, Any]]:
    if len(rows) <= MAX_ROWS:
        return rows

    def magnitude(row: dict[str, Any]) -> float:
        total = 0.0
        for key in y_keys:
            val = row.get(key)
            if isinstance(val, (int, float)):
                total += abs(float(val))
        return total

    ranked = sorted(rows, key=magnitude, reverse=True)
    return ranked[:MAX_ROWS]


def _build_kpi_spec(
    df: pd.DataFrame, plot: PlotRecommendation
) -> dict[str, Any] | None:
    if df.empty:
        return None
    row = df.iloc[0]
    kpis: list[dict[str, str]] = []
    for col in df.columns:
        kpis.append(
            {
                "label": str(col),
                "value": _format_kpi_value(row[col]),
            }
        )
        if len(kpis) >= 4:
            break
    if not kpis:
        return None
    return {
        "title": plot.title or "Result",
        "kpis": kpis,
    }


def _aggregate(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (data rows, series defs) for a chart."""
    work = df[[c for c in {x_col, y_col, hue_col} if c]].dropna(how="any").copy()
    if work.empty:
        return [], []

    y_numeric = pd.to_numeric(work[y_col], errors="coerce")
    is_numeric = y_numeric.notna().all()
    if is_numeric:
        work[y_col] = y_numeric

    if hue_col:
        if is_numeric:
            pivot = (
                work.groupby([x_col, hue_col], dropna=False)[y_col]
                .sum()
                .unstack(fill_value=0)
            )
        else:
            pivot = (
                work.groupby([x_col, hue_col], dropna=False)
                .size()
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
        return _truncate_rows(data, safe_keys), series

    # Single series
    if is_numeric:
        grouped = work.groupby(x_col, dropna=False)[y_col].sum()
    else:
        grouped = work.groupby(x_col, dropna=False).size()
        y_col = f"{y_col}_count"

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
    return _truncate_rows(data, [y_col]), series


def build_chart_spec(
    df: pd.DataFrame,
    plot: PlotRecommendation,
) -> dict[str, Any] | None:
    """Convert one plot recommendation + dataframe into a ResultChart JSON dict."""
    plot_type = plot.plot_type

    if plot_type == "kpi":
        return _build_kpi_spec(df, plot)

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

    data, series = _aggregate(df, x_col, y_col, hue_col)
    if not data or not series:
        logger.info("build_chart_spec: empty aggregation for %s — skip", plot.title)
        return None

    if plot_type == "delta" and len(series) > 1:
        series = series[:1]

    return {
        "type": plot_type,
        "title": plot.title,
        "x": {"key": x_col, "label": x_col},
        "y": {"label": plot.y or y_col, "format": plot.y_format or "compact"},
        "series": series,
        "data": data,
    }


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
