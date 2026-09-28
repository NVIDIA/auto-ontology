# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompts and structured-output schemas for SQL result visualization."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

ChartType = Literal["bar", "hbar", "line", "area", "grouped-bar", "delta", "kpi"]
ValueFormat = Literal["number", "compact", "percent", "currency"]


class VisualizationDecisionModel(BaseModel):
    """Whether the complete SQL result or a chart best answers the question."""

    model_config = ConfigDict(extra="forbid")

    render_as: Literal["chart", "table"] = Field(
        ...,
        description=(
            "Use table when answering requires preserving the result's descriptive "
            "columns; use chart for a requested comparison, trend, ranking, "
            "distribution, aggregate, or visualization."
        ),
    )


class PlotRecommendation(BaseModel):
    """One chart suggestion: type + column encodings; data is filled from SQL rows."""

    model_config = ConfigDict(extra="forbid")

    plot_type: ChartType = Field(
        ...,
        description=(
            "Chart type. Use kpi for a single scalar / one-entity result. "
            "Use grouped-bar when a hue/series column splits values. "
            "Use hbar for rankings with many category labels. "
            "Use line/area for time-series. Use delta for signed gain/loss."
        ),
    )
    title: str = Field(..., description="Short human-readable chart title.")
    x: Optional[str] = Field(
        default=None,
        description="Column for the category / time axis. Required except for kpi.",
    )
    y: Optional[str] = Field(
        default=None,
        description="Numeric value column. Required except for kpi.",
    )
    hue: Optional[str] = Field(
        default=None,
        description=(
            "Optional series/group column (maps to multiple series, e.g. grouped-bar)."
        ),
    )
    y_format: ValueFormat = Field(
        default="compact",
        description="How to format Y values on the axis.",
    )


class VisualizationRecommendation(BaseModel):
    """The single best chart suggestion for the SQL result set."""

    model_config = ConfigDict(extra="forbid")

    plots: list[PlotRecommendation] = Field(
        ...,
        min_length=1,
        max_length=1,
        description=(
            "Exactly one plot suggestion — the best visualization for the question. "
            "Do not return multiple chart types for the same data."
        ),
    )


VISUALIZATION_DECISION_PROMPT = """\
I have a dataset resulting from the following SQL query:
{sql}

The natural language question corresponding to this query is:
{question}

The dataset has the following structure:
Size: {df_size}
Columns: {df_columns}

Decide whether the answer should render as a chart or as the complete result table.

Use "table" when:
- The user asks for records, rows, a list, inventory, specifications, attributes,
  configuration, details, or "all" information about each entity.
- Multiple descriptive columns are essential to answering the question.
- A chart would select only one numeric field and omit other requested fields.
- The user explicitly requests tabular output.

Use "chart" when:
- The user asks for a comparison, trend, ranking, distribution, aggregate, change
  over time, or a specific visualization.
- A single scalar or one-entity metric is naturally represented as a KPI.

Numeric columns such as counts do not by themselves make a detail/listing request
chartable. Choose "table" whenever charting would answer only one facet of the
question or discard requested information.
"""

VISUALIZATION_PROMPT = """\
I have a dataset resulting from the following SQL query:
{sql}

The natural language question corresponding to this query is:
{question}

The dataset has the following structure:
Size: {df_size}
Columns: {df_columns}

Based on this dataset, propose the SINGLE best visualization for the question.

IMPORTANT:
- Stick strictly to the original question. Do not add proportion, percentage, or \
distribution analysis unless explicitly requested.
- If the natural language question asks for a specific visualization type, use that type.
- Column names in x/y/hue MUST be exact matches from Columns above.
- Do NOT return both a bar and a line for the same data — pick one.

Preferences and guidelines:
- Single scalar or one-row / one-entity result: use plot_type "kpi" (x/y optional).
- Time-series (date/time or ordered numeric on X): prefer line. Prefer area when \
the question is about cumulative totals over time.
- Pure categorical X (few discrete labels): do NOT use line. Prefer bar; use hbar \
for rankings or long category labels; use grouped-bar when a hue column splits series.
- Signed change / gain-loss comparisons: use delta.
- Ordinal categorical X with many categories (> 10): line is acceptable; otherwise \
prefer bar/hbar.
- Never return a line chart for a single value.
- CRITICAL: Do NOT modify or reinterpret the original question. ONLY visualize \
exactly what was asked. If the question asks for individual values, counts, or \
totals, do NOT create proportion/percentage/share charts.

Output requirements:
- Return exactly ONE plot in the "plots" array (the best fit).
- Each plot needs plot_type, title, and (except kpi) x and y column names.
- hue may be null when not needed.
"""
