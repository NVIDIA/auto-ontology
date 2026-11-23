# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for parsing SQL result payloads into a clean DataFrame."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# A number written as text: an optional minus, no leading zero, and commas only
# as thousands separators. "02134" (a ZIP code) and "1,2" are not numbers.
_NUMBER_TEXT = re.compile(
    r"-?(?:0|[1-9]\d{0,2}(?:,\d{3})+|[1-9]\d*)(?:\.\d*)?|-?\.\d+", re.ASCII
)


def clean_and_convert(value: Any) -> Any:
    """Strip spaces and coerce strings that read as numbers to int/float."""
    if isinstance(value, str):
        cleaned = value.strip()
        if _NUMBER_TEXT.fullmatch(cleaned):
            number = float(cleaned.replace(",", ""))
            return int(number) if number == int(number) else number
        return cleaned
    if isinstance(value, (int, float)) and pd.notnull(value):
        return int(value) if value == int(value) else float(value)
    return value


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize string cells, drop NULL strings, coerce numeric columns."""
    if df.empty:
        return df
    out = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    out = out.replace("NULL", pd.NA)
    out = out.dropna(how="all")
    return out.apply(_clean_column)


def _clean_column(col: pd.Series) -> pd.Series:
    # One text cell that is not a number (a name, or a code such as "02134")
    # makes the column text: converting only its number-like cells would merge
    # "02134" with "2134" and mix types within the column.
    if any(isinstance(v, str) and not _NUMBER_TEXT.fullmatch(v) for v in col):
        return col.map(lambda v: v if isinstance(v, str) else clean_and_convert(v))
    return col.map(clean_and_convert)


def _norm_col(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def map_column(name: str | None, columns: list[str]) -> str | None:
    """Map an LLM-suggested column name onto an actual DataFrame column."""
    if not name or not isinstance(name, str):
        return None
    if name in columns:
        return name
    normalized = {_norm_col(col): col for col in columns}
    return normalized.get(_norm_col(name))


def parse_sql_response_to_dataframe(sql_response_from_db: Any) -> pd.DataFrame | None:
    """Parse ``path_state["sql_response_from_db"]`` into a DataFrame.

    Execution stores a list with one JSON-records string, e.g. ``['[{...}]']``.
    Also accepts a raw list of row dicts or a DataFrame.
    """
    if sql_response_from_db is None:
        return None

    if isinstance(sql_response_from_db, pd.DataFrame):
        return clean_dataframe(sql_response_from_db)

    rows: Any = sql_response_from_db

    # Unwrap the execution-agent envelope: list[str] with one JSON blob.
    if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], str):
        try:
            rows = json.loads(rows[0])
        except json.JSONDecodeError:
            logger.info("parse_sql_response: could not JSON-decode sql_response string")
            return None

    if isinstance(rows, str):
        try:
            rows = json.loads(rows)
        except json.JSONDecodeError:
            logger.info("parse_sql_response: sql_response string is not JSON")
            return None

    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        return clean_dataframe(pd.DataFrame(rows))

    if isinstance(rows, list) and not rows:
        return pd.DataFrame()

    logger.info(
        "parse_sql_response: unsupported sql_response shape %s",
        type(sql_response_from_db),
    )
    return None
