# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalise raw connector DataFrames into the shapes the writer expects."""

from datetime import timezone

import numpy as np
import pandas as pd

from gsf.catalog.constants import TableTypes


def flat_list_recursive(nested_list):
    output = []
    for i in nested_list:
        if isinstance(i, list):
            temp = flat_list_recursive(i)
            for j in temp:
                output.append(j)
        else:
            output.append(i)
    return output


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def normalize_tables(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and type a tables DataFrame.

    Accepts rows from either a SQL connector (``information_schema``) or the stored
    graph reload (``get_schema_tables``). Connector-specific columns such as
    ``owner`` are dropped; graph-only columns such as ``id`` and ``database``
    are preserved.
    """
    types = {
        "table_schema": "category",
        "table_name": "string",
        "table_type": "category",
        "created": "string",
        "description": "string",
    }
    base_columns = list(types.keys())
    df = (
        df.copy()
        if df is not None and not df.empty
        else pd.DataFrame(columns=base_columns)
    )
    if df.empty:
        return df

    for key in base_columns:
        if key not in df.columns:
            df[key] = pd.NA

    df["table_type"] = df["table_type"].fillna(TableTypes.BASE_TABLE)
    df = df.astype(dtype=types)

    if "created" in df:
        df["created"] = pd.to_datetime(df["created"], utc=True, format="mixed")
        df["created"] = df["created"].apply(
            lambda x: (
                x.tz_convert(timezone.utc).replace(microsecond=0) if pd.notna(x) else x
            )
        )

    for extra_col in ("owner",):
        if extra_col in df.columns:
            df = df.drop(columns=[extra_col])

    return df


_NULLABLE_TRUE = {"YES", "TRUE", "T", "Y", "1"}
_NULLABLE_FALSE = {"NO", "FALSE", "F", "N", "0"}


def coerce_nullable(value: object) -> bool | None:
    """Coerce one ``is_nullable`` value to a boolean.

    Connectors are required to return real booleans (see
    :meth:`gsf.connectors.base.SQLDatabase.get_columns`), and every connector
    in this repo does. This exists for the two cases that still arrive as
    text: a third-party connector that returns ``information_schema``'s
    ``'YES'``/``'NO'`` verbatim, and catalogs written before the column became
    a boolean.

    Both spellings are truthy as strings, which is what made ``bool(stored)``
    report the entire catalog as nullable. Unrecognised values raise rather
    than defaulting — silently guessing is what the bug was.
    """
    if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().upper()
    if text in _NULLABLE_TRUE:
        return True
    if text in _NULLABLE_FALSE:
        return False
    if text == "":
        return None
    raise ValueError(
        f"is_nullable must be a boolean (or 'YES'/'NO'); got {value!r}. "
        "Convert it in the connector — see SQLDatabase.get_columns."
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and type a columns DataFrame. Expects a DataFrame only."""
    types = {
        "table_schema": "category",
        "table_name": "category",
        "column_name": "string",
        "ordinal_position": "Int16",
        "data_type": "category",
        # "boolean", not "bool": the nullable dtype, because a connector that
        # cannot determine nullability leaves it NULL.
        "is_nullable": "boolean",
        "description": "string",
    }
    df = (
        df.copy()
        if df is not None and not df.empty
        else pd.DataFrame(columns=list(types.keys()))
    )
    if df.empty:
        return df

    for key in types.keys():
        if key not in df.columns:
            df[key] = pd.NA

    df["ordinal_position"] = pd.to_numeric(df["ordinal_position"])
    df["is_nullable"] = df["is_nullable"].map(coerce_nullable)
    df = df.astype(dtype=types)

    return df
