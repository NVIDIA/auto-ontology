# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which columns admit a hand-edited sample value.

Profiled samples arrive already carrying the column's own type, but an edited
one is whatever text the user typed. Only a column whose declared SQL type can
hold that text as-is is editable: the text family (``text``, ``varchar``,
``character varying``, ``enum``, ``clob``…) plus the semi-structured types that
can hold a bare JSON string (``json``, ``jsonb``, ``variant``, ``xml``).

Every other declared type reads as read-only, so a numeric, boolean, date or
UUID column keeps exactly the samples profiling found for it — nothing can
overwrite them with text the warehouse would never return.
"""

from __future__ import annotations

import re

_TEXT_TYPES = frozenset(
    {
        "char",
        "character",
        "varchar",
        "varchar2",
        "nchar",
        "nvarchar",
        "nvarchar2",
        "varying",
        "text",
        "ntext",
        "tinytext",
        "mediumtext",
        "longtext",
        "string",
        "clob",
        "nclob",
        "citext",
        "enum",
        # Semi-structured types: a JSONB or VARIANT value may be a bare string,
        # unlike an OBJECT or ARRAY, which is why those two are absent.
        "json",
        "jsonb",
        "variant",
        "xml",
    }
)

# Type names tokenize on word boundaries rather than by substring search: every
# candidate token is matched whole, so `interval` does not read as `int` and a
# two-word name like `character varying` still resolves.
_TYPE_TOKEN = re.compile(r"[a-z][a-z0-9_]*")


def sample_values_editable(data_type: str | None) -> bool:
    """Whether a column of *data_type* accepts hand-written sample values.

    Mirrored on the client by ``sampleValuesEditable`` in
    ``frontend/lib/column-types.ts``, which is what leaves the sample list of
    every other column read-only in the catalog UI.

    An unrecognized or missing type answers ``False``. Profiling keeps working
    either way — this governs edits alone — so an unknown type name costs a
    column its edit affordance, never its samples.
    """
    if not data_type:
        return False
    lowered = data_type.strip().lower()
    # An array or composite column holds no single string a text box could
    # stand for, whatever its element type is.
    if lowered.endswith("[]") or lowered.startswith("_"):
        return False
    base, _, _params = lowered.partition("(")
    return bool(set(_TYPE_TOKEN.findall(base)) & _TEXT_TYPES)


def sample_values_edit_error(data_type: str | None) -> str | None:
    """Return why a *data_type* column rejects edited samples, or ``None``.

    The message is user-facing: it travels to the catalog UI as the detail of a
    422 raised by ``gsf.server.datasources.service.update_node_properties``.
    """
    if sample_values_editable(data_type):
        return None
    # Phrased around the type name rather than in front of it: "a integer" is
    # what an indefinite article yields for half the type names there are.
    declared = data_type.strip() if data_type else ""
    described = f"typed {declared}" if declared else "with no declared type"
    return (
        f"Sample values of a column {described} cannot be edited; only "
        "text-typed columns (text, varchar, json, …) accept them."
    )


__all__ = [
    "sample_values_edit_error",
    "sample_values_editable",
]
