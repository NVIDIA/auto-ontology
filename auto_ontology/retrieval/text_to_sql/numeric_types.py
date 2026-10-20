# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make the numeric *type* of a Postgres result predictable for exact graders.

Some graders (BIRD-Interact's checker is one) execute our SQL and the reference
SQL, round every number to a fixed number of places in Python, and then compare
the rows with ``==``. psycopg returns ``numeric`` as :class:`~decimal.Decimal`
and ``float4``/``float8`` as :class:`float`, and ``Decimal('0.01') != 0.01``, so
two answers that agree to the last digit still fail when one side is a float
and the other a numeric.

The grader cannot be changed from here and the reference is never visible, so
this module offers two deterministic, SQL-level levers:

* :func:`float_casts_to_numeric` rewrites our own explicit
  ``CAST(.. AS REAL / FLOAT / DOUBLE PRECISION)`` to ``NUMERIC``. Reference
  SQL that rounds must cast to ``numeric`` (Postgres has no ``round(float, n)``),
  so this is the right default.
* :func:`flip_numeric_output_types` moves the numeric output columns of a
  top-level ``SELECT`` to the *other* type, for the one retry a benchmark
  allows after a wrong answer.

The module is pure (``sqlglot`` only) so it can be exercised without a database.
"""

from __future__ import annotations

import hashlib
import json
import logging
import numbers
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Literal, Optional, Sequence

import sqlglot
from sqlglot import expressions as exp

logger = logging.getLogger(__name__)

NumericKind = Literal["float", "decimal"]
TargetType = Literal["numeric", "double"]

_POSTGRES_DIALECTS = frozenset({"postgres", "postgresql"})
_FLOAT_TYPES = frozenset({exp.DataType.Type.FLOAT, exp.DataType.Type.DOUBLE})


def _is_postgres(dialect: Optional[str]) -> bool:
    return (dialect or "").strip().lower() in _POSTGRES_DIALECTS


def _parse(sql: str) -> Optional[exp.Expression]:
    try:
        return sqlglot.parse_one(sql, read="postgres")
    except Exception as exc:  # noqa: BLE001 — never break the pipeline on a parse error
        logger.info("numeric_types: could not parse SQL (%s)", exc)
        return None


def float_casts_to_numeric(sql: str, dialect: Optional[str]) -> str:
    """Rewrite explicit float casts (``REAL``, ``FLOAT``, ``DOUBLE PRECISION``,
    ``::real``, ``::float8`` ...) to ``NUMERIC``; Postgres only.

    Returns *sql* byte-for-byte when nothing needs to change, so SQL that never
    casts to a float is not re-serialised. Only explicit casts are touched:
    columns that are natively ``float`` stay as they are, because a reference
    that reads the same column returns the same type.
    """
    if not _is_postgres(dialect) or not sql.strip():
        return sql
    tree = _parse(sql)
    if tree is None:
        return sql

    changed = False
    for cast in list(tree.find_all(exp.Cast, exp.TryCast)):
        target = cast.args.get("to")
        if isinstance(target, exp.DataType) and target.this in _FLOAT_TYPES:
            cast.set("to", exp.DataType.build("numeric"))
            changed = True
    return tree.sql(dialect="postgres") if changed else sql


def numeric_column_kinds(frame: Any) -> list[Optional[NumericKind]]:
    """Per output column: ``"float"``, ``"decimal"`` or ``None`` (anything else).

    Read off the first non-null value of each column of a result ``DataFrame``
    (duck-typed: ``shape``, ``iloc`` and ``dropna`` are all that is used).
    """
    if frame is None or getattr(frame, "shape", (0, 0))[1] == 0:
        return []
    kinds: list[Optional[NumericKind]] = []
    for position in range(frame.shape[1]):
        values = frame.iloc[:, position].dropna()
        if len(values) == 0:
            kinds.append(None)
            continue
        first = values.iloc[0]
        if isinstance(first, Decimal):
            kinds.append("decimal")
        elif isinstance(first, numbers.Real) and not isinstance(
            first, numbers.Integral
        ):
            kinds.append("float")
        else:
            kinds.append(None)
    return kinds


# The one number this module copies from the grader it serves (BIRD-Interact's
# checker rounds every number to 2 places, half-up, before comparing). Change it
# here if another grader needs the same treatment with a different precision.
GRADER_DECIMAL_PLACES = 2
_CENT = Decimal(1).scaleb(-GRADER_DECIMAL_PLACES)


def _normalise_cell(value: Any) -> str:
    """One result cell as text that does not depend on its numeric *type*.

    Numbers are rounded to ``GRADER_DECIMAL_PLACES`` places, half-up, through
    ``Decimal`` — the way the grader this module serves compares them — so ``Decimal('0.01')`` and
    ``0.013369`` come out the same and ``int 5`` equals ``Decimal('5.00')``.
    """
    if value is None or str(value) in ("NaT", "<NA>"):
        return "NULL"
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, numbers.Real):
        try:
            if isinstance(value, numbers.Integral):
                exact = Decimal(int(value))
            else:
                exact = Decimal(repr(float(value)))
            return str(exact.quantize(_CENT, rounding=ROUND_HALF_UP))
        except (InvalidOperation, ValueError):
            return str(value)  # NaN, infinity
    if isinstance(value, Decimal):
        return str(value.quantize(_CENT, rounding=ROUND_HALF_UP))
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def result_values_digest(frame: Any) -> Optional[str]:
    """A digest of a result's *values* in order, blind to numeric type.

    Two results with the same digest hold the same rows, in the same order,
    once rounded the way the grader rounds — whether the numbers are floats or
    numerics. Row order counts on purpose: a retry that reordered rows rewrote
    the query, so it is not a type-only change. ``None`` for an empty result.
    Used to tell a debug retry that only needs another numeric type from one
    that changed the answer itself.
    """
    if frame is None:
        return None
    shape = getattr(frame, "shape", (0, 0))
    if shape[0] == 0 or shape[1] == 0:
        return None
    rows = [
        repr(tuple(_normalise_cell(cell) for cell in record))
        for record in frame.itertuples(index=False, name=None)
    ]
    return hashlib.sha1("\n".join(rows).encode()).hexdigest()


def dominant_kind(
    kinds: Optional[Sequence[Optional[NumericKind]]],
) -> Optional[NumericKind]:
    """The kind a result mostly is: ``"decimal"`` wins over ``"float"``, ``None``
    when no column is fractional."""
    present = {kind for kind in kinds or () if kind}
    if "decimal" in present:
        return "decimal"
    if "float" in present:
        return "float"
    return None


def opposite_target(first_submit_kind: NumericKind) -> TargetType:
    """The type to retry with after *first_submit_kind* was graded wrong."""
    return "double" if first_submit_kind == "decimal" else "numeric"


def flip_numeric_output_types(
    sql: str,
    dialect: Optional[str],
    kinds: Sequence[Optional[NumericKind]],
    target: TargetType,
) -> Optional[str]:
    """Cast the numeric output columns of the top-level ``SELECT`` to *target*.

    Only columns whose kind differs from the target are wrapped (``"float"``
    columns for ``numeric``, ``"decimal"`` columns for ``double``); aliases are
    kept. Returns ``None`` — "leave the SQL alone" — when the shape is not safe
    to edit by position: not Postgres, not a plain ``SELECT`` (set operations),
    a ``*`` projection, a projection count that does not match *kinds*, or no
    column that needs converting.
    """
    if not _is_postgres(dialect) or not sql.strip():
        return None
    tree = _parse(sql)
    if not isinstance(tree, exp.Select):
        return None
    projections = tree.expressions
    if not projections or len(projections) != len(kinds):
        return None
    if any(projection.find(exp.Star) is not None for projection in projections):
        return None

    convert_from: NumericKind = "float" if target == "numeric" else "decimal"
    target_type = exp.DataType.build(
        "numeric" if target == "numeric" else "double precision"
    )

    converted = 0
    for projection, kind in zip(projections, kinds):
        if kind != convert_from:
            continue
        if isinstance(projection, exp.Alias):
            projection.set(
                "this",
                exp.Cast(this=projection.this.copy(), to=target_type.copy()),
            )
        else:
            projection.replace(exp.Cast(this=projection.copy(), to=target_type.copy()))
        converted += 1
    return tree.sql(dialect="postgres") if converted else None


__all__ = [
    "dominant_kind",
    "flip_numeric_output_types",
    "float_casts_to_numeric",
    "numeric_column_kinds",
    "opposite_target",
    "result_values_digest",
]
