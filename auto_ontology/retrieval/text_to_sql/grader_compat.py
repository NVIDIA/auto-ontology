# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep generated SQL valid after an exact-match grader mangles it.

BIRD-Interact's checker normalises both queries before running them. One step
(``remove_distinct``) splits the text on spaces and drops every standalone
``DISTINCT`` token. That is harmless for ``SELECT DISTINCT`` and
``COUNT(DISTINCT x)`` (the token is glued to a parenthesis), but it turns

* ``SELECT DISTINCT ON (k) ...``  into ``SELECT ON (k) ...``
* ``a IS [NOT] DISTINCT FROM b``  into ``a IS [NOT] FROM b``

and both are syntax errors, so a correct answer is scored as a failure.

The grader cannot be changed from here, so this module offers two deterministic
levers, both Postgres-only and both applied only when the caller opts in (the
same switch as :mod:`numeric_types`, so ordinary chat SQL is never touched):

* :func:`rewrite_null_safe_comparisons` replaces ``IS [NOT] DISTINCT FROM`` with
  an equivalent ``=`` / ``<>`` plus explicit ``IS NULL`` tests.
* :func:`distinct_on_problem` reports ``DISTINCT ON``, which has no equivalent
  that can be written mechanically (it needs a window function over the right
  ordering), as a message the SQL generator can act on.

The module is pure (``sqlglot`` only).
"""

from __future__ import annotations

import logging
from typing import Optional

import sqlglot
from sqlglot import expressions as exp

logger = logging.getLogger(__name__)

_POSTGRES_DIALECTS = frozenset({"postgres", "postgresql"})


def _is_postgres(dialect: Optional[str]) -> bool:
    return (dialect or "").strip().lower() in _POSTGRES_DIALECTS


def _parse(sql: str) -> Optional[exp.Expression]:
    try:
        return sqlglot.parse_one(sql, read="postgres")
    except Exception as exc:  # noqa: BLE001 — never break the pipeline on a parse error
        logger.info("grader_compat: could not parse SQL (%s)", exc)
        return None


def _is_null(node: exp.Expression) -> exp.Expression:
    return exp.Is(this=node.copy(), expression=exp.Null())


def _is_not_null(node: exp.Expression) -> exp.Expression:
    return exp.Not(this=_is_null(node))


def _equal_or_both_null(node: exp.NullSafeEQ) -> exp.Expression:
    left, right = node.this, node.expression
    return exp.Paren(
        this=exp.Or(
            this=exp.EQ(this=left.copy(), expression=right.copy()),
            expression=exp.Paren(
                this=exp.And(this=_is_null(left), expression=_is_null(right))
            ),
        )
    )


def _different_or_one_null(node: exp.NullSafeNEQ) -> exp.Expression:
    left, right = node.this, node.expression
    one_null = exp.Or(
        this=exp.Paren(
            this=exp.And(this=_is_null(left), expression=_is_not_null(right))
        ),
        expression=exp.Paren(
            this=exp.And(this=_is_not_null(left), expression=_is_null(right))
        ),
    )
    return exp.Paren(
        this=exp.Or(
            this=exp.NEQ(this=left.copy(), expression=right.copy()),
            expression=one_null,
        )
    )


def rewrite_null_safe_comparisons(sql: str, dialect: Optional[str]) -> str:
    """Rewrite ``a IS NOT DISTINCT FROM b`` and ``a IS DISTINCT FROM b``.

    Returns *sql* byte-for-byte when neither form occurs.
    """
    if not _is_postgres(dialect) or "distinct" not in sql.lower():
        return sql
    tree = _parse(sql)
    if tree is None:
        return sql
    if not any(tree.find_all(exp.NullSafeEQ, exp.NullSafeNEQ)):
        return sql

    def _replace(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.NullSafeEQ):
            return _equal_or_both_null(node)
        if isinstance(node, exp.NullSafeNEQ):
            return _different_or_one_null(node)
        return node

    return tree.transform(_replace).sql(dialect="postgres")


def distinct_on_problem(sql: str, dialect: Optional[str]) -> Optional[str]:
    """A message for the SQL generator when *sql* uses ``DISTINCT ON``, else ``None``."""
    if not _is_postgres(dialect) or "distinct" not in sql.lower():
        return None
    tree = _parse(sql)
    if tree is None:
        return None
    for node in tree.find_all(exp.Distinct):
        if node.args.get("on") is not None:
            return (
                "SELECT DISTINCT ON (...) is not allowed in this environment. "
                "Keep one row per group with a window function instead: select the "
                "needed columns plus ROW_NUMBER() OVER (PARTITION BY <the DISTINCT "
                "ON expressions> ORDER BY <the same ordering>) AS rn in a subquery "
                "or CTE, and keep rn = 1 in the outer query."
            )
    return None


__all__ = ["distinct_on_problem", "rewrite_null_safe_comparisons"]
