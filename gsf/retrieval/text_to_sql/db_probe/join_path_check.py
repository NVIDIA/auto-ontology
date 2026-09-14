# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Join-path verification for the repair path.

Sibling of ``literal_check.py``/``jsonb_path_check.py``: those catch a
hallucinated filter value or JSONB key; this catches a hallucinated **join
column** — an ``ON a.col = b.col`` predicate between two tables that aren't
actually related that way. Unlike a bad literal or JSONB key, a fabricated
join is syntactically and semantically valid SQL (both columns exist, types
match) and returns rows, so nothing upstream (parse validation, execution)
ever errors on it — it just silently returns the wrong data.

Ground truth here is the same ``SEMANTIC_FK`` graph already used to build
``attribute_join_paths`` for SQL generation (see
``gsf/dal/attributes.py::find_join_path``), queried directly against the
*actual* join predicates the model wrote, not just the columns it was told
about up front. Three verdicts:

- a **different**, real join path connects the same two tables (either a
  direct edge on different columns, or only via intermediate bridge
  table(s)) — repairable: return the correct hops so reconstruction can be
  told exactly what to use, mirroring the "KNOWN JOIN KEYS" mechanism.
- both columns share a **hub**: no forward path either way, but each holds
  its own real, PK-anchored ``SEMANTIC_FK`` edge to the same attribute (see
  ``find_shared_hub_bridge``) — also repairable, routed through the hub.
- **no** path or shared hub is known in the graph at all. This does not, on
  its own, mean the join is wrong — the graph can be incomplete for
  relationships ingestion never resolved. A live value-overlap probe (same
  idea as ``semantic_fk.py``'s ingestion-time sample-value fallback) is used
  as a secondary, confirmatory signal: near-zero overlap between the two
  join columns' actual values is treated as a fabricated join; substantial
  overlap is treated as "probably a real, just-unmodeled relationship" and
  left alone, to avoid flagging legitimate business joins the graph was
  never meant to cover.

Only equality predicates inside a ``JOIN ... ON`` clause are considered, and
only when at least one side is a column already known to the FK graph in
some capacity (see ``column_participates_in_semantic_fk``) — this keeps the
check scoped to the FK-shaped join-hallucination failure mode it targets,
not arbitrary business-logic joins (date ranges, status matches) the graph
was never meant to model.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import sqlglot
from sqlglot import exp

from gsf.dal.attributes import (
    column_participates_in_semantic_fk,
    find_join_path,
    find_shared_hub_bridge,
)
from gsf.dal.datasources import (
    find_column_id_by_table_and_name,
    find_table_key_columns,
)
from gsf.retrieval.text_to_sql.db_probe.executor import ProbeExecutor
from gsf.retrieval.text_to_sql.db_probe.literal_check import (
    _as_column,
    _candidate_tables,
    _sqlglot_dialect,
    _table_nodes,
)

logger = logging.getLogger(__name__)

# How much value-overlap between two join columns counts as "probably a real
# relationship, just not modeled in the graph" — below this, treat the join
# as fabricated. Deliberately generous: this only fires when the graph has
# nothing to say either way, so a false "leave it alone" is far cheaper than
# a false "flag a legitimate join".
_OVERLAP_KEEP_THRESHOLD = 0.2


def _find_join_path_either_direction(col_a_id: str, col_b_id: str) -> list[dict]:
    """``find_join_path``, tried in both column orders.

    ``find_join_path`` walks ``SEMANTIC_FK`` outgoing-only (child FK ->
    parent attribute — see its docstring), so a real, single-hop edge is
    only found when the *child* column is passed as the anchor. Every
    caller here derives ``col_a``/``col_b`` from a written predicate's
    left/right order, which has nothing to do with which side is the FK
    child — a real edge written with the parent on the left (e.g.
    ``parent.id = child.parent_id``) would otherwise come back with 0 hops
    purely because of how the model happened to order the equality, and
    fall through to "unverified" even though it's correct. Trying the
    reverse order costs one extra graph query only when the forward one
    finds nothing.
    """
    hops = find_join_path(col_a_id, col_b_id)
    if hops:
        return hops
    return find_join_path(col_b_id, col_a_id)


def _join_equalities(tree: exp.Expression) -> list[tuple[exp.Column, exp.Column]]:
    """Column-to-column equality predicates that express a join, wherever written.

    Two syntactic shapes carry the same join topology as an explicit
    ``JOIN ... ON``, and are collected here too:

    - a ``WHERE`` clause connecting two tables the old, implicit way
      (``FROM a, b WHERE a.col = b.col``);
    - a correlated subquery's ``WHERE`` comparing an outer-query column to an
      inner-query column (e.g. ``(SELECT ... FROM b WHERE b.col = a.col)`` in
      the outer query's projection or filter) — this is exactly as capable of
      encoding a hallucinated relationship as a real ``JOIN``, but was
      previously invisible to this check entirely.

    ``tree.find_all(exp.Where)`` walks every nesting level, so a correlated
    subquery's own ``WHERE`` is picked up alongside the top-level one without
    extra recursion here. Predicates already inside a ``JOIN ... ON`` are not
    re-found via ``Where`` (``ON`` is a distinct clause), so nothing is
    double-counted between the two loops below.

    Only predicates where *both* sides are bare columns are considered — a
    join condition compares two identity columns, not a column against a
    literal or an expression (those are filters, not join topology). Same-
    table and unresolvable-alias predicates are filtered out by the caller
    (see ``find_join_path_mismatches``), so a same-table ``WHERE`` filter
    (e.g. ``a.status = a.other_col``) is never mistaken for a join here.
    """
    out: list[tuple[exp.Column, exp.Column]] = []
    for join in tree.find_all(exp.Join):
        on = join.args.get("on")
        if on is None:
            continue
        for eq in on.find_all(exp.EQ):
            left = _as_column(eq.this)
            right = _as_column(eq.expression)
            if left is not None and right is not None:
                out.append((left, right))
    for where in tree.find_all(exp.Where):
        for eq in where.find_all(exp.EQ):
            left = _as_column(eq.this)
            right = _as_column(eq.expression)
            if left is not None and right is not None:
                out.append((left, right))
    return out


def _resolve_table_name(
    col: exp.Column, all_nodes: list[exp.Table], by_key: dict[str, exp.Table]
) -> Optional[str]:
    """The real (unaliased) table name a column belongs to, per the SQL's own aliasing."""
    candidates = _candidate_tables(col, all_nodes, by_key)
    if len(candidates) == 1:
        return candidates[0].name
    return None


def _value_overlap(
    executor: ProbeExecutor,
    dialect: Optional[str],
    table_a: str,
    col_a: str,
    table_b: str,
    col_b: str,
) -> Optional[float]:
    """Fraction of *table_a*'s distinct ``col_a`` values also present in *table_b*'s ``col_b``.

    ``None`` when the probe couldn't run (no connector, budget exhausted,
    query failure) — callers must treat that as "can't confirm", not as a
    verdict either way.

    Identifiers are always quoted: *table_a*/*col_a*/etc. are already the
    real, correctly-cased names resolved from schema metadata by the
    caller, and an unquoted reference here would let the engine's own
    case-folding rules silently rewrite it (e.g. Postgres lowercases an
    unquoted ``ArtifactsCore`` to ``artifactscore``, which then doesn't
    exist) — producing a query-execution failure that this function can
    only see as "probe couldn't run," so the mismatch falls through to
    "unverified" even when the join is actually correct. Quoting the
    already-correct case is a no-op for lowercase-only identifiers in every
    dialect this pipeline targets, so this can't regress a case that works
    today — it only fixes the mixed-case ones that silently didn't.
    """
    d = _sqlglot_dialect(dialect)
    a_ref = exp.column(col_a, quoted=True).sql(dialect=d)
    b_ref = exp.column(col_b, quoted=True).sql(dialect=d)
    a_table_ref = exp.table_(table_a, quoted=True).sql(dialect=d)
    b_table_ref = exp.table_(table_b, quoted=True).sql(dialect=d)
    sql = (
        f"SELECT "
        f"COUNT(DISTINCT a.{a_ref}) AS total, "
        f"COUNT(DISTINCT CASE WHEN b.{b_ref} IS NOT NULL THEN a.{a_ref} END) AS matched "
        f"FROM {a_table_ref} a "
        f"LEFT JOIN {b_table_ref} b ON a.{a_ref} = b.{b_ref} "
        f"WHERE a.{a_ref} IS NOT NULL"
    )
    res = executor.run(sql, purpose="join_path_check_overlap")
    if not res["ok"] or not res["rows"]:
        return None
    row = res["rows"][0]
    total = row.get("total") or 0
    matched = row.get("matched") or 0
    if not total:
        return None
    return matched / total


def find_join_path_mismatches(
    executor: ProbeExecutor,
    dialect: Optional[str],
    sql: str,
    database_name: Optional[str],
) -> list[dict[str, Any]]:
    """Return join predicates that don't correspond to a known FK relationship.

    Each entry: ``{"table_a", "col_a", "table_b", "col_b", "verdict", "hops",
    "bridge_tables"}``. ``verdict`` is one of:

    - ``"wrong_column"``: same two tables, a different real edge exists — use
      ``hops[0]`` instead.
    - ``"missing_bridge"``: these two tables should not be joined directly at
      all — route through ``hops`` (which cross ``bridge_tables``).
    - ``"unverified"``: no known path either way, and (when a live probe was
      possible) value overlap was low — likely fabricated, but there is no
      known repair; ``hops``/``bridge_tables`` are empty.

    Empty list means every checked predicate matched a known edge, or (when
    the graph had nothing to say and no probe could run) nothing could be
    confirmed as wrong — never treated as "the whole query is fine" beyond
    what was actually checked.
    """
    try:
        tree = sqlglot.parse_one(sql, read=_sqlglot_dialect(dialect))
    except Exception as exc:  # noqa: BLE001 — never break the pipeline on a parse error
        logger.info("join_path_check: could not parse SQL (%s)", exc)
        return []
    if tree is None:
        return []

    all_nodes, by_key = _table_nodes(tree)
    if not all_nodes:
        return []

    mismatches: list[dict[str, Any]] = []
    checked_pairs: set[tuple[str, str, str, str]] = set()

    for left, right in _join_equalities(tree):
        table_a = _resolve_table_name(left, all_nodes, by_key)
        table_b = _resolve_table_name(right, all_nodes, by_key)
        if not table_a or not table_b or table_a.lower() == table_b.lower():
            continue  # self-join or unresolvable alias — not this check's job

        pair_key = tuple(
            sorted(
                [
                    (table_a.lower(), left.name.lower()),
                    (table_b.lower(), right.name.lower()),
                ]
            )
        )
        flat_key = (pair_key[0][0], pair_key[0][1], pair_key[1][0], pair_key[1][1])
        if flat_key in checked_pairs:
            continue
        checked_pairs.add(flat_key)

        col_a_id = find_column_id_by_table_and_name(table_a, left.name, database_name)
        col_b_id = find_column_id_by_table_and_name(table_b, right.name, database_name)
        if not col_a_id or not col_b_id:
            continue  # can't resolve — never flag on missing information

        if not (
            column_participates_in_semantic_fk(col_a_id)
            or column_participates_in_semantic_fk(col_b_id)
        ):
            continue  # neither side is FK-shaped — out of scope for this check

        hops = _find_join_path_either_direction(col_a_id, col_b_id)

        if len(hops) == 1:
            hop = hops[0]
            hop_pair = {
                (hop["source_table"].lower(), hop["source_column"].lower()),
                (hop["target_table"].lower(), hop["target_column"].lower()),
            }
            written_pair = {
                (table_a.lower(), left.name.lower()),
                (table_b.lower(), right.name.lower()),
            }
            if hop_pair == written_pair:
                continue  # verified — exactly what the graph says

            # The graph knows *a* direct relationship between these two
            # tables — but two tables can legitimately have more than one
            # (e.g. a treatments table with both prescribing_clinician_id
            # and reviewing_clinician_id, each a real FK to clinicians). The
            # graph having learned one doesn't mean the model's different,
            # also-real column pair is wrong — same reasoning as the
            # missing_bridge case above, so the same live check applies
            # before concluding it's fabricated.
            overlap = None
            if executor.budget_left:
                overlap = _value_overlap(
                    executor, dialect, table_a, left.name, table_b, right.name
                )
                if overlap is None:
                    overlap = _value_overlap(
                        executor, dialect, table_b, right.name, table_a, left.name
                    )
            if overlap is not None and overlap >= _OVERLAP_KEEP_THRESHOLD:
                logger.info(
                    "join_path_check: graph knows a different column pair for "
                    "%s <-> %s, but live overlap %.2f >= threshold on the "
                    "written %s.%s = %s.%s — treating it as a real, "
                    "additional relationship and leaving it alone",
                    table_a,
                    table_b,
                    overlap,
                    table_a,
                    left.name,
                    table_b,
                    right.name,
                )
                continue

            mismatches.append(
                {
                    "table_a": table_a,
                    "col_a": left.name,
                    "table_b": table_b,
                    "col_b": right.name,
                    "verdict": "wrong_column",
                    "hops": [hop],
                    "bridge_tables": [],
                }
            )
            continue

        if len(hops) > 1:
            # The graph knows *an* indirect path — but unlike the "no path
            # at all" branch below, this used to trust that as proof the
            # model's direct join is wrong, with no live check. That's only
            # true when the direct columns aren't *also* a real relationship
            # the graph simply never modeled (e.g. an undeclared FK-shaped
            # column with no formal constraint) — the graph having *an*
            # answer doesn't mean it has the *only* answer. Apply the same
            # value-overlap safety net used below before concluding the
            # direct join is fabricated, so a correct-but-unmodeled direct
            # join isn't torn out in favor of a technically-known but wrong
            # indirect route.
            overlap = None
            if executor.budget_left:
                overlap = _value_overlap(
                    executor, dialect, table_a, left.name, table_b, right.name
                )
                if overlap is None:
                    overlap = _value_overlap(
                        executor, dialect, table_b, right.name, table_a, left.name
                    )
            if overlap is not None and overlap >= _OVERLAP_KEEP_THRESHOLD:
                logger.info(
                    "join_path_check: graph only knows an indirect path for "
                    "%s.%s = %s.%s, but live overlap %.2f >= threshold — "
                    "treating the direct join as real and leaving it alone",
                    table_a,
                    left.name,
                    table_b,
                    right.name,
                    overlap,
                )
                continue

            bridge_tables = sorted(
                {h[side] for h in hops for side in ("source_table", "target_table")}
                - {table_a, table_b}
            )
            mismatches.append(
                {
                    "table_a": table_a,
                    "col_a": left.name,
                    "table_b": table_b,
                    "col_b": right.name,
                    "verdict": "missing_bridge",
                    "hops": hops,
                    "bridge_tables": bridge_tables,
                }
            )
            continue

        # No path at all via the forward-only traversal. Before falling
        # back to a live probe, check for a shared-identity hub: two
        # columns can each hold a real, PK-anchored forward SEMANTIC_FK to
        # the same attribute without either being reachable from the
        # other (find_join_path's guard against fabricating a join between
        # two coincidentally-shared-target columns also hides this
        # legitimate case). Trusted at the same confidence as the
        # multi-hop `missing_bridge` branch above — both sides already
        # have a real, ingested FK edge, so no extra probe is needed here.
        hub = find_shared_hub_bridge(col_a_id, col_b_id)
        if hub.get("hub_table") and hub.get("hub_column"):
            # A direct join between two hub-shared FK columns is only worth
            # flagging when NEITHER side is itself unique/PK-backed on its
            # own table. When one side is (e.g. a satellite table's PK is
            # also its FK to the hub), that column is guaranteed 1:1 with
            # the hub row, so `A.col = B.col` directly is mathematically
            # identical to routing `A -> hub <- B` through the bridge — just
            # without the extra join.
            keys_a = find_table_key_columns(table_a, database_name)
            keys_b = find_table_key_columns(table_b, database_name)
            col_a_is_identity = left.name.lower() in (keys_a["pk"] + keys_a["unique"])
            col_b_is_identity = right.name.lower() in (keys_b["pk"] + keys_b["unique"])
            if col_a_is_identity or col_b_is_identity:
                continue  # direct join is provably equivalent — not a mismatch
            mismatches.append(
                {
                    "table_a": table_a,
                    "col_a": left.name,
                    "table_b": table_b,
                    "col_b": right.name,
                    "verdict": "missing_bridge",
                    "hops": [
                        {
                            "source_schema": "",
                            "source_table": table_a,
                            "source_column": left.name,
                            "target_schema": "",
                            "target_table": hub["hub_table"],
                            "target_column": hub["hub_column"],
                        },
                        {
                            "source_schema": "",
                            "source_table": table_b,
                            "source_column": right.name,
                            "target_schema": "",
                            "target_table": hub["hub_table"],
                            "target_column": hub["hub_column"],
                        },
                    ],
                    "bridge_tables": [hub["hub_table"]],
                }
            )
            continue

        # No shared hub either — the graph has nothing to confirm or deny
        # this join. Fall back to a live value-overlap probe as a
        # secondary signal before flagging, to avoid false-positiving on a
        # real relationship the graph never happened to model at all.
        overlap = None
        if executor.budget_left:
            overlap = _value_overlap(
                executor, dialect, table_a, left.name, table_b, right.name
            )
            if overlap is None:
                overlap = _value_overlap(
                    executor, dialect, table_b, right.name, table_a, left.name
                )
        if overlap is not None and overlap >= _OVERLAP_KEEP_THRESHOLD:
            logger.info(
                "join_path_check: no graph edge for %s.%s = %s.%s, but live "
                "overlap %.2f >= threshold — treating as a real, unmodeled "
                "relationship and leaving it alone",
                table_a,
                left.name,
                table_b,
                right.name,
                overlap,
            )
            continue

        mismatches.append(
            {
                "table_a": table_a,
                "col_a": left.name,
                "table_b": table_b,
                "col_b": right.name,
                "verdict": "unverified",
                "hops": [],
                "bridge_tables": [],
                "overlap": overlap,
            }
        )

    return mismatches


def _set_column_name(col: exp.Column, new_name: str) -> None:
    """Rename *col* in place to *new_name*, leaving it untouched if the name
    is already correct and preserving whatever quoting the original
    identifier had (many of these DBs use case-sensitive mixed-case column
    names like ``"REC_COMP"`` — re-serializing through a fresh, unquoted
    ``exp.to_identifier`` would silently fold that to lowercase and break
    the query against Postgres's case-sensitive quoted-identifier rules)."""
    if col.name == new_name:
        return
    quoted = bool(col.this.args.get("quoted")) if col.this else False
    col.set("this", exp.to_identifier(new_name, quoted=quoted))


def try_self_apply_wrong_column_fixes(
    mismatches: list[dict[str, Any]], sql: str, dialect: Optional[str]
) -> tuple[str, list[dict[str, Any]]]:
    """Deterministically fix ``wrong_column`` mismatches by patching the
    column name(s) on the existing join predicate in place.

    Scoped to ``wrong_column`` only, not ``missing_bridge`` or
    ``unverified``: a wrong-column fix keeps the same two tables (and the
    same alias references already in the SQL) and only swaps which column
    on each side is compared, which is a safe, local AST edit. A
    ``missing_bridge`` fix has to insert an entirely new JOIN clause (a new
    table, a new alias, rewritten ON conditions) — real structural surgery,
    not a local swap, so it's left to reconstruction. ``unverified`` has no
    known correct column to swap in at all.

    Returns the (possibly rewritten) SQL and the sub-list of mismatches
    left for reconstruction (untouched ``wrong_column`` mismatches whose
    predicate couldn't be relocated, plus every ``missing_bridge``/
    ``unverified`` mismatch, unchanged).
    """
    fixable = {
        id(m): m for m in mismatches if m["verdict"] == "wrong_column" and m.get("hops")
    }
    if not fixable:
        return sql, mismatches

    try:
        tree = sqlglot.parse_one(sql, read=_sqlglot_dialect(dialect))
    except Exception as exc:  # noqa: BLE001 — never break the pipeline on a parse error
        logger.info("join_path_check self-apply: could not parse SQL (%s)", exc)
        return sql, mismatches
    if tree is None:
        return sql, mismatches

    all_nodes, by_key = _table_nodes(tree)
    if not all_nodes:
        return sql, mismatches

    applied: set[int] = set()
    for left, right in _join_equalities(tree):
        table_left = _resolve_table_name(left, all_nodes, by_key)
        table_right = _resolve_table_name(right, all_nodes, by_key)
        if not table_left or not table_right:
            continue
        for mid, m in fixable.items():
            if mid in applied:
                continue
            # Match this predicate to the mismatch by the exact written
            # (table, column) pair on each side — not just the table set —
            # so a query with two different wrong joins between the same
            # two tables doesn't accidentally patch the wrong one.
            written = {
                (table_left.lower(), left.name.lower()),
                (table_right.lower(), right.name.lower()),
            }
            wanted = {
                (m["table_a"].lower(), m["col_a"].lower()),
                (m["table_b"].lower(), m["col_b"].lower()),
            }
            if written != wanted:
                continue
            hop = m["hops"][0]
            if table_left.lower() == hop["source_table"].lower():
                new_left_col, new_right_col = hop["source_column"], hop["target_column"]
            else:
                new_left_col, new_right_col = hop["target_column"], hop["source_column"]
            _set_column_name(left, new_left_col)
            _set_column_name(right, new_right_col)
            applied.add(mid)
            break

    if not applied:
        return sql, mismatches

    new_sql = tree.sql(dialect=_sqlglot_dialect(dialect))
    remaining = [m for m in mismatches if id(m) not in applied]
    return new_sql, remaining


def _hub_bridge_hops(m: dict[str, Any]) -> Optional[tuple[dict, dict]]:
    """``(hop_a, hop_b)`` when *m* is the narrow, unambiguous shared-hub shape
    this self-apply targets: exactly one bridge table, reached from each of
    ``table_a``/``table_b`` by a single hop. ``None`` for anything else
    (multi-hop chains, or a hop shape this function doesn't recognize) —
    those are left to reconstruction, same as today.
    """
    hops = m.get("hops") or []
    bridge_tables = m.get("bridge_tables") or []
    if len(hops) != 2 or len(bridge_tables) != 1:
        return None
    hub = bridge_tables[0].lower()
    hop_a = next(
        (h for h in hops if h["source_table"].lower() == m["table_a"].lower()), None
    )
    hop_b = next(
        (h for h in hops if h["source_table"].lower() == m["table_b"].lower()), None
    )
    if hop_a is None or hop_b is None:
        return None
    if hop_a["target_table"].lower() != hub or hop_b["target_table"].lower() != hub:
        return None
    return hop_a, hop_b


def _chain_bridge_hops(m: dict[str, Any]) -> Optional[tuple[dict, dict]]:
    """``(hop_a, hop_b)`` for the other narrow, unambiguous shape: a genuine
    2-hop *path* through one intermediate table (``table_a -> bridge ->
    table_b``), as opposed to ``_hub_bridge_hops``'s shared-parent shape
    (``table_a -> hub <- table_b``, two independent edges to a common
    table). ``find_join_path`` returns this as an ordered chain — hop[0]
    lands on the bridge, hop[1] leaves it — rather than
    ``find_shared_hub_bridge``'s two same-target hops, so it needs its own
    recognizer, but once recognized the fix is the identical mechanical
    edit: insert one JOIN for the bridge, rewrite the other to reference it.
    Reorients whichever hop runs "backwards" (bridge -> table_b instead of
    table_b -> bridge) so both come out in the same ``(source=table,
    target=bridge)`` shape ``_hub_bridge_hops`` produces, letting the rest
    of the self-apply logic treat both shapes identically.
    """
    hops = m.get("hops") or []
    bridge_tables = m.get("bridge_tables") or []
    if len(hops) != 2 or len(bridge_tables) != 1:
        return None
    bridge = bridge_tables[0].lower()
    h0, h1 = hops
    if h0["target_table"].lower() != bridge or h1["source_table"].lower() != bridge:
        return None
    endpoints = {h0["source_table"].lower(), h1["target_table"].lower()}
    if endpoints != {m["table_a"].lower(), m["table_b"].lower()}:
        return None

    def _reversed(h: dict) -> dict:
        return {
            "source_schema": h.get("target_schema", ""),
            "source_table": h["target_table"],
            "source_column": h["target_column"],
            "target_schema": h.get("source_schema", ""),
            "target_table": h["source_table"],
            "target_column": h["source_column"],
        }

    if h0["source_table"].lower() == m["table_a"].lower():
        return h0, _reversed(h1)
    return _reversed(h1), h0


def _bridge_hops(m: dict[str, Any]) -> Optional[tuple[dict, dict]]:
    """Either recognized self-applicable bridge shape for *m* — shared-hub
    first, then linear-chain — or ``None`` if neither matches."""
    return _hub_bridge_hops(m) or _chain_bridge_hops(m)


def try_self_apply_missing_bridge_fixes(
    mismatches: list[dict[str, Any]], sql: str, dialect: Optional[str]
) -> tuple[str, list[dict[str, Any]]]:
    """Deterministically fix the narrow, unambiguous subclass of
    ``missing_bridge`` mismatches: exactly one intermediate table between
    ``table_a``/``table_b``, in either of two recognized shapes (see
    :func:`_bridge_hops`) — a shared hub (``table_a -> hub <- table_b``, two
    independent edges to a common table) or a genuine 2-hop chain
    (``table_a -> bridge -> table_b``, a real path through one intermediate)
    — where that intermediate table isn't already referenced anywhere else
    in the query.

    Scoped this narrowly on purpose. Given the two hops, the new JOIN's ON
    condition is fully determined and there's exactly one safe place to
    attach it — immediately before the join being fixed, since whichever of
    the two tables isn't introduced by that join clause is already in scope
    by then. That makes it a local, mechanical AST edit, the same way
    :func:`try_self_apply_wrong_column_fixes` is for ``wrong_column``.

    Left to reconstruction (unchanged mismatch, same as if this were never
    called): longer chains (more than one intermediate table, or a hop shape
    :func:`_bridge_hops` doesn't recognize — real structural judgment about
    where/how many joins to insert); WHERE-implicit joins (no ``exp.Join``
    node to attach to); and any case where the intermediate table is
    already referenced elsewhere in the query (attach-vs-reuse is an
    alias-management judgment call this local edit can't safely make).

    Note this only makes the join *graph-legal* — bridging two child tables
    through a shared parent is schema-correct but can still fan out into an
    unintended many-to-many pairing if both children have multiple rows per
    parent. Callers should still re-check the result (e.g. the cheap static
    SQL checks in ``sql_parse_validation.py``) rather than treat a self-
    applied fix as automatically correct.

    Returns the (possibly rewritten) SQL and the sub-list of mismatches left
    for reconstruction (untouched ``missing_bridge``/``unverified``
    mismatches, unchanged).
    """
    fixable: dict[int, dict[str, Any]] = {}
    for m in mismatches:
        if m["verdict"] != "missing_bridge":
            continue
        if _bridge_hops(m) is not None:
            fixable[id(m)] = m
    if not fixable:
        return sql, mismatches

    try:
        tree = sqlglot.parse_one(sql, read=_sqlglot_dialect(dialect))
    except Exception as exc:  # noqa: BLE001 — never break the pipeline on a parse error
        logger.info("join_path_check self-apply(bridge): could not parse SQL (%s)", exc)
        return sql, mismatches
    if tree is None:
        return sql, mismatches

    all_nodes, by_key = _table_nodes(tree)
    if not all_nodes:
        return sql, mismatches

    existing_names = {t.name.lower() for t in all_nodes if t.name}
    existing_aliases = set(by_key.keys())
    d = _sqlglot_dialect(dialect)

    applied: set[int] = set()
    for join in tree.find_all(exp.Join):
        on = join.args.get("on")
        if on is None:
            continue
        select = join.find_ancestor(exp.Select)
        if select is None:
            continue
        joins = select.args.get("joins") or []
        if join not in joins:
            continue  # already rewritten via an outer EQ in this same ON clause

        for eq in on.find_all(exp.EQ):
            if not fixable:
                break
            left = _as_column(eq.this)
            right = _as_column(eq.expression)
            if left is None or right is None:
                continue
            table_left = _resolve_table_name(left, all_nodes, by_key)
            table_right = _resolve_table_name(right, all_nodes, by_key)
            if not table_left or not table_right:
                continue

            for mid, m in list(fixable.items()):
                if mid in applied:
                    continue
                written = {
                    (table_left.lower(), left.name.lower()),
                    (table_right.lower(), right.name.lower()),
                }
                wanted = {
                    (m["table_a"].lower(), m["col_a"].lower()),
                    (m["table_b"].lower(), m["col_b"].lower()),
                }
                if written != wanted:
                    continue

                hub_table = m["bridge_tables"][0]
                if hub_table.lower() in existing_names:
                    # Hub already referenced elsewhere in the query —
                    # attach-vs-reuse ambiguity, leave to reconstruction.
                    continue
                hop_a, hop_b = _bridge_hops(m)
                hop_for = {
                    m["table_a"].lower(): hop_a,
                    m["table_b"].lower(): hop_b,
                }

                # This join clause's own table (``join.this``) is the side
                # that only comes into scope *at* this join — the other side
                # is necessarily already in scope (either the FROM table or
                # an earlier join), regardless of which one is "table_a"/
                # "table_b" in the mismatch or which order they were
                # written in the predicate. The new hub join must attach to
                # the already-in-scope side; rewriting which table a join
                # clause introduces isn't a local edit.
                introduced_table = (
                    join.this.name
                    if isinstance(join.this, exp.Table) and join.this.name
                    else None
                )
                if introduced_table is None:
                    continue  # not a plain table join (e.g. a subquery) — skip
                introduced_table = introduced_table.lower()
                if introduced_table not in (table_left.lower(), table_right.lower()):
                    continue  # unexpected shape — leave to reconstruction

                if table_left.lower() == introduced_table:
                    introduced_col, prior_col = left, right
                    prior_table = table_right
                else:
                    introduced_col, prior_col = right, left
                    prior_table = table_left
                hop_introduced = hop_for.get(introduced_table)
                hop_prior = hop_for.get(prior_table.lower())
                if hop_introduced is None or hop_prior is None:
                    continue

                base_alias = "".join(ch for ch in hub_table if ch.isalnum())[:3].lower()
                alias = base_alias or "hub"
                n = 1
                while alias in existing_aliases:
                    n += 1
                    alias = f"{base_alias or 'hub'}{n}"

                # New join: <prior_table> JOIN <hub> ON <prior>.<hop_prior.source> = <hub>.<hop_prior.target>
                # — always safe to insert right before the join being fixed,
                # since prior_table is guaranteed already in scope by then.
                # Rename prior_col to hop_prior's real FK column *before*
                # capturing its SQL text — the written predicate may have
                # had the wrong column on this side too (that's exactly
                # what made it a mismatch), so the new join must use the
                # graph-known column, not whatever was originally written.
                _set_column_name(prior_col, hop_prior["source_column"])
                prior_col_sql = prior_col.sql(dialect=d)
                # Both quoted for the same reason _value_overlap's probe SQL
                # quotes its identifiers: hub_table is a table this self-apply
                # is introducing for the first time, not one already present
                # in relevant_tables — so the "Re-run
                # quote_known_mixed_case_identifiers" pass after self-apply
                # (join_path_check.py agent, right below this call site)
                # has no metadata for it and can't fix its casing after the
                # fact. Left unquoted, an unquoted mixed-case name like
                # "Communication" gets silently case-folded by Postgres to
                # "communication", which then doesn't exist — this was the
                # exact cause of two observed "relation ... does not exist"
                # submission failures (a self-applied bridge join to a real,
                # correctly-identified table that nonetheless failed at
                # execution because it was never quoted).
                hub_prior_col_sql = exp.column(
                    hop_prior["target_column"], table=alias, quoted=True
                ).sql(dialect=d)
                hub_table_sql = exp.table_(hub_table, alias=alias, quoted=True).sql(
                    dialect=d
                )
                try:
                    new_join = sqlglot.parse_one(
                        f"SELECT 1 FROM x JOIN {hub_table_sql} "
                        f"ON {prior_col_sql} = {hub_prior_col_sql}",
                        read=d,
                    ).args["joins"][0]
                except Exception as exc:  # noqa: BLE001
                    logger.info(
                        "join_path_check self-apply(bridge): could not build "
                        "hub join for %s (%s)",
                        hub_table,
                        exc,
                    )
                    continue

                idx = joins.index(join)
                joins.insert(idx, new_join)
                select.set("joins", joins)

                # Rewrite this join's own ON: <hub>.<hop_introduced.target> = <introduced_table>.<hop_introduced.source>
                # — introduced_table's alias is exactly the one this join
                # clause defines, so it's already correct; only its column
                # name and the other side's table/column change.
                _set_column_name(introduced_col, hop_introduced["source_column"])
                prior_col.set("table", exp.to_identifier(alias))
                _set_column_name(prior_col, hop_introduced["target_column"])

                existing_aliases.add(alias)
                existing_names.add(hub_table.lower())
                applied.add(mid)
                del fixable[mid]
                break

    if not applied:
        return sql, mismatches

    new_sql = tree.sql(dialect=d)
    remaining = [m for m in mismatches if id(m) not in applied]
    return new_sql, remaining


def build_join_path_repair_error(mismatches: list[dict[str, Any]]) -> str:
    """Render mismatches into a targeted reconstruction instruction."""
    lines = []
    for m in mismatches:
        written = f"{m['table_a']}.{m['col_a']} = {m['table_b']}.{m['col_b']}"
        if m["verdict"] == "wrong_column":
            hop = m["hops"][0]
            correct = (
                f"{hop['source_table']}.{hop['source_column']} = "
                f"{hop['target_table']}.{hop['target_column']}"
            )
            lines.append(
                f"- You joined {written}, but the real relationship between "
                f"these tables uses {correct}. Use the correct column pair."
            )
        elif m["verdict"] == "missing_bridge":
            chain = " -> ".join(
                f"{h['source_table']}.{h['source_column']} = "
                f"{h['target_table']}.{h['target_column']}"
                for h in m["hops"]
            )
            lines.append(
                f"- You joined {written} directly, but these two tables are "
                f"not directly related. Route the join through "
                f"{', '.join(m['bridge_tables'])} instead, using: {chain}."
            )
        else:
            lines.append(
                f"- You joined {written}, but no relationship between these "
                f"columns could be verified and their actual values barely "
                f"overlap. This join is very likely wrong. Before looking for "
                f"a way to connect these two tables, first check whether the "
                f"value you need is already available on a table you're "
                f"already joined to — you may not need this join at all. If "
                f"it genuinely isn't available anywhere in scope, the value "
                f"likely comes from a different table entirely, not just a "
                f"different column pair between these same two tables."
            )
    body = "\n".join(lines)
    # "unverified" mismatches have no known correct join — the real fix may
    # be dropping the join and sourcing the value from elsewhere entirely
    # (see the per-item message above), which the blanket "change only the
    # join" instruction below would otherwise forbid.
    has_unverified = any(m["verdict"] == "unverified" for m in mismatches)
    scope_instruction = (
        "Rewrite the SQL using the correct join condition(s) above. Change "
        "ONLY the mismatched join(s); keep all other joins, columns, "
        "grouping, and filters exactly as they are."
        if not has_unverified
        else "Rewrite the SQL to fix the join(s) flagged above. For any "
        "flagged as unverified, you may need to change which table/column "
        "supplies that value entirely (including dropping the join and "
        "reading it from a table already in scope), not just swap the join "
        "predicate. Leave every other join, column, grouping, and filter "
        "exactly as they are."
    )
    return (
        "One or more JOIN conditions in the generated SQL do not match a "
        "real relationship between the tables, so the query silently joins "
        "the wrong rows instead of erroring:\n"
        f"{body}\n\n"
        f"{scope_instruction}"
    )


def _group_join_equalities_by_table_pair(
    tree: exp.Expression,
    all_nodes: list[exp.Table],
    by_key: dict[str, exp.Table],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Group column-equality join predicates by the (table_a, table_b) they connect.

    Each group entry is ``{"left", "right", "table_left", "table_right"}`` —
    the raw predicate kept oriented exactly as written, plus its resolved
    (real, unaliased) table names. Both orientations of the same two tables
    (``a.col = b.col`` and ``b.col = a.col``) land in one group, keyed by the
    table names sorted case-insensitively.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for left, right in _join_equalities(tree):
        table_left = _resolve_table_name(left, all_nodes, by_key)
        table_right = _resolve_table_name(right, all_nodes, by_key)
        if (
            not table_left
            or not table_right
            or table_left.lower() == table_right.lower()
        ):
            continue
        key = tuple(sorted([table_left.lower(), table_right.lower()]))
        groups.setdefault(key, []).append(
            {
                "left": left,
                "right": right,
                "table_left": table_left,
                "table_right": table_right,
            }
        )
    return groups


def _is_verified_edge(predicate: dict[str, Any], database_name: Optional[str]) -> bool:
    """Whether *predicate* is exactly the graph-known relationship between its two tables."""
    left, right = predicate["left"], predicate["right"]
    table_left, table_right = predicate["table_left"], predicate["table_right"]
    col_a_id = find_column_id_by_table_and_name(table_left, left.name, database_name)
    col_b_id = find_column_id_by_table_and_name(table_right, right.name, database_name)
    if not col_a_id or not col_b_id:
        return False
    hops = _find_join_path_either_direction(col_a_id, col_b_id)
    if len(hops) != 1:
        return False
    hop = hops[0]
    hop_pair = {
        (hop["source_table"].lower(), hop["source_column"].lower()),
        (hop["target_table"].lower(), hop["target_column"].lower()),
    }
    written_pair = {
        (table_left.lower(), left.name.lower()),
        (table_right.lower(), right.name.lower()),
    }
    return hop_pair == written_pair


def _oriented_columns(predicate: dict[str, Any], table_a: str) -> tuple[str, str]:
    """``(a_col, b_col)`` for *predicate*, remapped onto the fixed ``a``/``b`` aliases."""
    if predicate["table_left"].lower() == table_a.lower():
        return predicate["left"].name, predicate["right"].name
    return predicate["right"].name, predicate["left"].name


def _count_case_dirty_rejects(
    executor: ProbeExecutor,
    dialect: Optional[str],
    table_a: str,
    table_b: str,
    core: list[dict[str, Any]],
    extra: list[dict[str, Any]],
) -> Optional[tuple[int, int]]:
    """``(total_rejects, casing_only_rejects)`` among core-matched rows the extra
    predicate(s) reject.

    Rather than just comparing row counts with/without the extra predicate
    (which can't tell "dirty data" apart from "correctly excluding a
    different real-world row"), this inspects the rejected rows directly:
    ``casing_only_rejects`` counts how many of them would pass if the extra
    predicate's columns were compared via ``LOWER(TRIM(...))`` instead of raw
    equality — i.e. the two sides agree in substance and differ only in
    casing/whitespace. ``None`` on probe failure.
    """
    d = _sqlglot_dialect(dialect)
    a_ref = exp.table_(table_a).sql(dialect=d)
    b_ref = exp.table_(table_b).sql(dialect=d)

    def _eq(a_col: str, b_col: str, normalized: bool) -> str:
        a_ref_col = f"a.{exp.column(a_col).sql(dialect=d)}"
        b_ref_col = f"b.{exp.column(b_col).sql(dialect=d)}"
        if normalized:
            return f"LOWER(TRIM({a_ref_col})) = LOWER(TRIM({b_ref_col}))"
        return f"{a_ref_col} = {b_ref_col}"

    core_on = " AND ".join(_eq(*_oriented_columns(p, table_a), False) for p in core)
    extra_raw = " AND ".join(_eq(*_oriented_columns(p, table_a), False) for p in extra)
    extra_normalized = " AND ".join(
        _eq(*_oriented_columns(p, table_a), True) for p in extra
    )

    # SUM(CASE WHEN ... THEN 1 ELSE 0 END) instead of COUNT(*) FILTER (WHERE ...):
    # FILTER is Postgres/standard-SQL syntax, not supported by MySQL, so this
    # keeps the probe portable across every dialect this module supports.
    sql = (
        f"SELECT "
        f"SUM(CASE WHEN NOT ({extra_raw}) THEN 1 ELSE 0 END) AS total_rejects, "
        f"SUM(CASE WHEN NOT ({extra_raw}) AND ({extra_normalized}) THEN 1 ELSE 0 END) "
        f"AS casing_only_rejects "
        f"FROM {a_ref} a JOIN {b_ref} b ON {core_on}"
    )
    res = executor.run(sql, purpose="join_case_dirty_rejects")
    if not res["ok"] or not res["rows"]:
        return None
    row = res["rows"][0]
    try:
        return int(row.get("total_rejects") or 0), int(
            row.get("casing_only_rejects") or 0
        )
    except (TypeError, ValueError):
        return None


def find_case_dirty_join_mismatches(
    executor: ProbeExecutor,
    dialect: Optional[str],
    sql: str,
    database_name: Optional[str],
) -> list[dict[str, Any]]:
    """Return compound joins where an extra predicate rejects real matches
    purely due to casing/whitespace, not a real difference.

    When a query joins two tables on more than one equality predicate and one
    of them is the graph-verified relationship, the extra predicate(s) are
    usually added as defensive "belt and suspenders." That's only safe to
    remove if it's actually dirty data — an extra predicate can also be doing
    its job, correctly excluding a genuinely different real-world row that
    happens to share the verified key. A plain row-count comparison can't
    tell those apart, so this inspects the *rejected* rows directly: among
    rows the verified predicate alone matches but the extra predicate
    rejects, are all of them cases where the extra predicate's columns agree
    once ``LOWER(TRIM(...))``'d? Only then is it flagged — a real semantic
    difference (even one rejected row) leaves the check silent.

    Each entry: ``{"table_a", "table_b", "verified_columns", "extra_columns",
    "rejected_rows", "verdict": "case_dirty_join"}``.
    """
    try:
        tree = sqlglot.parse_one(sql, read=_sqlglot_dialect(dialect))
    except Exception as exc:  # noqa: BLE001 — never break the pipeline on a parse error
        logger.info("case_dirty_join_check: could not parse SQL (%s)", exc)
        return []
    if tree is None:
        return []

    all_nodes, by_key = _table_nodes(tree)
    if not all_nodes:
        return []

    groups = _group_join_equalities_by_table_pair(tree, all_nodes, by_key)
    mismatches: list[dict[str, Any]] = []

    for predicates in groups.values():
        if len(predicates) < 2 or not executor.budget_left:
            continue

        verified = [p for p in predicates if _is_verified_edge(p, database_name)]
        verified_ids = {id(p) for p in verified}
        extra = [p for p in predicates if id(p) not in verified_ids]
        if not verified or not extra:
            continue  # need a known-good core to test the extra predicate against

        table_a, table_b = predicates[0]["table_left"], predicates[0]["table_right"]
        counts = _count_case_dirty_rejects(
            executor, dialect, table_a, table_b, verified, extra
        )
        if counts is None:
            continue  # probe failed
        total_rejects, casing_only_rejects = counts

        if total_rejects > 0 and total_rejects == casing_only_rejects:
            # Every rejected row agrees once normalized — a confirmed casing
            # artifact, not a real difference.
            mismatches.append(
                {
                    "table_a": table_a,
                    "table_b": table_b,
                    "verified_columns": [
                        f"{p['table_left']}.{p['left'].name} = "
                        f"{p['table_right']}.{p['right'].name}"
                        for p in verified
                    ],
                    "extra_columns": [
                        f"{p['table_left']}.{p['left'].name} = "
                        f"{p['table_right']}.{p['right'].name}"
                        for p in extra
                    ],
                    "rejected_rows": total_rejects,
                    "verdict": "case_dirty_join",
                }
            )

    return mismatches


def build_case_dirty_join_repair_error(mismatches: list[dict[str, Any]]) -> str:
    """Render case-dirty-join mismatches into a targeted reconstruction instruction."""
    lines = []
    for m in mismatches:
        core = ", ".join(m["verified_columns"])
        extra = ", ".join(m["extra_columns"])
        lines.append(
            f"- Joining {m['table_a']} to {m['table_b']} on {core} AND {extra} "
            f"silently rejects {m['rejected_rows']} row(s) that the verified "
            f"relationship ({core}) alone matches — and in every one of them, "
            f"{extra} agree once both sides are lower-cased and trimmed. This "
            f"is dirty data (casing/whitespace), not a real mismatch."
        )
    body = "\n".join(lines)
    return (
        "One or more JOINs use an extra condition that rejects real matches "
        "purely due to inconsistent casing/whitespace between the two tables:\n"
        f"{body}\n\n"
        "Wrap that condition's columns in LOWER(TRIM(...)) on both sides "
        "instead of comparing them raw. Do NOT drop the condition — it may "
        "still be needed to correctly exclude other, genuinely different "
        "rows. Change ONLY the affected condition; keep all other joins, "
        "columns, grouping, and filters exactly as they are."
    )


__all__ = [
    "find_join_path_mismatches",
    "build_join_path_repair_error",
    "find_case_dirty_join_mismatches",
    "build_case_dirty_join_repair_error",
    "try_self_apply_wrong_column_fixes",
    "try_self_apply_missing_bridge_fixes",
]
