# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Text to PQL pipeline (the twin of ``sql_gen``): retrieve, generate, validate, predict, repair.

Differences from SQL generation:

* The model emits a PQL query **and**, optionally, an *entity-selection* read-only SQL that scopes which
  entities to score (the D4 ``SQL pre-filter -> entity-ID list -> predict(indices=...)`` pattern).
* Validity is checked with the **cheap** graph parse (``KumoModel.validate_pql`` -> ``parse_query``, no
  inference): its error is the repair signal. ``predict()`` then produces the actual answer (and also
  self-validates). Entity scaling above the per-call cap is handled by ``KumoModel.predict`` (batch mode).
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Protocol

from langchain_core.language_models import BaseChatModel
from nemo_retriever.tabular_data.sql_database import SQLDatabase

from gsf.retrieval.kumo.prompts import build_pql_prompt
from gsf.utils.llm_invoke import invoke_text

logger = logging.getLogger(__name__)


def quote_ident(identifier: str, char: str = '"') -> str:
    """Quote a SQL identifier with *char*, escaping embedded occurrences by doubling."""
    return char + identifier.replace(char, char * 2) + char


def _sql_table(table: str, table_names: dict[str, str] | None) -> str:
    """Schema-qualified SQL name for a graph table, else a quoted bare name.

    The Kumo graph refers to tables by their bare name (``GPUS``); the live
    database needs ``"SCHEMA"."GPUS"``. ``table_names`` maps the graph name to
    that qualified form (matched case-insensitively).
    """
    if table_names:
        for name, qualified in table_names.items():
            if name.lower() == table.lower():
                return qualified
    return quote_ident(table)


_FROM_JOIN_RE = re.compile(r'\b(FROM|JOIN)\s+"?([A-Za-z_]\w*)"?', re.IGNORECASE)


def _qualify_from_clauses(sql: str, table_names: dict[str, str] | None) -> str:
    """Schema-qualify bare table names after FROM / JOIN in model-authored SQL.

    Only the table token following FROM / JOIN is rewritten (quoted or not), so
    column references are untouched. Used for the LLM's entity-selection SQL,
    whose table names are the bare graph names.
    """
    if not table_names:
        return sql
    lookup = {name.lower(): qualified for name, qualified in table_names.items()}

    def repl(match: "re.Match[str]") -> str:
        qualified = lookup.get(match.group(2).lower())
        return f"{match.group(1)} {qualified}" if qualified else match.group(0)

    return _FROM_JOIN_RE.sub(repl, sql)


_PQL_FENCE = re.compile(r"```pql\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_GENERIC_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_PREDICT_START = re.compile(r"\bPREDICT\b", re.IGNORECASE)
_FOR_ENTITY = re.compile(
    r"\bFOR\s+(?P<each>EACH\s+)?(?P<table>[A-Za-z_][\w]*)\.(?P<pk>[A-Za-z_][\w]*)",
    re.IGNORECASE,
)
_LIST_DISTINCT = re.compile(r"\bLIST_DISTINCT\b", re.IGNORECASE)
_CHANGE_INTENT = re.compile(
    r"\b(gain|gains|loss|losses|increase|increases|decrease|decreases|change|changes|delta|growth|"
    r"decline|declines|drop|drops|rise|rises|up|down)\b",
    re.IGNORECASE,
)
_CHANGE_COL_MARKER = re.compile(
    r"(?:^|_)(?:change|delta|qoq|mom|yoy)(?:_|$)", re.IGNORECASE
)
_WINDOWED_AGG_TARGET = re.compile(
    r"\b(SUM|AVG|MIN|MAX)\s*\(\s*([A-Za-z_]\w*)\s*\.\s*([A-Za-z_]\w*)"
    r"\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*([A-Za-z]+)\s*\)",
    re.IGNORECASE,
)
_METRIC_SUFFIX = re.compile(
    r"(_usd|_amount|_count|_qty|_rate|_pct|_percent|_value|_total)$", re.IGNORECASE
)

# --- Static PQL lint (catches known-bad shapes BEFORE the backend, with an actionable repair message) ----
# These shapes pass KumoRFM's cheap parse (validate_pql) yet fail at predict time with opaque backend errors
# (Invalid Syntax, Internal Server Error, Missing foreign key). Catching them here turns a slow, uninformative
# multi-attempt failure into a targeted repair signal the next generation can actually act on.
_SELECT_TOKEN = re.compile(r"\bSELECT\b", re.IGNORECASE)
_PQL_BANNED_TIME_FUNCS = re.compile(
    r"\b(CURRENT_TIMESTAMP|CURRENT_DATE|CURRENT_TIME|GETDATE|SYSDATE|SYSTIMESTAMP|TODAY)\b|\bNOW\s*\(",
    re.IGNORECASE,
)
_RANK_TOP = re.compile(r"\bRANK\s+TOP\b", re.IGNORECASE)
_FOR_EACH_KW = re.compile(r"\bFOR\s+EACH\b", re.IGNORECASE)
_AGG_OPEN = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX|LIST_DISTINCT)\s*\(", re.IGNORECASE)
_TABLE_COL = re.compile(r"\b([A-Za-z_]\w*)\s*\.\s*([A-Za-z_]\w*|\*)")
# The trailing ``, <start>, <end>, <unit>`` window args inside an aggregation (e.g. ``, 0, 90, days``).
_WINDOW_TAIL = re.compile(r",\s*-?\d+\s*,\s*-?\d+\s*,\s*[A-Za-z]+\s*$")
_REL_COMPARISON = re.compile(
    r"\b([A-Za-z_]\w*)\s*\.\s*([A-Za-z_]\w*)\s*(>=|<=|>|<(?!>))"
)
_NON_ORDINAL_STYPES = frozenset({"categorical", "multicategorical", "ID", "text"})


class PqlStaticError(ValueError):
    """A PQL shape rejected by the cheap static lint before it reaches the backend."""


def _balanced_paren_body(text: str, open_idx: int) -> str:
    """Return the substring inside the parenthesis that opens at ``open_idx`` (handles nesting)."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i]
    return text[open_idx + 1 :]


def _aggregation_clauses(pql: str) -> list[tuple[str, str | None]]:
    """Yield ``(aggregated_table, inner_where_or_None)`` for each ``AGG(<table>.* [WHERE ...], ...)``.

    The inner WHERE is the part of the aggregation body after ``WHERE`` and before the trailing window args;
    the FOR-EACH entity filter lives outside the parens and is intentionally not returned here.
    """
    clauses: list[tuple[str, str | None]] = []
    for match in _AGG_OPEN.finditer(pql):
        body = _balanced_paren_body(pql, match.end() - 1)
        body = _WINDOW_TAIL.sub("", body).strip()
        parts = re.split(r"\bWHERE\b", body, maxsplit=1, flags=re.IGNORECASE)
        table_match = _TABLE_COL.search(parts[0])
        if not table_match:
            continue
        where_clause = parts[1] if len(parts) > 1 else None
        clauses.append((table_match.group(1), where_clause))
    return clauses


def validate_pql_static(
    pql: str,
    *,
    edges: list[tuple[str, str, str]] | None = None,
    col_stypes: dict[str, dict[str, str]] | None = None,
) -> None:
    """Reject known-bad PQL shapes with an actionable message, before the backend is ever called.

    Catches the failure families seen in the ERP stress sweep: a SQL subquery inside PQL/``ASSUMING``, SQL
    time functions like ``CURRENT_TIMESTAMP``, a cross-table predicate inside an aggregation ``WHERE``, a
    misplaced ``RANK TOP``, and — when ``edges`` (the graph's ``(src, fk, dst)`` foreign keys) are supplied —
    an aggregation whose event table has no DIRECT foreign key to the prediction entity (KumoRFM does not
    traverse multi-hop paths inside an aggregation). When ``col_stypes`` (``table -> {column -> declared
    stype}``) is supplied, it also rejects a ``>``/``<`` comparison against a declared non-ordinal column.
    Raises :class:`PqlStaticError` (a ``ValueError``) so it flows into the existing repair loop as
    ``prev_error``.
    """
    text = pql.strip()
    if not text:
        raise PqlStaticError("Empty PQL.")

    # 1. PQL has no subqueries — a stray SELECT means the model wrote SQL inside the PREDICT (e.g. ASSUMING).
    if _SELECT_TOKEN.search(text):
        raise PqlStaticError(
            "PQL has no SQL subqueries — remove the SELECT. To reach a related table, rely on the graph's "
            "foreign keys (the entity→event link is implicit); to restrict entities, use a plain WHERE on the "
            "entity table after FOR EACH; and put any entity-selection SQL in the SEPARATE ```sql block — "
            "never inside PREDICT and never inside ASSUMING."
        )

    # 2. SQL time functions are not valid in PQL — the forward window is the (start, end, unit) args.
    banned = _PQL_BANNED_TIME_FUNCS.search(text)
    if banned:
        raise PqlStaticError(
            f"PQL does not support the SQL time expression '{banned.group(0).strip()}'. The prediction window "
            "is already relative to the data's anchor time via the (start, end, unit) arguments. To restrict "
            "to entities in a future window (e.g. open / at-risk orders), do that date filter in the SEPARATE "
            "entity-selection ```sql block, not inside the PQL."
        )

    # 3. RANK TOP belongs immediately after the target, BEFORE FOR EACH — never after it.
    for_each = _FOR_EACH_KW.search(text)
    if for_each:
        rank = _RANK_TOP.search(text)
        if rank and rank.start() > for_each.start():
            raise PqlStaticError(
                "RANK TOP is misplaced: it goes immediately after the target and BEFORE FOR EACH "
                "(PREDICT <target> RANK TOP k FOR EACH <entity>.<pk>), not after it. Move it, or drop it — a "
                "binary/regression target needs no RANK because its scores already rank the entities."
            )

    # 4. An aggregation's WHERE can only filter the aggregated event table (no cross-table predicate).
    for agg_table, where_clause in _aggregation_clauses(text):
        if not where_clause:
            continue
        for tbl, _col in _TABLE_COL.findall(where_clause):
            if tbl.lower() != agg_table.lower():
                raise PqlStaticError(
                    f"The WHERE inside the {agg_table} aggregation can only filter columns of '{agg_table}' "
                    f"(the aggregated event table); it references '{tbl}', which PQL cannot express as a "
                    f"cross-table predicate inside an aggregation. Drop the '{tbl}.' condition (or use a column "
                    f"that lives on '{agg_table}'); to scope by a related table's attribute, filter the "
                    f"entities in the entity-selection ```sql block instead."
                )

    # 5. Each aggregation's event table must have a DIRECT foreign key to the FOR EACH entity. KumoRFM does not
    #    auto-traverse multi-hop paths inside an aggregation (e.g. po_receipts -> purchase_orders -> suppliers),
    #    so it rejects such queries with an opaque "Missing foreign key". Catch it here and name the tables that
    #    DO link directly to the entity (typically a rollup), so the repair lands on a valid shape.
    if edges:
        entity = parse_entity(text)
        if entity is not None:
            entity_table = entity[0].lower()
            direct = sorted(
                {src for src, _fk, dst in edges if dst.lower() == entity_table}
            )
            direct_lower = {t.lower() for t in direct}
            for agg_table, _where in _aggregation_clauses(text):
                at = agg_table.lower()
                if at != entity_table and at not in direct_lower:
                    alts = (
                        ", ".join(direct)
                        if direct
                        else "(none — pick a different entity)"
                    )
                    raise PqlStaticError(
                        f"'{agg_table}' has no direct foreign key to the prediction entity '{entity[0]}', and "
                        f"KumoRFM cannot traverse a multi-hop path inside an aggregation. Aggregate a table that "
                        f"links DIRECTLY to '{entity[0]}' instead (one of: {alts}) — a monthly/daily rollup is "
                        f"usually the right choice — or change the entity. To bring in an attribute that lives "
                        f"two hops away, pre-filter the entities in the entity-selection ```sql block."
                    )

    if col_stypes:
        for tbl, col, op in _REL_COMPARISON.findall(text):
            stype = col_stypes.get(tbl.lower(), {}).get(col.lower())
            if stype in _NON_ORDINAL_STYPES:
                raise PqlStaticError(
                    f"'{tbl}.{col}' is a {stype} column and cannot be compared with '{op}'. Compare a "
                    f"categorical column with '=' / '!=' against one of its sampled values (see the column "
                    f"list) instead — e.g. a status/flag value — or use a numeric column for a threshold."
                )


class PqlValidatorModel(Protocol):
    """Minimal KumoModel interface used here (satisfied by :class:`kumo_rfm.kumo_client.KumoModel`)."""

    def validate_pql(self, query: str) -> Any: ...

    def predict(
        self, query: str, indices: list[Any] | None = None, **kwargs: Any
    ) -> Any: ...


@dataclass
class PqlGenerationResult:
    """Outcome of a text-to-PQL generation + prediction."""

    question: str
    pql: str = ""
    entity_sql: str | None = None
    success: bool = False
    attempts: int = 0
    error: str | None = None
    num_entities: int = 0
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    explanation: str | None = None
    explanation_warning: str | None = None
    explanation_details: Any = None
    group_by: str | None = None
    truncated: bool = False
    note: str | None = None
    prediction_table: str | None = None
    prediction_table_columns: list[str] = field(default_factory=list)


class PqlGroupByError(ValueError):
    """A grouped-prediction request that cannot be served (bad group-by column, or wrong target type)."""


def _is_forecast(pql: str) -> bool:
    """A ``FORECAST n TIMEFRAMES`` query: predicts a time series for a SINGLE entity (SDK constraint)."""
    return re.search(r"\bFORECAST\b", pql, re.IGNORECASE) is not None


_EXISTENCE_COUNT = re.compile(r"PREDICT\s+COUNT\s*\(\s*\w+\s*\.\s*\*", re.IGNORECASE)


def _is_existence_count_pql(pql: str) -> bool:
    """True for the degenerate ``COUNT(<child>.*)`` parent→immediate-child existence shape with no
    discriminating filter — KumoRFM device-side-asserts on it DETERMINISTICALLY, so an assert there is the
    shape and must fail fast. Any other shape that asserts (e.g. a large SUM aggregation) is hitting an
    INTERMITTENT CUDA fault and should be retried instead. A filter on the aggregation makes the count a
    real outcome (e.g. late/slip), so a WHERE before FOR EACH means it is not the degenerate shape."""
    match = _EXISTENCE_COUNT.search(pql)
    if not match:
        return False
    for_each = _FOR_EACH_KW.search(pql)
    head = pql[: for_each.start()] if for_each else pql
    return "where" not in head.lower()


def _order_forecast(df: Any) -> Any:
    """Order a forecast result chronologically by its ``TIME`` column (timeframes in sequence)."""
    for col in df.columns:
        if col.upper() == "TIME":
            try:
                return df.sort_values(col)
            except Exception:  # noqa: BLE001 - odd column; keep original order
                return df
    return df


def _rank_prediction(df: Any) -> Any:
    """Sort a prediction descending by its score column (``True_PROB``/``*_PROB``/``*_PRED``) so the agent
    sees the top entities first; leaves order untouched if there is no such column (e.g. pre-ranked
    RANK/link-prediction output)."""
    cols = list(df.columns)
    lowered = {c.lower(): c for c in cols}
    sort_col = None
    for key in ("true_prob", "probability"):
        if key in lowered:
            sort_col = lowered[key]
            break
    if sort_col is None:
        for c in cols:
            cl = c.lower()
            if cl.endswith("_prob") and cl != "false_prob":
                sort_col = c
                break
    if sort_col is None:
        for c in cols:
            if c.lower().endswith("_pred") or c.upper() == "TARGET_PRED":
                sort_col = c
                break
    if sort_col is not None:
        try:
            return df.sort_values(sort_col, ascending=False)
        except Exception:  # noqa: BLE001 - non-numeric/odd column; fall back to original order
            return df
    return df


# Prediction-frame columns that are never the predicted value (KumoRFM metadata columns).
_NON_VALUE_COLS = frozenset({"entity", "anchor_timestamp", "time"})


def _entity_id_column(df: Any, pk: str) -> str:
    """The column in a prediction frame that holds the entity primary-key values (KumoRFM uses ``ENTITY``)."""
    lowered = {c.lower(): c for c in df.columns}
    if "entity" in lowered:
        return lowered["entity"]
    if pk.lower() in lowered:
        return lowered[pk.lower()]
    return list(df.columns)[0]


def _prediction_value_column(df: Any, *, id_col: str) -> str:
    """The single numeric column holding the predicted value/probability to aggregate.

    Handles regression (one predicted-value column) and binary classification (the positive-class
    probability, ``*_True`` / ``True_PROB`` / ``*_PROB``). Raises for shapes that can't be aggregated by a
    scalar (multi-class, link prediction)."""
    import pandas.api.types as pat

    candidates = [
        c for c in df.columns if c != id_col and c.lower() not in _NON_VALUE_COLS
    ]
    lowered = {c.lower(): c for c in candidates}
    for key in ("true_prob", "probability"):
        if key in lowered:
            return lowered[key]
    positive = [
        c
        for c in candidates
        if (c.lower().endswith("_true") or c.lower().endswith("_prob"))
        and "false" not in c.lower()
    ]
    if len(positive) == 1:
        return positive[0]
    numeric = [c for c in candidates if pat.is_numeric_dtype(df[c])]
    if len(numeric) == 1:
        return numeric[0]
    for c in candidates:
        if c.lower().endswith("_pred") or c.upper() == "TARGET_PRED":
            return c
    raise PqlGroupByError(
        "This prediction's output cannot be aggregated by a group (it is multi-class or link prediction, which "
        "has no single value to sum/average). Group-by is supported for regression and binary predictions."
    )


def aggregate_prediction_by(
    prediction: Any,
    *,
    table: str,
    pk: str,
    group_by: str,
    connector: SQLDatabase,
    table_names: dict[str, str] | None = None,
) -> Any:
    """Roll a per-entity prediction up to per-group totals — the post-hoc aggregation PQL cannot do itself.

    Joins each scored entity to its ``group_by`` attribute (a read-only lookup on the entity table), then
    groups and aggregates the predicted value: ``total`` (sum, for additive targets like revenue/counts),
    ``average`` (mean, for rates/probabilities) and ``n_entities`` (group size). Returns a small DataFrame
    sorted by total descending. Raises :class:`PqlGroupByError` if ``group_by`` is not a column of the
    entity table or the prediction has no scalar value to aggregate.
    """
    id_col = _entity_id_column(prediction, pk)
    value_col = _prediction_value_column(prediction, id_col=id_col)
    try:
        dim = connector.execute(
            f"SELECT {pk}, {group_by} FROM {_sql_table(table, table_names)} "
            f"WHERE {pk} IS NOT NULL"
        )
    except Exception as exc:  # noqa: BLE001 - turn an opaque warehouse error into an actionable group-by error
        try:
            valid = list(
                connector.execute(
                    f"SELECT * FROM {_sql_table(table, table_names)} LIMIT 1"
                ).columns
            )
        except Exception:  # noqa: BLE001 - best-effort column listing for the message
            valid = []
        raise PqlGroupByError(
            f"Cannot group by '{group_by}': it is not a column of '{table}'."
            + (f" Valid columns: {valid}." if valid else "")
        ) from exc

    left = prediction[[id_col, value_col]].copy()
    left["__key"] = left[id_col].astype(str)
    right = dim.copy()
    right["__key"] = right[pk].astype(str)
    merged = left.merge(right[["__key", group_by]], on="__key", how="left")
    grouped = (
        merged.dropna(subset=[group_by])
        .groupby(group_by)[value_col]
        .agg(total="sum", average="mean", n_entities="count")
        .reset_index()
    )
    return grouped.sort_values("total", ascending=False)


# Markers for a transient infra/server hiccup (not a problem with the query). The PQL has already passed
# validate_pql, so retrying the SAME query is right (e.g. the DuckDB mirror "closed pending query result"
# race clears on retry, and KumoRFM 5xx/timeouts are transient).
_TRANSIENT_EXEC_MARKERS = (
    "closed pending query result",
    "internal server error",
    "an unexpected exception occurred",
    "model inference fail",
    "service unavailable",
    "temporarily unavailable",
    "timed out",
    "timeout",
    "connection reset",
    "connection aborted",
    "502",
    "503",
    "504",
    "429",
)

# Markers for a KumoRFM context/GPU-capacity failure (a wide graph like gpu_fleet lands ~27MB of context,
# right at the SDK's 30MB ceiling). The GPU errors (CUDA "illegal memory access" / OOM) are INTERMITTENT —
# the same full-neighbourhood request that fails one moment can succeed on retry — so they are retried at
# FULL accuracy first (see _predict_resilient). The "context size exceeds" rejection is the exception: it is
# DETERMINISTIC (same query -> same oversize context), so it steps straight down to a smaller neighbourhood
# rather than re-hitting the same wall 3 times (see _retry_at_full_neighbourhood). Shrinking the
# neighbourhood is the last resort to salvage *a* prediction when full settings cannot complete.
_CONTEXT_CAPACITY_MARKERS = (
    "cuda",
    "illegal memory access",
    "out of memory",
    "context size exceeds",
)

# A parse error at PREDICT time is spurious: the query already passed validate_pql (the cheap parse) moments
# earlier, so a parse failure during predict is the KumoRFM server in a degraded state (observed right after a
# transient 5xx, alongside gRPC GOAWAY). Retry the SAME call rather than discard a known-valid query — link
# prediction in particular recovers within a couple of retries once the server settles.
_SPURIOUS_PREDICT_PARSE_MARKERS = (
    "failed to parse query",
    "could not process the text",
)

_EMPTY_CONTEXT_MARKERS = (
    "failed to collect any context examples",
    "too restrictive",
)


def _is_transient_exec_error(message: str) -> bool:
    """True if a predict error is a transient infra/server hiccup."""
    return any(marker in message.lower() for marker in _TRANSIENT_EXEC_MARKERS)


def _is_context_capacity_error(message: str) -> bool:
    """True if a predict error is a KumoRFM context/GPU-capacity failure."""
    return any(marker in message.lower() for marker in _CONTEXT_CAPACITY_MARKERS)


def _is_context_size_limit_error(message: str) -> bool:
    """True for the DETERMINISTIC 'context size exceeds the limit' rejection. The SDK builds the context,
    measures it, and rejects it for being over the ceiling, so the identical query yields the identical
    oversize context every time — retrying the same neighbourhood cannot help, only a smaller one can."""
    return "context size exceeds" in message.lower()


# Markers for a TERMINAL backend failure: the generated *shape* is not serveable for this dataset, so neither
# retrying the same call nor regenerating a similar query will help. Unlike the (intermittent) capacity errors
# above, these are deterministic, so we fail fast with a clear message instead of burning the repair budget
# and the neighbourhood backoff re-hitting the same wall.
#   * "device-side assert" — a deterministic KumoRFM CUDA assertion hit by degenerate parent→immediate-child
#     existence targets (e.g. COUNT(order_lines.*) FOR EACH sales_orders); stepping the neighbourhood down
#     does not help (it is the data/shape, not the context size).
# NOTE: link-prediction's "Unsupported cast from list<item: string> to utf8" is NOT here — it is fixed at the
# source by the LIST_DISTINCT string-dtype compatibility shim in kumo_client, so recommendations now run.
_UNSUPPORTED_SHAPE_MARKERS = ("device-side assert",)


def _is_unsupported_shape_error(message: str) -> bool:
    """True if a predict error is a deterministic, unrecoverable 'this shape is not serveable' failure."""
    return any(marker in message.lower() for marker in _UNSUPPORTED_SHAPE_MARKERS)


def _is_empty_context_error(message: str) -> bool:
    """True if a predict/explain error is an empty entity set (WHERE matched no learnable entity)."""
    return any(marker in message.lower() for marker in _EMPTY_CONTEXT_MARKERS)


def _friendly_empty_context_message() -> str:
    """A user-facing explanation for an empty-entity-set failure, with the concrete next step."""
    return (
        "KumoRFM found no entity matching the prediction's filter, so it could not gather any context "
        "examples. The WHERE clause likely references an id that does not exist (or a placeholder). Pass a "
        "real entity id taken from a prediction ranking or an rfm__sql_query result, not a placeholder."
    )


def _is_spurious_predict_parse_error(message: str) -> bool:
    """True if a PREDICT-time parse error is the spurious kind (the query already passed validate_pql)."""
    return any(marker in message.lower() for marker in _SPURIOUS_PREDICT_PARSE_MARKERS)


def _friendly_unsupported_message(pql: str, raw_error: str) -> str:
    """A user-facing explanation (no raw backend stack text) for a terminal unsupported-shape failure, with a
    concrete next step the agent can take."""
    return (
        "KumoRFM could not compute this prediction shape for this dataset — the parent→child relationship "
        "is not learnable as a generic existence/count prediction. Predict a specific outcome instead "
        "(e.g. a late/slip status with a WHERE filter on the event table), or use rfm__sql_query for a "
        "historical count."
    )


def _is_retryable_exec_error(message: str) -> bool:
    """True if a predict error is worth retrying the SAME call: transient infra hiccups, the (intermittent)
    capacity errors, AND a spurious post-validate parse error (server degradation). Retrying capacity errors at
    the FULL neighbourhood is what preserves accuracy — we never reduce the model's context unless full settings
    persistently fail. A terminal unsupported-shape error is never retried (it is deterministic), even if it
    mentions CUDA."""
    if _is_unsupported_shape_error(message):
        return False
    return (
        _is_transient_exec_error(message)
        or _is_context_capacity_error(message)
        or _is_spurious_predict_parse_error(message)
    )


def _retry_at_full_neighbourhood(message: str) -> bool:
    """Which errors to retry at the FULL neighbourhood before stepping down. Same as the general retry rule
    but EXCLUDES the deterministic context-size-limit rejection: retrying the identical query at the same
    neighbourhood just re-hits the same oversize context, so step straight down to a smaller one. Transient
    infra/GPU (CUDA/OOM) and spurious post-validate parse errors are still retried at full to keep accuracy."""
    return _is_retryable_exec_error(message) and not _is_context_size_limit_error(
        message
    )


_REFLECTION_FUNCTION_NAME = "__reflection__"


def _emit_retry_note(text: str) -> None:
    """Surface a tool-internal retry as a collapsible trace note on the running predict node, via NAT's
    intermediate-step pipeline, so the UI shows that an attempt failed and is being retried rather than an
    endless spinner. No-op without an active NAT context (CLI / tests); never raises into the predict path."""
    try:
        import uuid

        from nat.builder.context import Context
        from nat.data_models.intermediate_step import IntermediateStepPayload
        from nat.data_models.intermediate_step import IntermediateStepType
        from nat.data_models.intermediate_step import StreamEventData

        manager = getattr(Context.get(), "intermediate_step_manager", None)
        if manager is None:
            return
        step_id = str(uuid.uuid4())
        # Unique step name so each note stays its own trace step (folded under the running predict),
        # matching the side-channel note convention instead of merging into one.
        name = f"{_REFLECTION_FUNCTION_NAME}:{step_id[:8]}"
        manager.push_intermediate_step(
            IntermediateStepPayload(
                UUID=step_id,
                event_type=IntermediateStepType.FUNCTION_START,
                name=name,
                data=StreamEventData(input=name),
            )
        )
        manager.push_intermediate_step(
            IntermediateStepPayload(
                UUID=step_id,
                event_type=IntermediateStepType.FUNCTION_END,
                name=name,
                data=StreamEventData(input=name, output=text),
            )
        )
    except Exception:  # noqa: BLE001
        logger.debug("retry trace note failed", exc_info=True)


def _predict_with_retry(
    call: Callable[[], Any],
    *,
    retries: int,
    backoff: float = 1.5,
    retryable: Callable[[str], bool] = _is_retryable_exec_error,
) -> Any:
    """Run *call* (a no-arg predict), retrying the SAME query while *retryable* matches the error.

    Used only after ``validate_pql`` has passed, so a failure here is an execution error — a transient or
    intermittent-capacity one is retried as-is rather than discarding a valid query. ``retryable`` selects
    which errors to retry (default: transient + capacity); callers that want to react to capacity errors
    differently (e.g. explain, which steps down hops) pass a narrower predicate.
    """
    last_exc: Exception | None = None
    for i in range(retries + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - classified by message; re-raised below
            last_exc = exc
            if i == retries or not retryable(str(exc)):
                raise
            logger.info(
                "Predict error (retry %d/%d, same query): %s",
                i + 1,
                retries,
                str(exc)[:160],
            )
            if i == 0:
                _emit_retry_note(
                    "A prediction attempt failed (transient model or capacity error). Retrying."
                )
            time.sleep(backoff * (i + 1))
    raise last_exc  # pragma: no cover - loop always returns or raises


# How hard to retry at the FULL (accuracy-preserving) neighbourhood before stepping down. One retry rides
# through a single transient KumoRFM hiccup (a 5xx is often followed by a spurious parse error before it
# settles); beyond that, dropping to a smaller context escapes a persistent GPU/capacity fault faster than
# repeating the same heavy request. Tier 1 is therefore 2 attempts.
_FULL_NEIGHBORHOOD_RETRIES = 1

# Entity ceiling for a grouped prediction: high enough to score whole dimension populations (customers,
# products, suppliers) so per-group totals are complete, but bounded so a grouping over a huge event entity
# (e.g. every sales_order) can't explode — past this the totals are flagged partial.
_GROUP_BY_MAX_ENTITIES = 10_000

_PREDICT_BATCH_SIZE = 1000

# Last-resort fallbacks if the full neighbourhood persistently hits a capacity error. None = the SDK default
# for the run mode (full accuracy). The fallbacks step DOWN gradually (the SDK FAST default is [32, 32]) so
# we keep as much of the model's receptive field as possible while still fitting under the GPU ceiling; the
# final [8, 8] (~1-2MB context) is the smallest field worth salvaging a result with when larger ones fault.
_CONTEXT_BACKOFF_NEIGHBORS: tuple[list[int] | None, ...] = (
    None,
    [24, 24],
    [16, 16],
    [8, 8],
)


def _predict_resilient(
    predict_fn: Callable[[list[int] | None], Any],
    *,
    device_assert_terminal: bool = True,
) -> Any:
    """Run ``predict_fn(num_neighbors)`` accuracy-first.

    Predict at the FULL neighbourhood and retry it hard on intermittent infra/GPU errors — the same
    full-accuracy request usually succeeds on retry, so we never silently trade accuracy for a smaller
    context. ONLY if the full neighbourhood keeps hitting a context/GPU-capacity error after those retries
    do we step down to a smaller neighbourhood, purely to salvage *a* correct prediction rather than fail.
    Non-capacity errors propagate (to the regenerate loop).

    A CUDA ``device-side assert`` is deterministic ONLY for the degenerate existence-count shape
    (``device_assert_terminal=True`` → propagate immediately); for any other shape it is an INTERMITTENT
    fault (large-batch predicts hit it ~1-in-5), so it is retried and stepped down like a capacity error.
    """

    def _retryable(message: str) -> bool:
        if _is_unsupported_shape_error(message):
            return not device_assert_terminal
        return _retry_at_full_neighbourhood(message)

    last_exc: Exception | None = None
    for num_neighbors in _CONTEXT_BACKOFF_NEIGHBORS:
        retries = _FULL_NEIGHBORHOOD_RETRIES if num_neighbors is None else 0
        try:
            return _predict_with_retry(
                lambda nn=num_neighbors: predict_fn(nn),
                retries=retries,
                retryable=_retryable,
            )
        except Exception as exc:  # noqa: BLE001 - classified by message; re-raised below
            last_exc = exc
            assert_err = _is_unsupported_shape_error(str(exc))
            if assert_err and device_assert_terminal:
                raise
            if not (assert_err or _is_context_capacity_error(str(exc))):
                raise
            if num_neighbors is None:
                logger.warning(
                    "Full neighbourhood hit a KumoRFM capacity/assert error; falling back to a smaller "
                    "neighbourhood to salvage a prediction (reduced context). Error: %s",
                    str(exc)[:160],
                )
                _emit_retry_note(
                    "The prediction kept failing at full context. Retrying with a smaller context window to "
                    "recover a result."
                )
            else:
                logger.info(
                    "Capacity/assert error at neighbourhood %s; shrinking further.",
                    num_neighbors,
                )
                _emit_retry_note(
                    f"Still over capacity at neighbourhood {num_neighbors}. Shrinking the context further and retrying."
                )
    raise last_exc  # pragma: no cover - loop always returns or raises


def _predict_in_batches(
    indices: list[Any] | None,
    predict_call: Callable[[list[Any] | None, list[int] | None], Any],
    *,
    device_assert_terminal: bool = True,
) -> Any:
    """Run the resilient predict over ``indices`` in ordered chunks of at most ``_PREDICT_BATCH_SIZE``.

    The KumoRFM server raises a CUDA device-side assert on a single very large per-entity batch, so a
    whole-population prediction (thousands of entities) is split into chunks that each stay under the
    server's ceiling and the per-chunk frames are concatenated. A small scope (or ``None`` for a single
    entity / the SDK default) runs in one call. Each chunk goes through :func:`_predict_resilient`
    independently; since each entity's prediction depends only on its own subgraph, the concatenated result
    is equivalent to scoring the whole scope at once.
    """
    if not indices or len(indices) <= _PREDICT_BATCH_SIZE:
        return _predict_resilient(
            lambda nn: predict_call(indices, nn),
            device_assert_terminal=device_assert_terminal,
        )
    import pandas as pd

    frames = []
    for start in range(0, len(indices), _PREDICT_BATCH_SIZE):
        chunk = indices[start : start + _PREDICT_BATCH_SIZE]
        frames.append(
            _predict_resilient(
                lambda nn, c=chunk: predict_call(c, nn),
                device_assert_terminal=device_assert_terminal,
            )
        )
    return pd.concat(frames, ignore_index=True)


_EXPLAIN_BACKOFF_NEIGHBORS: tuple[list[int] | None, ...] = (None, [16], [8])


def _explain_resilient(explain_fn: Callable[[list[int] | None], Any]) -> Any:
    """Run ``explain_fn(num_neighbors)`` for one entity, accuracy-first: retry hard at the FULL neighbourhood
    before stepping the depth DOWN to salvage the call.

    Stepping down is not free for an explanation the way it is for a plain prediction: a shallow neighbourhood
    starves the gradient attribution and the per-cell scores collapse to zero, so the "top drivers" — the
    whole point of explain — silently vanish. KumoRFM's intermittent CUDA ``device-side assert`` fires in
    bursts (the same query that fails one moment succeeds the next), so we ride those out at full depth
    (``_FULL_NEIGHBORHOOD_RETRIES``) and only shrink as a last resort. First success wins; an empty entity set
    (bad/placeholder id) is terminal — a shallower neighbourhood cannot populate an empty filter.
    """
    last_exc: Exception | None = None
    for num_neighbors in _EXPLAIN_BACKOFF_NEIGHBORS:
        retries = _FULL_NEIGHBORHOOD_RETRIES if num_neighbors is None else 1
        try:
            return _predict_with_retry(
                lambda nn=num_neighbors: explain_fn(nn),
                retries=retries,
                retryable=_is_transient_exec_error,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_empty_context_error(str(exc)):
                logger.info(
                    "Explain hit an empty entity set (bad/placeholder id); not stepping the "
                    "neighbourhood down (it cannot help an empty filter)."
                )
                raise
            logger.info(
                "Explain failed at neighbourhood %s (%s); trying a shallower one.",
                num_neighbors or "default",
                str(exc)[:140],
            )
    raise last_exc  # pragma: no cover


def extract_pql(text: str) -> str:
    """Pull the PQL out of a model response (```pql fence, [PQL] tag, or first PREDICT...)."""
    match = _PQL_FENCE.search(text)
    if match and match.group(1).strip():
        candidate = match.group(1)
    else:
        tag = re.search(r"\[PQL\](.*?)\[/PQL\]", text, re.DOTALL | re.IGNORECASE)
        candidate = tag.group(1) if tag else text
    start = _PREDICT_START.search(candidate)
    pql = candidate[start.start() :] if start else candidate
    return pql.strip().rstrip(";").strip()


def extract_entity_sql(text: str) -> str | None:
    """Pull the optional entity-selection SQL (a ```sql block that is not the ```pql block)."""
    for match in _GENERIC_FENCE.finditer(text):
        block = match.group(1).strip()
        if (
            block
            and "PREDICT" not in block.upper()
            and re.match(r"(?is)^\s*(WITH|SELECT)\b", block)
        ):
            return block.rstrip(";").strip()
    return None


def parse_entity(pql: str) -> tuple[str, str] | None:
    """Parse the entity from population and single-entity PQL ``FOR`` clauses."""
    match = _FOR_ENTITY.search(pql)
    return (match.group("table"), match.group("pk")) if match else None


def _metric_stem(column: str) -> str:
    """Return the business-metric stem from a numeric column name."""
    return _METRIC_SUFFIX.sub("", column.lower())


def _explicit_change_candidate(
    table: str,
    column: str,
    *,
    col_stypes: dict[str, dict[str, str]] | None,
    schema_text: str = "",
) -> str | None:
    """Find a same-table explicit change/delta target for a level metric column.

    This is intentionally schema-driven and domain-agnostic: for a question about gains/losses/change, if
    the model picked ``revenue_usd`` and the same table exposes ``revenue_qoq_change_usd``, the latter is the
    target. Columns already containing a change marker are left alone.
    """
    if _CHANGE_COL_MARKER.search(column):
        return None
    table_cols: dict[str, str] = {}
    if col_stypes:
        for t_name, cols in col_stypes.items():
            if t_name.lower() == table.lower():
                table_cols = {c.lower(): c for c in cols}
                break
    if schema_text:
        table_col_pattern = re.compile(
            rf"\b{re.escape(table)}\.([A-Za-z_]\w*)\b", re.IGNORECASE
        )
        for match in table_col_pattern.finditer(schema_text):
            table_cols.setdefault(match.group(1).lower(), match.group(1))
        block_pattern = re.compile(
            rf"\b{re.escape(table)}\b\s*\((.*?)\)", re.IGNORECASE | re.DOTALL
        )
        for match in block_pattern.finditer(schema_text):
            for name in re.findall(r"\b[A-Za-z_]\w*\b", match.group(1)):
                if name.upper() not in {
                    "CREATE",
                    "TABLE",
                    "PRIMARY",
                    "KEY",
                    "FOREIGN",
                    "REFERENCES",
                }:
                    table_cols.setdefault(name.lower(), name)
    if not table_cols:
        return None
    stem = _metric_stem(column)
    candidates = [
        original
        for lowered, original in table_cols.items()
        if _CHANGE_COL_MARKER.search(lowered)
        and (lowered.startswith(stem + "_") or stem in lowered.split("_"))
    ]
    if not candidates:
        return None
    original_lower = column.lower()

    def score(name: str) -> tuple[int, int, int, int]:
        lowered = name.lower()
        return (
            1 if "qoq" in lowered else 0,
            1 if original_lower.endswith("_usd") and lowered.endswith("_usd") else 0,
            1 if "pct" not in lowered and "percent" not in lowered else 0,
            -len(lowered),
        )

    return max(candidates, key=score)


def prefer_explicit_change_targets(
    pql: str,
    question: str,
    *,
    col_stypes: dict[str, dict[str, str]] | None = None,
    schema_text: str = "",
) -> str:
    """Rewrite level targets to explicit change targets for gain/loss/change questions when available."""
    if not _CHANGE_INTENT.search(question or ""):
        return pql

    def repl(match: re.Match[str]) -> str:
        agg, table, column, start, end, unit = match.groups()
        replacement = _explicit_change_candidate(
            table, column, col_stypes=col_stypes, schema_text=schema_text
        )
        if not replacement:
            return match.group(0)
        return f"{agg}({table}.{replacement}, {start}, {end}, {unit})"

    return _WINDOWED_AGG_TARGET.sub(repl, pql)


def _resolve_indices(
    pql: str,
    entity_sql: str | None,
    connector: SQLDatabase,
    max_entities: int,
    table_names: dict[str, str] | None = None,
) -> list[Any]:
    """Resolve the entity-id list to score: from ``entity_sql`` if given, else all PKs of the entity table.

    Both paths go through the connector's ``execute`` seam, capped at ``max_entities``.
    Bare graph table names are schema-qualified via ``table_names`` so the SQL
    resolves against the live database.
    """
    if entity_sql:
        df = connector.execute(_qualify_from_clauses(entity_sql, table_names))
    else:
        entity = parse_entity(pql)
        if entity is None:
            return []
        table, pk = entity
        # Bare (unquoted) column identifier: the connector lower-cased the graph
        # column names, so a quoted lower-case name won't match a Snowflake column
        # (stored upper-case). Bare lets each dialect apply its default casing.
        df = connector.execute(
            f"SELECT DISTINCT {pk} FROM {_sql_table(table, table_names)} "
            f"WHERE {pk} IS NOT NULL LIMIT {int(max_entities)}"
        )
    if df.empty:
        return []
    ids = df.iloc[:, 0].dropna().tolist()
    if len(ids) > max_entities:
        logger.info("Capping entities from %d to %d.", len(ids), max_entities)
        ids = ids[:max_entities]
    return ids


def _persist_full_prediction(
    prediction: Any,
    *,
    pql: str,
    mirror_path: str,
    table: str,
    lock: Any,
    result: PqlGenerationResult,
) -> None:
    """Write the FULL per-entity prediction to a mirror scratch table for SQL comparison.

    Renames KumoRFM's ``ENTITY`` column to the entity primary key so a follow-up
    ``rfm__sql_query`` can JOIN the scratch table to source tables naturally.
    Best-effort: a persistence failure never fails an otherwise-good prediction.
    """
    from kumo_rfm.predict_store import persist_prediction

    try:
        entity = parse_entity(pql)
        pk = entity[1] if entity else None
        id_col = _entity_id_column(prediction, pk or "")
        frame = prediction
        if pk and id_col != pk:
            frame = prediction.rename(columns={id_col: pk})
        persist_prediction(mirror_path, table, frame, lock=lock)
        result.prediction_table = table
        result.prediction_table_columns = list(frame.columns)
    except Exception:  # noqa: BLE001 - persistence is an enhancement, never a hard dependency
        logger.warning(
            "Failed to persist prediction to scratch table %s", table, exc_info=True
        )


_ANCHOR_TABLE_RE = re.compile(
    r"(?:SUM|COUNT|AVG|MIN|MAX|FIRST|LAST|LIST_DISTINCT)\s*\(\s*([A-Za-z_]\w*)\.",
    re.IGNORECASE,
)
_ANCHOR_WINDOW_RE = re.compile(
    r",\s*\d+\s*,\s*(\d+)\s*,\s*(day|days|week|weeks|month|months|year|years)\s*\)",
    re.IGNORECASE,
)
_ANCHOR_UNIT_DAYS = {
    "day": 1,
    "days": 1,
    "week": 7,
    "weeks": 7,
    "month": 30,
    "months": 30,
    "year": 365,
    "years": 365,
}


def _forecast_anchor(
    pql: str,
    connector: SQLDatabase,
    time_columns: dict[str, str | None] | None,
    *,
    now: Any = None,
    table_names: dict[str, str] | None = None,
) -> Any:
    """Anchor a forward prediction at *now* (a true "forecast from today"), capped so the
    ``[anchor, anchor + horizon]`` window never runs past the data.

    KumoRFM derives its default anchor from the data's MAX timestamp. On a dataset whose data extends past
    today that makes it forecast a window with no in-context future and return degenerate (collapsed or
    wildly over-extrapolated) predictions. Anchoring at *now* lands the window inside real data; the cap
    keeps it valid even when *now* is already at/after the data's edge. Returns a ``pd.Timestamp`` for
    ``anchor_time``, or ``None`` to let the SDK derive it (non-temporal query / unknown horizon / no time
    column / lookup failure)."""
    if not time_columns:
        return None
    tm = _ANCHOR_TABLE_RE.search(pql or "")
    wm = _ANCHOR_WINDOW_RE.search(pql or "")
    if not tm or not wm:
        return None
    time_col = time_columns.get(tm.group(1))
    if not time_col:
        return None
    horizon_days = int(wm.group(1)) * _ANCHOR_UNIT_DAYS[wm.group(2).lower()]
    try:
        import pandas as pd

        df = connector.execute(
            f"SELECT MAX({time_col}) AS m FROM {_sql_table(tm.group(1), table_names)}"
        )
        data_max = (
            pd.Timestamp(df.iloc[0, 0])
            if (not df.empty and df.iloc[0, 0] is not None)
            else None
        )
    except Exception:  # noqa: BLE001 - anchor is best-effort; fall back to the SDK default
        return None
    if data_max is None:
        return None
    safe = data_max - pd.Timedelta(days=horizon_days + 7)
    now_ts = (
        pd.Timestamp(now)
        if now is not None
        else pd.Timestamp(pd.Timestamp.today().date())
    )
    return min(now_ts, safe)


def _resolve_single_index(
    pql: str,
    entity: str | None,
    connector: SQLDatabase,
    table_names: dict[str, str] | None = None,
) -> list[Any]:
    """Resolve one entity-id (the PK value to explain) to its correctly-typed value via the read-only guard.

    Matches on the string form of the PK so a user-supplied id (always a string) finds an int/str key alike.
    """
    parsed = parse_entity(pql)
    if parsed is None or entity is None:
        return []
    table, pk = parsed
    safe = str(entity).replace("'", "''")
    string_type = "VARCHAR"
    df = connector.execute(
        f"SELECT {pk} FROM {_sql_table(table, table_names)} "
        f"WHERE CAST({pk} AS {string_type}) = '{safe}' LIMIT 1"
    )
    return df.iloc[:, 0].tolist() if not df.empty else []


def _scope_explain_entity(pql: str, entity: str) -> str:
    """Make an explanation PQL visibly target the exact entity passed to the tool.

    The SDK still receives the warehouse-resolved, correctly typed value through ``indices``. Keeping the
    same scope in the PQL is important for trace accuracy and prevents a model-produced example id from
    describing a different entity than the one actually explained.
    """
    match = _FOR_ENTITY.search(pql)
    if match is None:
        return pql

    table, pk = match.group("table"), match.group("pk")
    tail = pql[match.end() :]
    ident = rf"{re.escape(table)}\s*\.\s*{re.escape(pk)}"
    value = r"(?:'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|[^\s]+)"

    # Discard a stale single-entity value or the legacy FOR EACH ... WHERE pk=value form before applying
    # the authoritative entity supplied by the tool call.
    if match.group("each"):
        tail = re.sub(
            rf"^\s+WHERE\s+{ident}\s*=\s*{value}(?=\s*(?:ASSUMING|$))",
            "",
            tail,
            flags=re.IGNORECASE,
        )
    else:
        tail = re.sub(rf"^\s*=\s*{value}", "", tail, count=1, flags=re.IGNORECASE)

    safe = str(entity).replace("'", "''")
    return f"{pql[: match.start()]}FOR {table}.{pk}='{safe}'{tail}"


_GENERIC_ENTITY_FILTER_RE = re.compile(
    r"(?P<head>\bFOR\s+EACH\s+(?P<table>[A-Za-z_]\w*)\.(?P<pk>[A-Za-z_]\w*))"
    r"\s+WHERE\s+(?P=table)\.(?P<column>[A-Za-z_]\w*)\s*=\s*'(?P<value>[^']+)'"
    r"(?P<tail>\s*(?:ASSUMING\b.*)?$)",
    re.IGNORECASE,
)
_GENERIC_ENTITY_TYPE_COLUMNS = {"customer_type", "account_type", "entity_type", "type"}


def _strip_unasked_generic_entity_filter(pql: str, question: str) -> str:
    """Remove inferred entity-type filters when the user asked for the whole population.

    LLM repair sometimes turns generic words like "accounts" into a type filter, e.g.
    ``customers.customer_type = 'enterprise'``. That silently narrows top-N predictions. Keep explicit
    user scopes: if the column or value is named in the question, leave the filter intact.
    """
    match = _GENERIC_ENTITY_FILTER_RE.search(pql)
    if not match:
        return pql
    column = match.group("column").lower()
    if column not in _GENERIC_ENTITY_TYPE_COLUMNS:
        return pql
    question_l = (question or "").lower()
    value_l = match.group("value").lower()
    if column in question_l or value_l in question_l:
        return pql
    return (
        pql[: match.start()]
        + match.group("head")
        + match.group("tail")
        + pql[match.end() :]
    )


def predict_all(
    pql: str,
    *,
    kumo_model: PqlValidatorModel,
    connector: SQLDatabase,
    entity_sql: str | None = None,
    max_entities: int = 2000,
    time_columns: dict[str, str | None] | None = None,
    table_names: dict[str, str] | None = None,
) -> Any:
    """Run a pre-built, known-good PQL over the full entity scope and return the COMPLETE prediction frame.

    Unlike :func:`generate_pql`, this takes an explicit PQL (no LLM generation) and does not truncate to a
    preview — callers that must compare or rank *every* scored entity (e.g. a population-wide forecast vs a
    baseline) need all rows. Resolves the entity scope (explicit ``entity_sql`` or the PQL's own WHERE) and
    runs the same accuracy-first resilient predict path as :func:`generate_pql`.
    """
    scope = entity_sql if entity_sql is not None else extract_entity_sql(pql)
    indices = _resolve_indices(pql, scope, connector, max_entities, table_names)
    anchor = _forecast_anchor(pql, connector, time_columns, table_names=table_names)

    def _predict_call(idx: list[Any] | None, num_neighbors: list[int] | None) -> Any:
        kw: dict[str, Any] = {}
        if num_neighbors is not None:
            kw["num_neighbors"] = num_neighbors
        if anchor is not None:
            kw["anchor_time"] = anchor
        return kumo_model.predict(pql, indices=idx or None, **kw)

    raw = _predict_in_batches(
        indices, _predict_call, device_assert_terminal=_is_existence_count_pql(pql)
    )
    return _rank_prediction(raw)


def generate_pql(
    question: str,
    *,
    llm: BaseChatModel,
    kumo_model: PqlValidatorModel,
    connector: SQLDatabase,
    graph_ddl: str,
    graph_edges: list[tuple[str, str, str]] | None = None,
    graph_col_stypes: dict[str, dict[str, str]] | None = None,
    column_reference: str = "",
    vector_store: Any = None,
    success_cache: Any = None,
    dataset: str = "",
    max_tries: int = 5,
    escalation_llm: BaseChatModel | None = None,
    max_entities: int = 2000,
    max_preview_rows: int = 50,
    explain: bool = False,
    explain_entity: str | None = None,
    on_pql: Callable[[str, str | None], None] | None = None,
    group_by: str | None = None,
    mirror_path: str | None = None,
    persist_table: str | None = None,
    persist_lock: Any = None,
    time_columns: dict[str, str | None] | None = None,
    table_names: dict[str, str] | None = None,
) -> PqlGenerationResult:
    """Generate a PQL, validate it cheaply against the graph, scope entities, and predict (with repair).

    The repair signal is the cheap ``validate_pql`` parse error (or a ``predict`` error); on failure
    ``(prev_pql, error)`` is fed back. ``escalation_llm`` handles the final attempt (D11).
    """
    # RAG retrieval intentionally omitted in this port (no verified-example corpus):
    # the prompt runs with no few-shot examples or domain docs.
    examples: list[dict[str, str]] = []
    docs: list[str] = []

    result = PqlGenerationResult(question=question)
    prev_pql: str | None = None
    prev_error: str | None = None

    # Snowflake stores unquoted identifiers uppercase (so the graph tables are
    # uppercase); tell the LLM to match that case in the PQL.
    try:
        dialect = getattr(connector, "dialect", None)
    except Exception:  # noqa: BLE001 - dialect is a property; never fail generation over it
        dialect = None

    for attempt in range(1, max_tries + 1):
        active_llm = (
            escalation_llm
            if (escalation_llm is not None and attempt == max_tries)
            else llm
        )
        prompt = build_pql_prompt(
            graph_ddl=graph_ddl,
            columns=column_reference,
            docs=docs,
            examples=examples,
            question=question,
            explain_entity=explain_entity if explain else None,
            prev_pql=prev_pql,
            prev_error=prev_error,
            dialect=dialect,
        )
        result.attempts = attempt
        try:
            raw = invoke_text(active_llm, prompt)
        except Exception as exc:  # noqa: BLE001 - LLM/gateway failure: record and retry, never crash the run
            prev_error = str(exc)
            result.error = prev_error
            logger.info(
                "PQL attempt %d/%d: LLM generation failed: %s",
                attempt,
                max_tries,
                prev_error[:160],
            )
            continue
        pql = extract_pql(raw)
        pql = prefer_explicit_change_targets(
            pql,
            question,
            col_stypes=graph_col_stypes,
            schema_text="\n".join([graph_ddl, column_reference, *docs]),
        )
        pql = _strip_unasked_generic_entity_filter(pql, question)
        entity_sql = extract_entity_sql(raw)
        if explain and explain_entity:
            pql = _scope_explain_entity(pql, explain_entity)
        result.pql = pql
        result.entity_sql = entity_sql
        if on_pql is not None and pql and pql.strip():
            try:
                on_pql(pql, entity_sql)
            except Exception:  # noqa: BLE001 - surfacing the PQL early must never break the run
                logger.debug("on_pql callback failed", exc_info=True)
        try:
            validate_pql_static(pql, edges=graph_edges, col_stypes=graph_col_stypes)
            kumo_model.validate_pql(pql)
            if explain:
                indices = _resolve_single_index(
                    pql, explain_entity, connector, table_names
                )
                if not indices:
                    raise ValueError(
                        f"Entity '{explain_entity}' not found in the dataset for this query."
                    )

                _anchor = _forecast_anchor(
                    pql, connector, time_columns, table_names=table_names
                )

                def _explain_call(num_neighbors: list[int] | None) -> Any:
                    kw: dict[str, Any] = {"explain": True, "run_mode": "fast"}
                    if num_neighbors is not None:
                        kw["num_neighbors"] = num_neighbors
                    if _anchor is not None:
                        kw["anchor_time"] = _anchor
                    return kumo_model.predict(pql, indices=indices, **kw)

                expl = _explain_resilient(_explain_call)
                result.num_entities = 1
                result.explanation = getattr(expl, "summary", None)
                result.explanation_warning = getattr(expl, "warning", None)
                result.explanation_details = getattr(expl, "details", None)
                prediction = getattr(expl, "prediction", None)
                if prediction is not None and hasattr(prediction, "columns"):
                    if _anchor is not None and not any(
                        str(column).casefold() == "anchor_timestamp"
                        for column in prediction.columns
                    ):
                        prediction = prediction.copy()
                        prediction["ANCHOR_TIMESTAMP"] = _anchor
                    result.columns = list(prediction.columns)
                    result.rows = prediction.head(max_preview_rows).to_dict("records")
                    if persist_table and mirror_path and persist_lock is not None:
                        _persist_full_prediction(
                            prediction,
                            pql=pql,
                            mirror_path=mirror_path,
                            table=persist_table,
                            lock=persist_lock,
                            result=result,
                        )
            else:
                forecast = _is_forecast(pql)
                whole_population = group_by or bool(persist_table)
                entity_cap = (
                    max(max_entities, _GROUP_BY_MAX_ENTITIES)
                    if whole_population
                    else max_entities
                )
                scope_sql = entity_sql
                if group_by and entity_sql:
                    scope_sql = None
                    if re.search(r"\b(WHERE|LIMIT)\b", entity_sql, re.IGNORECASE):
                        result.note = (
                            "A grouped breakdown scores every entity and aggregates by "
                            f"'{group_by}', so the planner's entity filter was not applied — these totals "
                            "cover the whole population. Ask without grouping to filter a sub-population."
                        )
                    result.entity_sql = None
                indices = _resolve_indices(
                    pql, scope_sql, connector, entity_cap, table_names
                )
                if forecast:
                    if len(indices) > 1:
                        parsed = parse_entity(pql)
                        entity_table = parsed[0] if parsed else "entity"
                        result.note = (
                            f"LIMITATION: KumoRFM forecasts a multi-period time series for ONE {entity_table} at "
                            f"a time, so a month-by-month forecast for ALL {entity_table}s in one run is not "
                            f"supported. It supports EITHER (a) a multi-horizon forecast for a single "
                            f"{entity_table}, OR (b) a single-horizon prediction across all {entity_table}s. "
                            f"{len(indices)} {entity_table}s matched; shown below is the multi-period forecast "
                            f"for just one ({indices[0]}). To cover the whole population, run a single-horizon "
                            f"prediction across all {entity_table}s first, then a multi-horizon forecast for the "
                            f"few {entity_table}s of interest (one at a time). Tell the user both options and ask "
                            f"which they want — do NOT imply the all-entity month-by-month result was produced."
                        )
                    indices = indices[:1]

                _anchor = _forecast_anchor(
                    pql, connector, time_columns, table_names=table_names
                )

                def _predict_call(
                    idx: list[Any] | None, num_neighbors: list[int] | None
                ) -> Any:
                    kw: dict[str, Any] = {}
                    if num_neighbors is not None:
                        kw["num_neighbors"] = num_neighbors
                    if _anchor is not None:
                        kw["anchor_time"] = _anchor
                    return kumo_model.predict(pql, indices=idx or None, **kw)

                raw = _predict_in_batches(
                    indices,
                    _predict_call,
                    device_assert_terminal=_is_existence_count_pql(pql),
                )
                if group_by:
                    if forecast or _LIST_DISTINCT.search(pql):
                        raise PqlGroupByError(
                            "group_by is only supported for per-entity regression/binary predictions, not "
                            "forecasts or link prediction."
                        )
                    entity = parse_entity(pql)
                    if entity is None:
                        raise PqlGroupByError(
                            "Could not determine the prediction entity to group from the PQL."
                        )
                    grouped = aggregate_prediction_by(
                        raw,
                        table=entity[0],
                        pk=entity[1],
                        group_by=group_by,
                        connector=connector,
                        table_names=table_names,
                    )
                    result.group_by = group_by
                    result.num_entities = len(raw)
                    result.truncated = len(indices) >= entity_cap
                    result.columns = list(grouped.columns)
                    result.rows = grouped.to_dict("records")
                else:
                    prediction = (
                        _order_forecast(raw) if forecast else _rank_prediction(raw)
                    )
                    result.num_entities = len(indices)
                    result.truncated = not forecast and len(indices) >= entity_cap
                    result.columns = list(prediction.columns)
                    result.rows = prediction.head(max_preview_rows).to_dict("records")
                    if (
                        persist_table
                        and mirror_path
                        and persist_lock is not None
                        and not forecast
                    ):
                        _persist_full_prediction(
                            prediction,
                            pql=pql,
                            mirror_path=mirror_path,
                            table=persist_table,
                            lock=persist_lock,
                            result=result,
                        )
            result.success = True
            result.error = None
            # Success-cache omitted with the RAG port (no vector store to write back to).
            return result
        except PqlGroupByError as exc:
            # A bad group-by (unknown column / un-aggregatable target) is a usage error, not a query the repair
            # loop can fix — surface it immediately with the actionable message.
            result.error = str(exc)
            logger.info("Grouped-prediction request rejected: %s", str(exc)[:160])
            break
        except Exception as exc:  # noqa: BLE001 - error feeds the repair loop
            prev_pql, prev_error = pql, str(exc)
            result.error = prev_error
            logger.info(
                "PQL attempt %d/%d failed: %s", attempt, max_tries, prev_error[:160]
            )
            if _is_unsupported_shape_error(prev_error):
                if _is_existence_count_pql(pql):
                    result.error = _friendly_unsupported_message(pql, prev_error)
                else:
                    result.error = (
                        "KumoRFM hit an intermittent backend error while computing this prediction "
                        "and it persisted across retries. Please try the question again."
                    )
                logger.info("Device-side assert on a valid query; not regenerating.")
                break
            # A context/GPU-capacity error on a valid query survived the neighbourhood backoff: regenerating
            # the PQL cannot help (the data/graph is what's too big), so stop and report rather than burn
            # the remaining attempts re-hitting the same server limit.
            if _is_context_capacity_error(prev_error):
                logger.info(
                    "Persistent context/GPU-capacity error on a valid query; not regenerating."
                )
                break
            if _is_empty_context_error(prev_error):
                result.error = _friendly_empty_context_message()
                logger.info(
                    "Empty entity set (no context examples) on a valid query; not regenerating."
                )
                break

    return result
