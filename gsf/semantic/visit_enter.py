"""Per-table taxonomy compilation: FK detection, Term + ColumnAttributes."""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator

from sqlglot import exp

from gsf.connectors import get_connectors
from gsf.dal.attributes import merge_column_attribute
from gsf.dal.datasources import (
    store_column_date_formats,
    store_column_sample_values,
    store_column_uniqueness,
)
from gsf.semantic.date_format import infer_date_format, is_date_type
from gsf.dal.terms import fetch_terms_and_attributes_for_table, merge_term
from gsf.semantic.date_format import infer_date_format, is_date_type
from gsf.semantic.deterministic import column_attribute_specs
from gsf.semantic.domain import DomainSummary
from gsf.semantic.embed import _MAX_EMBEDDED_JSON_SAMPLE_LEN, SemanticEmbedder
from gsf.semantic.fk_suggester import suggest_potential_foreign_keys
from gsf.semantic.models import ColumnAttributeSpec, ProcessTableResult
from gsf.semantic.term_extractor import apply_display_names_to_specs, extract_term

if TYPE_CHECKING:
    from nemo_retriever.tabular_data.sql_database import SQLDatabase

logger = logging.getLogger(__name__)

# Row cap for the per-column profiling sample.
_PROFILING_SAMPLE_LIMIT = 1000
# Most-common values kept per column.
_PROFILING_TOP_N = 5
# String sample values longer than this are not persisted.
_MAX_SAMPLE_VALUE_LEN = 30
# Declared data-type substrings whose sample values are not persisted.
_EXCLUDED_SAMPLE_TYPES = ("date", "time", "timestamp", "datetime", "uuid")
# A text column with at most this many distinct values is treated as
# categorical: we capture its full distinct value set (via a DISTINCT probe)
# instead of only the most-common values from the first-N-row sample. This
# ensures rare-but-meaningful enum values (e.g. 'Banned', 'Restricted') land in
# the embedded description even when the dominant value fills the row prefix.
_LOW_CARDINALITY_MAX = 25
# Declared data-type substrings treated as free/categorical text.
_TEXT_SAMPLE_TYPES = ("char", "text", "string", "clob", "enum")
# JSONB nested-key names that mark a sibling value as unit-qualified (e.g.
# {"value": 45000, "unit": "annual"}). When a container has one of these,
# every other key in that container is annotated with the unit key's path so
# the SQL generator sees, right next to the value, that it must not compare
# two such values without checking the unit matches.
_UNIT_MARKER_KEYS = ("unit", "units", "uom", "currency", "measure")
# JSONB nested-key name substrings that plausibly hold a date/timestamp
# string. Gates which nested keys are worth pulling example values for below
# — cheap and imprecise by design (a name-based heuristic, not a value-based
# one), since the alternative (capturing examples for every nested key) would
# bloat every JSONB column's description with mostly-irrelevant values.
_DATE_KEY_HINTS = ("date", "_dt", "day", "time", "moment", "schedule", "_on", "_at")

# Shape patterns for free-text values that hold a date/timestamp as a string
# (a JSONB nested leaf, or a "text"-typed column that was never cast to a
# real date type — both are common in these benchmark schemas). Matched
# against the *shape* only (digit widths / separators), never asserting a
# day/month reading for ambiguous slash-separated forms — that can't be
# recovered from the shape alone and a wrong guess would be worse than no
# guess. Order doesn't matter: a value is tested against every pattern and
# can only ever match one (they're mutually exclusive shapes).
_DATE_SHAPE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("YYYY-MM-DD", re.compile(r"^\d{4}-\d{2}-\d{2}")),
    ("YYYY/MM/DD", re.compile(r"^\d{4}/\d{1,2}/\d{1,2}$")),
    ("DD/MM/YYYY or MM/DD/YYYY (ambiguous)", re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")),
    ("DD/MM/YY or MM/DD/YY (ambiguous)", re.compile(r"^\d{2}/\d{2}/\d{2}$")),
    ("Mon DD, YYYY", re.compile(r"^[A-Za-z]{3,9} \d{1,2},? \d{4}$")),
)

# Tables are processed in parallel (ThreadPoolExecutor in pipeline.py), but
# the commit phase must be serial: VDB search → judge → Neo4j merge → VDB embed.
# Without the lock, two threads could simultaneously propose the same Term,
# both find zero VDB hits (the first hasn't embedded yet), and create duplicates.
_term_commit_lock = threading.Lock()


@contextmanager
def _step(table_name: str, description: str) -> Iterator[None]:
    """Log one compilation step for a table, with how long it took.

    Tables compile in parallel, so every line carries the table name — the
    ``[table] step…`` / ``[table] step done`` pairing is what makes interleaved
    output readable. Most steps are LLM or warehouse round-trips, so the elapsed
    time is the useful part when compilation feels slow.
    """
    logger.info("[%s] %s…", table_name, description)
    started = time.monotonic()
    try:
        yield
    finally:
        logger.info(
            "[%s] %s done in %.1fs", table_name, description, time.monotonic() - started
        )


def _terms_with_assignments(
    term_result: Any,
    spec_by_column: dict[str, ColumnAttributeSpec],
) -> list[tuple[Any, list[Any]]]:
    """Terms that have at least one resolvable column attribute."""
    persisted = []
    for term in term_result.terms:
        assignments = [a for a in term.attributes if a.source_column in spec_by_column]
        if assignments:
            persisted.append((term, assignments))
    return persisted


def _resolve_connector(database_name: str | None) -> "SQLDatabase | None":
    """Return the loaded connector whose ``database_name`` matches, or None."""
    if not database_name:
        return None
    key = database_name.casefold()
    for connector in get_connectors():
        db = getattr(connector, "database_name", None)
        if db is not None and db.casefold() == key:
            return connector
    return None


def _is_excluded_sample_type(data_type: str | None) -> bool:
    """Whether a column's declared type disqualifies it from sample storage."""
    if not data_type:
        return False
    lowered = data_type.lower()
    return any(token in lowered for token in _EXCLUDED_SAMPLE_TYPES)


def _is_text_sample_type(data_type: str | None) -> bool:
    """Whether a column's declared type is free/categorical text."""
    if not data_type:
        return False
    lowered = data_type.lower()
    return any(token in lowered for token in _TEXT_SAMPLE_TYPES)


def _date_shapes_seen(values: list[str]) -> list[str]:
    """Distinct recognized date/timestamp shapes among *values*, in first-seen order.

    A value that matches none of :data:`_DATE_SHAPE_PATTERNS` is silently
    ignored rather than counted as "no date shape" for the whole column —
    this only exists to catch a column mixing more than one *recognized*
    shape (e.g. some rows ``YYYY-MM-DD``, others ``YYYY/MM/DD``), which is a
    real, observed bug: a fixed-format parser silently mis-parses whichever
    shape it wasn't written for. It intentionally does not try to classify
    every possible date-like string — an odd one-off format (e.g. a literal
    "12th Jun." style column) is left to speak for itself via the raw sample
    values already shown, rather than force a wrong or overly-broad match.
    """
    shapes: list[str] = []
    for value in values:
        for label, pattern in _DATE_SHAPE_PATTERNS:
            if pattern.match(value):
                if label not in shapes:
                    shapes.append(label)
                break
    return shapes


def _quoted_identifier(name: str, dialect: str | None) -> str:
    """Quote a schema, table or column name for *dialect*.

    Names carrying a space or a reserved word have to be quoted or the probe
    below silently loses the table: ``SELECT * FROM main.Sales Orders`` parses
    as table ``main.Sales``, raises "no such table", and the caller drops every
    column of that table from profiling. The quote character is dialect-specific
    (backticks on MySQL and Spark, double quotes elsewhere), so the naive
    hard-coded ``"`` would trade a SQLite bug for a MySQL one.
    """
    try:
        return exp.to_identifier(name, quoted=True).sql(dialect=dialect or None)
    except Exception:
        # Unknown dialect: fall back to the SQL-standard quote rather than
        # emitting a bare identifier, since bare is what breaks on spaces.
        return '"' + name.replace('"', '""') + '"'


def _distinct_values_if_low_cardinality(
    connector: "SQLDatabase",
    qualified: str,
    col_name: str,
    cap: int,
) -> list[str] | None:
    """Return the full distinct value set for a low-cardinality column.

    Runs ``SELECT DISTINCT <col> ... LIMIT cap + 1``. Returns the distinct
    values (as strings) when the column has at most *cap* distinct non-null
    values; returns ``None`` for high-cardinality columns (more than *cap*
    distinct values) or on any error, so the caller falls back to the
    most-common-values behaviour. The ``LIMIT`` keeps the probe cheap even on
    huge, high-cardinality columns (the scan stops after cap + 1 distinct rows).
    """
    quoted = _quoted_identifier(col_name, getattr(connector, "dialect", None))
    try:
        df = connector.execute(
            f"SELECT DISTINCT {quoted} FROM {qualified} "
            f"WHERE {quoted} IS NOT NULL LIMIT {cap + 1}"
        )
    except Exception:
        return None
    if df is None or df.empty:
        return None
    values = [str(v) for v in df.iloc[:, 0].tolist()]
    if len(values) > cap:
        return None
    return values


def calculate_columns_profiling(
    table: dict[str, Any],
    columns: list[dict[str, Any]],
    connector: "SQLDatabase",
) -> dict[str, dict[str, Any]]:
    """Profile a table's columns from a live sample of up to 1000 rows.

    Runs ``SELECT * ... LIMIT 1000`` and, for every column, computes an
    ``is_unique`` flag (all non-null values distinct) and the 5 most-common
    values. For non-unique text columns whose sample yields fewer than 5
    distinct values, runs a ``SELECT DISTINCT`` probe to capture rare enum
    values that the row prefix may have missed.

    Persists to Neo4j Column nodes: ``is_unique`` for every column,
    ``format`` for date/time-typed or text columns whose sampled values share
    one storage notation — resolved by validating every value against
    strptime candidates (see ``gsf.semantic.date_format``), so it can settle
    genuinely ambiguous-looking shapes (``13/07/2011`` can only be day-first)
    rather than merely flag them — and ``sample_values`` for every column
    except those whose declared type is a date/time/uuid (individual string
    values longer than 30 chars are dropped). A text column whose real values
    mix more than one recognized date/timestamp shape (e.g. some rows
    ``YYYY-MM-DD``, others ``YYYY/MM/DD``) gets an explicit mixed-format
    warning appended, checked against the full sample rather than only the
    stored top-5 so a rare minority shape isn't missed. JSONB nested keys
    whose name hints at a date (see ``_DATE_KEY_HINTS``) get the same
    treatment plus a raw example value when no recognized shape matches at
    all.

    Returns ``{column_name: {"sample_values": [top-5 values], "is_unique":
    bool, "format": str | None}}`` for *all* columns (values unfiltered —
    includes dates, uuids and long strings).
    """
    schema_name = table.get("schema_name")
    table_name = table["name"]
    dialect = getattr(connector, "dialect", None)
    quoted_table = _quoted_identifier(table_name, dialect)
    qualified = (
        f"{_quoted_identifier(schema_name, dialect)}.{quoted_table}"
        if schema_name
        else quoted_table
    )

    try:
        df = connector.execute(
            f"SELECT * FROM {qualified} LIMIT {_PROFILING_SAMPLE_LIMIT}"
        )
    except Exception:
        logger.warning(
            "[%s] column profiling query failed — skipping", table_name, exc_info=True
        )
        return {}

    if df is None or df.empty:
        return {}

    type_by_column = {
        col.get("name"): col.get("data_type") for col in columns if col.get("name")
    }

    profiling: dict[str, dict[str, Any]] = {}
    sample_values: dict[str, list] = {}
    uniqueness: dict[str, bool] = {}
    date_formats: dict[str, str] = {}

    for column in df.columns:
        col_name = str(column)
        try:
            # Cast to string first: some columns hold unhashable values (e.g.
            # Postgres array columns come back as Python lists, JSON/JSONB as
            # dict/list), and both is_unique and value_counts hash values.
            series = df[column].dropna().map(str)

            is_unique = bool(len(series) > 0 and series.is_unique)
            top5 = list(series.value_counts().head(_PROFILING_TOP_N).index)
        except Exception:
            logger.warning(
                "[%s] profiling failed for column %r — skipping column",
                table_name,
                col_name,
                exc_info=True,
            )
            continue

        declared_type = type_by_column.get(col_name)

        # For categorical text columns that did not yield a full top-N set from
        # the first-N-row sample, prefer the full distinct value set. Rare enum
        # values (e.g. 'Banned') otherwise never make it into the embedded
        # description when a dominant value fills the sampled row prefix.
        # Skip the DISTINCT probe when the sample already produced _PROFILING_TOP_N
        # values — that is enough for embedding and avoids a full-table scan
        # per column on warehouses where DISTINCT + LIMIT does not early-stop.
        col_values = top5
        if (
            not is_unique
            and len(top5) < _PROFILING_TOP_N
            and _is_text_sample_type(declared_type)
        ):
            distinct_vals = _distinct_values_if_low_cardinality(
                connector, qualified, col_name, _LOW_CARDINALITY_MAX
            )
            if distinct_vals is not None:
                merged = list(top5)
                for value in distinct_vals:
                    if value not in merged:
                        merged.append(value)
                col_values = merged

        # SQLite (and other loosely-typed sources) declare dates as TEXT, so the
        # declared type alone misses them: infer from the values as well, which
        # only yields a format when every sampled value shares one notation.
        # Unlike main's port, this does NOT suppress sample_values below just
        # because a format was found — samples stay available to FK inference
        # (semantic_fk.py's SQL-probe fallback reads persisted sample_values),
        # and date values are short enough that keeping them costs little.
        date_format = (
            infer_date_format(series)
            if is_date_type(declared_type) or _is_text_sample_type(declared_type)
            else None
        )
        if date_format:
            date_formats[col_name] = date_format

        uniqueness[col_name] = is_unique
        profiling[col_name] = {
            "sample_values": col_values,
            "is_unique": is_unique,
            "format": date_format,
        }

        if _is_excluded_sample_type(declared_type):
            continue

        # JSONB columns: the raw value is a Python dict and its string
        # representation always exceeds _MAX_SAMPLE_VALUE_LEN, so the normal
        # filter would silently discard everything.  Instead, extract the
        # unique key names from the sampled rows — flat keys at the top level
        # and one level of nesting — so the SQL generator sees the actual
        # field paths it needs to write correct ->/->> expressions.
        if "json" in (declared_type or "").lower():
            top_level_keys: list[str] = []
            containers: dict[str, list[str]] = {}
            # Example raw values per (container, nested_key) — only collected
            # for keys whose name hints at a date/timestamp (see
            # _DATE_KEY_HINTS); every other nested key stays name-only, same
            # as before. These are never a Column's real "sample_values" (the
            # column itself is JSONB, not text) — they exist solely to let a
            # date-shape check run on a nested leaf the way it already can on
            # a plain text column.
            nested_date_examples: dict[tuple[str, str], list[str]] = {}
            # Example raw values per top-level flat key (e.g. 'Tx_Adh' ->
            # ['High', 'Medium', 'Compliant']). Unlike nested_date_examples
            # above, this is collected for every flat key, not just
            # date-hinted ones — a bare key name like 'Tx_Adh' gives the SQL
            # generator no way to tell it apart from a same-purpose flat
            # column (e.g. 'med_adh') that already shows its real sample
            # values; showing the actual values here closes that gap.
            top_level_examples: dict[str, list[str]] = {}
            for raw_val in df[column].dropna():
                if not isinstance(raw_val, dict):
                    continue
                for k, v in raw_val.items():
                    if isinstance(v, dict):
                        nested = containers.setdefault(k, [])
                        for nested_k, nested_v in v.items():
                            if nested_k not in nested:
                                nested.append(nested_k)
                            is_date_hint = any(
                                hint in nested_k.lower() for hint in _DATE_KEY_HINTS
                            )
                            if nested_v is None or not is_date_hint:
                                continue
                            examples = nested_date_examples.setdefault(
                                (k, nested_k), []
                            )
                            str_v = str(nested_v)
                            if (
                                len(examples) < 3
                                and str_v not in examples
                                and (len(str_v) <= _MAX_SAMPLE_VALUE_LEN)
                            ):
                                examples.append(str_v)
                    else:
                        if k not in top_level_keys:
                            top_level_keys.append(k)
                        if v is None:
                            continue
                        examples = top_level_examples.setdefault(k, [])
                        str_v = str(v)
                        if (
                            len(examples) < _PROFILING_TOP_N
                            and str_v not in examples
                            and (len(str_v) <= _MAX_SAMPLE_VALUE_LEN)
                        ):
                            examples.append(str_v)

            # embed.py's ColumnAttribute embedding text (_format_sample_values)
            # drops any individual sample_values entry over
            # _MAX_EMBEDDED_JSON_SAMPLE_LEN chars WHOLE, not truncated — so an
            # over-length annotation risks losing the entry, including the
            # bare key name, from retrieval entirely (this was already true
            # pre-existing for the WARNING/date-format annotations below,
            # which had no fallback at all). ``_safe_annotate`` keeps
            # appending as much annotation as fits, falling back a step at a
            # time rather than an all-or-nothing choice, so a key's own name
            # is never lost just because a longer example/format string
            # didn't fit next to it. The SQL-gen prompt (sql_from_semantic.py)
            # applies no such filter — it always sees the full, un-capped
            # entry — so this only trims what retrieval-time embedding sees.
            # Note this budgets the *formatted* entry (brackets/quotes
            # included) against _MAX_EMBEDDED_JSON_SAMPLE_LEN (the JSON-only,
            # 60-char budget), not the raw-value cutoff (_MAX_SAMPLE_VALUE_LEN,
            # 30 chars) used above when deciding whether a value was worth
            # collecting as an example candidate at all — those are two
            # different questions with two different budgets.
            def _safe_annotate(base: str, *annotations: str) -> str:
                entry = base
                for ann in annotations:
                    candidate = f"{entry} {ann}"
                    if len(candidate) <= _MAX_EMBEDDED_JSON_SAMPLE_LEN:
                        entry = candidate
                    else:
                        break
                return entry

            json_keys: list[str] = []
            for k, nested_keys in containers.items():
                unit_key = next(
                    (nk for nk in nested_keys if nk.lower() in _UNIT_MARKER_KEYS),
                    None,
                )
                for nested_k in nested_keys:
                    base = f"{k}.{nested_k}"
                    unit_ann = (
                        f"[unit: {k}.{unit_key}]"
                        if unit_key and nested_k != unit_key
                        else None
                    )
                    examples = nested_date_examples.get((k, nested_k))
                    date_ann = None
                    if examples:
                        shapes = _date_shapes_seen(examples)
                        if len(shapes) > 1:
                            # Try every observed shape first (this is the one
                            # place fewer items changes the *meaning*, not
                            # just the detail level — dropping a shape here
                            # could hide a real mixed-format bug), then fall
                            # back to just flagging that shapes are mixed
                            # without enumerating them, before giving up the
                            # annotation entirely.
                            full = (
                                "[WARNING mixed date formats observed: "
                                f"{', '.join(shapes)}]"
                            )
                            short = "[WARNING mixed date formats observed]"
                            date_ann = (
                                full
                                if len(f"{base} {full}") <= _MAX_EMBEDDED_JSON_SAMPLE_LEN
                                else short
                            )
                        elif shapes:
                            date_ann = f"[date format: {shapes[0]}]"
                        else:
                            date_ann = f"[e.g. {examples[0]!r}]"
                    anns = [a for a in (unit_ann, date_ann) if a]
                    json_keys.append(_safe_annotate(base, *anns))
            for k in top_level_keys:
                examples = top_level_examples.get(k)
                if not examples:
                    json_keys.append(k)
                    continue
                # Greedily fit as many distinct example values as the length
                # budget allows (1 up to len(examples), all collected up to
                # _PROFILING_TOP_N) instead of a fixed count — a short key
                # with short values (e.g. 'Tx_Adh': 'High'/'Low'/'Compliant')
                # gets more signal than a fixed cap of 1 would give it, while
                # a long key/value pair still degrades gracefully to fewer
                # examples, then to the bare key, rather than being dropped.
                best = k
                for n in range(1, len(examples) + 1):
                    candidate = f"{k} [e.g. {', '.join(repr(e) for e in examples[:n])}]"
                    if len(candidate) <= _MAX_EMBEDDED_JSON_SAMPLE_LEN:
                        best = candidate
                    else:
                        break
                json_keys.append(best)

            if json_keys:
                # Capped well above the typical real key count (see
                # mental_health.treatmentoutcomes.txprogmet, which has 11) so
                # a wide-but-not-huge JSONB column doesn't lose keys to a
                # cap tuned for a narrower column shape seen elsewhere.
                sample_values[col_name] = json_keys[:20]
            continue

        filtered = [v for v in col_values if len(v) <= _MAX_SAMPLE_VALUE_LEN]
        if _is_text_sample_type(declared_type):
            # Checked against the full up-to-1000-row sample, not just the
            # top-5/distinct values kept for display — a minority format can
            # be entirely absent from the top-5 while still breaking a
            # fixed-format parser on real rows (observed: a column mostly
            # 'YYYY-MM-DD' with a rare 'YYYY/MM/DD' minority).
            shapes = _date_shapes_seen(list(series))
            if len(shapes) > 1:
                filtered.append(
                    f"[WARNING mixed date formats observed: {', '.join(shapes)}]"
                )
        if filtered:
            sample_values[col_name] = filtered

    table_id = table["id"]
    store_column_sample_values(table_id, sample_values)
    store_column_uniqueness(table_id, uniqueness)
    store_column_date_formats(table_id, date_formats)

    return profiling


def process_table(
    table: dict[str, Any],
    ctx: dict[str, Any],
    *,
    domain_summary: DomainSummary | None,
    embedder: SemanticEmbedder | None = None,
    database_name: str | None = None,
) -> ProcessTableResult:
    """Build taxonomy nodes for one table: Term and ColumnAttributes."""
    table_id = table["id"]
    table_name = table["name"]

    # Columns profiling — requires a live connector; skipped when unavailable.
    # Persists sample_values, is_unique, and format onto Column nodes, and maps
    # each column to {"sample_values": [...], "is_unique": bool, "format": str
    # | None} for FK detection below.
    connector = _resolve_connector(database_name)
    columns_profiling_samples: dict[str, dict[str, Any]] = {}
    if connector is not None:
        column_count = len(ctx.get("columns", []))
        try:
            with _step(table_name, f"Sampling column values ({column_count} columns)"):
                columns_profiling_samples = calculate_columns_profiling(
                    table, ctx.get("columns", []), connector
                )
        except Exception:
            logger.warning(
                "[%s] column profiling failed — continuing without it",
                table_name,
                exc_info=True,
            )
    else:
        logger.info(
            "[%s] Skipping value sampling — no live connector for %r",
            table_name,
            database_name,
        )

    # --- FK detection (LLM + declared); results not written to Neo4j ---
    declared_fks = ctx.get("fks", [])
    with _step(table_name, "Detecting foreign keys"):
        fk_suggestions = suggest_potential_foreign_keys(
            table, ctx, columns_profiling_samples
        )
    suggested_fk_names = {s.column_name for s in fk_suggestions.suggestions}
    declared_fk_names = {
        fk["source_column"] for fk in declared_fks if fk.get("source_column")
    }
    all_fk_names = declared_fk_names | suggested_fk_names

    # --- Build attribute specs for non-FK columns ---
    specs = column_attribute_specs(
        ctx.get("columns", []),
        declared_fks,
        suggested_fk_columns=all_fk_names,
        columns_profiling_samples=columns_profiling_samples,
    )
    if not specs:
        logger.warning("[%s] no non-FK columns — skipping Term creation", table_name)
        return ProcessTableResult()

    # --- LLM: propose Term(s) and display names ---
    with _step(table_name, f"Generating terms and descriptions ({len(specs)} columns)"):
        term_result = extract_term(table, ctx, specs, domain_summary=domain_summary)
    apply_display_names_to_specs(term_result, specs)
    spec_by_column = {spec.source_column: spec for spec in specs}
    persisted_terms = _terms_with_assignments(term_result, spec_by_column)

    if not persisted_terms:
        logger.warning(
            "[%s] LLM assigned no columns to any Term (%d candidates)",
            table_name,
            len(specs),
        )
        return ProcessTableResult()

    # Serialize: dedup check + Neo4j writes + VDB embed must be atomic
    # so the next thread's VDB search sees this thread's newly embedded terms.
    result_term_names: list[str] = []
    result_attr_names: list[str] = []
    terms: list = []
    attrs_by_term: dict[str, list[dict]] = defaultdict(list)

    with _term_commit_lock:
        if embedder is not None:
            with _step(
                table_name,
                f"Checking {len(persisted_terms)} proposed term(s) for duplicates",
            ):
                _dedupe_terms(table_name, persisted_terms, embedder)

        with _step(table_name, f"Writing {len(persisted_terms)} term(s) to the graph"):
            _commit_terms(
                persisted_terms,
                spec_by_column,
                table_id,
                result_term_names,
                result_attr_names,
            )

        logger.info(
            "[%s] → Terms %s (%d attrs, %d suspected FKs)",
            table_name,
            result_term_names,
            len(result_attr_names),
            len(all_fk_names),
        )

        # Fetch persisted terms — used for embedding.
        try:
            terms, attrs = fetch_terms_and_attributes_for_table(table_id)
            for attr in attrs:
                if attr.get("term_name"):
                    attrs_by_term[attr["term_name"]].append(attr)
        except Exception:
            logger.warning("[%s] failed to fetch persisted terms", table_name)

        if embedder is not None and terms:
            try:
                with _step(table_name, f"Embedding {len(terms)} term(s)"):
                    for term in terms:
                        embedder.embed_term(term, attrs_by_term.get(term["name"], []))
            except Exception:
                logger.warning("[%s] inline embed failed", table_name)

    return ProcessTableResult(
        term_names=result_term_names,
        attr_names=result_attr_names,
    )


def _dedupe_terms(
    table_name: str,
    persisted_terms: list,
    embedder: SemanticEmbedder,
) -> None:
    """Rewrite proposed Term names onto existing ones the judge deems equivalent."""
    for term, _ in persisted_terms:
        try:
            candidates = embedder.search_similar_terms(term.name, term.description)
            if not candidates:
                continue

            from gsf.semantic.term_judge import judge_term_overlap

            merge_into = judge_term_overlap(term.name, term.description, candidates)
            if merge_into:
                logger.info(
                    "[%s] Merging proposed Term %r into existing %r",
                    table_name,
                    term.name,
                    merge_into,
                )
                term.name = merge_into
        except Exception:
            logger.warning(
                "[%s] Term dedup check failed for %r — proceeding as-is",
                table_name,
                term.name,
                exc_info=True,
            )


def _commit_terms(
    persisted_terms: list,
    spec_by_column: dict[str, ColumnAttributeSpec],
    table_id: str,
    result_term_names: list[str],
    result_attr_names: list[str],
) -> None:
    """Merge Terms and their ColumnAttributes into Neo4j."""
    for term, assignments in persisted_terms:
        merge_term(term.name, term.description, table_id, synonyms=term.synonyms)
        result_term_names.append(term.name)
        for assignment in assignments:
            spec = spec_by_column[assignment.source_column]
            merge_column_attribute(
                term_name=term.name,
                table_id=table_id,
                source_column=spec.source_column,
                attr_name=spec.display_name,
                datatype=spec.datatype,
                description=spec.description,
            )
            result_attr_names.append(spec.display_name)
