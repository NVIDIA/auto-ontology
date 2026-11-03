# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tools the SQL construction agent may call.

Every one of these already exists in the pipeline, but only *after* the model
has committed to a query: the probe agents, the static validators and the
reconstruction loop all run downstream of construction and feed their findings
back through a second LLM call. The toolbox hands the same checks to the model
*while* it is still writing, so a bad literal or a fabricated join costs a tool
call instead of a reconstruction round.

Three rules hold for all of them:

- They never raise. A tool that fails returns a string saying so, because an
  exception here aborts the turn loop and loses a question.
- They return rendered text, not objects. The only consumer is a language
  model, and the renderers here are what bound how much of a result can reach
  the context window.
- They are read-only. ``run_sql`` goes through :class:`ProbeExecutor`, which
  rejects anything that is not a single bounded ``SELECT``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import sqlglot
from sqlglot import exp

from auto_ontology.catalog.sql_parse import parse_query_single
from auto_ontology.dal.attributes import find_join_path
from auto_ontology.dal.datasources import (
    fetch_tables_by_ids,
    find_column_id_by_table_and_name,
)
from auto_ontology.retrieval.data_access.graph_schemas import (
    fetch_all_schema_ids,
    get_schemas_by_ids,
)
from auto_ontology.retrieval.data_access.relevant_tables import (
    dedupe_merge_relevant_tables,
    get_relevant_tables,
)
from auto_ontology.retrieval.text_to_sql.agents.sql_parse_validation import (
    detect_degenerate_sql,
    detect_missing_aggregation,
    detect_vacuous_group_by,
    quote_known_mixed_case_identifiers,
)
from auto_ontology.retrieval.text_to_sql.connector_routing import (
    resolve_connector_from_tables,
)
from auto_ontology.retrieval.text_to_sql.db_probe.config import DB_PROBE_TIMEOUT_S
from auto_ontology.retrieval.text_to_sql.db_probe.executor import ProbeExecutor
from auto_ontology.retrieval.text_to_sql.db_probe.join_path_check import (
    build_case_dirty_join_repair_error,
    build_join_path_repair_error,
    find_case_dirty_join_mismatches,
    find_join_path_mismatches,
)
from auto_ontology.retrieval.text_to_sql.db_probe.jsonb_path_check import (
    build_jsonb_path_repair_error,
    find_jsonb_path_mismatches,
)
from auto_ontology.retrieval.text_to_sql.db_probe.literal_check import (
    build_value_repair_error,
    find_integer_division_mismatches,
    find_literal_mismatches,
    find_numeric_scale_mismatches,
)
from auto_ontology.retrieval.text_to_sql.formatters_util import (
    format_tables_for_prompt,
    qualify_table,
)
from auto_ontology.retrieval.text_to_sql.sql_agent import config, elaboration
from auto_ontology.retrieval.text_to_sql.state import (
    AgentState,
    get_original_question,
)

logger = logging.getLogger(__name__)

# How many tables one search_tables call may add. The VDB is happy to return
# more, but each one costs the model a full schema block to read, and a
# construction that needs five new tables is a retrieval failure the agent
# will not rescue by reading harder.
_MAX_DISCOVERED_TABLES = 4

_SEARCH_K = 3

# Fan-out probes a question may spend. The check costs one execution on top of
# the query it describes, and a question that has already been told six times
# whether its joins repeat rows is not learning it on the seventh.
_MAX_FAN_OUT_CHECKS = 6


def _render_cell(value: Any, max_chars: int) -> str:
    text = "NULL" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}…"


def _render_rows(rows: list[dict[str, Any]] | None) -> str:
    """Render probe rows as a pipe-delimited table.

    Deliberately not JSON: the per-row key repetition in JSON is pure cost at
    this width, and the model only needs to read the values.
    """
    if not rows:
        return "(no rows)"
    max_cell = config.max_cell_chars()
    limit = config.max_rows_shown()
    shown = rows[:limit]
    columns = list(shown[0].keys())
    lines = [" | ".join(columns)]
    lines.append("-" * min(len(lines[0]), 120))
    for row in shown:
        lines.append(
            " | ".join(_render_cell(row.get(col), max_cell) for col in columns)
        )
    if len(rows) > limit:
        lines.append(f"… {len(rows) - limit} more row(s) not shown")
    return "\n".join(lines)


def _known_types(relevant_tables: list[dict]) -> dict[tuple[str, str], str]:
    """``{(table, column): data_type}`` from the already-fetched schema.

    Same map the probe agents build, and for the same reason: it lets the
    checks tell "wrong key" from "not that kind of column at all" without a
    round-trip.
    """
    types: dict[tuple[str, str], str] = {}
    for table in relevant_tables:
        table_name = table.get("name")
        if not table_name:
            continue
        for col in table.get("columns") or []:
            if not isinstance(col, dict):
                continue
            col_name = col.get("name")
            data_type = col.get("data_type")
            if col_name and data_type:
                types[(table_name.lower(), col_name.lower())] = data_type
    return types


#: Shortest description phrase worth reporting, in words. Two-word overlaps
#: ("the account", "of money") fire on almost every question and would train
#: the model to ignore the finding.
_WORDING_MIN_PHRASE_WORDS = 3

#: Words carrying no schema meaning, so a phrase made only of these does not
#: count as a match.
_WORDING_STOPWORDS = frozenset(
    "a an and or of the in on at to for is are was were be been by with from "
    "as that this it its their his her not no into unit".split()
)


def _uses_column(sql: str, name: str) -> bool:
    """Whether *sql* mentions the column *name*.

    A substring test rather than a token scan, so that quoted names with spaces
    ("Charter Funding Type") match as readily as bare ones. It over-collects --
    a column whose name occurs inside a longer identifier counts as used -- and
    that is the direction to err in: counting a column present when it is not
    costs one missed finding, while counting it absent costs a false alarm on
    every query the check sees.
    """
    return name.lower() in sql.lower()


def _description_overlap(description: str, question: str) -> str:
    """The longest run of description words that the question also contains.

    Word runs rather than a similarity score: "amount of money" appearing
    verbatim in both is evidence a reader can check, where a cosine distance
    over the same pair is not something the model can act on.
    """
    words = [w for w in re.findall(r"[a-z0-9]+", description.lower())]
    best = ""
    for start in range(len(words)):
        for end in range(len(words), start, -1):
            phrase = " ".join(words[start:end])
            if len(phrase) <= len(best) or phrase not in question:
                continue
            if all(w in _WORDING_STOPWORDS for w in words[start:end]):
                continue
            best = phrase
    return best


@dataclass
class SQLAgentToolbox:
    """Per-question tool context.

    One instance lives for one construction. The probe executor is shared
    across every tool so that a model which spends its budget exploring has
    less left for diagnosis — that is the intended trade, not an oversight.
    """

    connector: Any
    dialect: Optional[str]
    database_name: Optional[str]
    relevant_tables: list[dict]
    data_retriever: Any = None
    #: The question as asked. Only the elaboration checks read it — they are
    #: the only ones that care what the question did *not* ask for.
    question: str = ""
    executor: ProbeExecutor = field(init=False)
    sql_calls: int = field(default=0, init=False)
    fan_out_checks: int = field(default=0, init=False)
    discovered_tables: list[dict] = field(default_factory=list, init=False)
    #: Every tool invocation, for the trajectory log.
    calls: list[dict[str, Any]] = field(default_factory=list, init=False)
    _schemas: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.executor = ProbeExecutor(
            self.connector,
            max_calls=config.max_probe_calls(),
            timeout_s=DB_PROBE_TIMEOUT_S,
        )

    @classmethod
    def from_state(cls, state: AgentState) -> "SQLAgentToolbox":
        path_state = state.get("path_state") or {}
        relevant_tables = list(path_state.get("relevant_tables") or [])
        connector = resolve_connector_from_tables(
            relevant_tables, state.get("connectors") or []
        )
        return cls(
            connector=connector,
            dialect=getattr(connector, "dialect", None),
            database_name=path_state.get("target_db"),
            relevant_tables=relevant_tables,
            data_retriever=state.get("data_retriever"),
            question=get_original_question(state) or "",
        )

    def close(self) -> None:
        self.executor.close()

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def run_sql(self, sql: str, purpose: str = "") -> str:
        """Execute one read-only query and render its rows."""
        if self.sql_calls >= config.max_sql_calls():
            return self._record(
                "run_sql",
                {"sql": sql, "purpose": purpose},
                "Query budget exhausted. Write your final SQL now using what "
                "you have already learned.",
            )
        self.sql_calls += 1
        entry = self.executor.run(sql, purpose=purpose or "agent exploration")
        if not entry["ok"]:
            result = f"Query failed: {entry['error']}"
        else:
            header = f"{entry['row_count']} row(s)"
            if entry["truncated"]:
                header += " (truncated)"
            result = f"{header}\n{_render_rows(entry['rows'])}"
            note = self._fan_out_note(sql)
            if note:
                result += f"\n\n{note}"
        return self._record("run_sql", {"sql": sql, "purpose": purpose}, result)

    def _fan_out_note(self, sql: str) -> str:
        """Whether this query's joins repeat rows, measured on the real data.

        Deduplication is the single largest disagreement with gold in the
        benchmark -- 226 questions, split almost evenly between returning
        duplicates gold does not have and removing duplicates gold keeps. The
        split being even is why no instruction fixes it: whichever way the
        prompt leans, it is wrong about half the time. What settles it is not
        a rule but a fact about the data, and the database can state that fact
        in one round trip.

        Reported in both directions deliberately. Speaking up only when rows
        repeat would answer the under-deduplicating half and leave the
        over-deduplicating half guessing from silence.
        """
        if self.fan_out_checks >= _MAX_FAN_OUT_CHECKS:
            return ""
        # Parsed and re-rendered under the same dialect, falling back to
        # sqlglot's default when the connector names one it does not know.
        # A failure here only costs the note, so it stays quiet.
        try:
            tree = sqlglot.parse_one(sql, read=self.dialect or None)
            inner = tree.sql(dialect=self.dialect or None)
        except Exception:
            try:
                tree = sqlglot.parse_one(sql)
                inner = tree.sql()
            except Exception:
                return ""
        if tree is None or not list(tree.find_all(exp.Join)):
            return ""
        # An aggregate without GROUP BY collapses to one row whatever the join
        # did, and a query that already deduplicates has had the question
        # answered. Neither can show fan-out, so neither is worth a round trip.
        if tree.find(exp.Distinct) is not None:
            return ""
        if tree.find(exp.Group) is None and tree.find(exp.AggFunc) is not None:
            return ""

        probe = (
            f"SELECT COUNT(*) AS total_rows, "
            f"(SELECT COUNT(*) FROM (SELECT DISTINCT * FROM ({inner}))) AS distinct_rows "
            f"FROM ({inner})"
        )
        self.fan_out_checks += 1
        entry = self.executor.run(probe, purpose="check whether the join repeats rows")
        if not entry["ok"] or not entry["rows"]:
            return ""
        row = entry["rows"][0]
        try:
            total, distinct = (int(v) for v in list(row.values())[:2])
        except (TypeError, ValueError):
            return ""
        if total == 0:
            return ""
        if total == distinct:
            return (
                f"Join check: {total} row(s), all distinct. These joins do not "
                f"repeat anything, so deduplicating would change nothing and "
                f"any DISTINCT here is unnecessary."
            )
        return (
            f"Join check: {total} row(s) but only {distinct} distinct — these "
            f"joins repeat {total - distinct} row(s). If the question counts or "
            f"lists entities rather than pairings, that repetition is in your "
            f"answer and needs DISTINCT; if it asks for the rows themselves, "
            f"leave it."
        )

    def check_sql(self, sql: str) -> str:
        """Run the pipeline's static validators against a draft, without executing it.

        This is the same gate ``validate_sql_query`` applies after
        construction, so anything it rejects here would have cost a full
        reconstruction round downstream.
        """
        findings = self.static_findings(sql)
        result = "\n\n".join(findings) if findings else "Passed all static checks."
        return self._record("check_sql", {"sql": sql}, result)

    def static_findings(self, sql: str) -> list[str]:
        """Static problems with *sql*, empty when it passes.

        Split from :meth:`check_sql` so the submission gate can act on the
        findings rather than parse the prose rendered for the model.
        """
        findings: list[str] = []

        quoted = quote_known_mixed_case_identifiers(
            sql, self.relevant_tables, self.dialect
        )
        if quoted != sql:
            findings.append(
                "Mixed-case identifiers need quoting. Use this form instead:\n"
                f"{quoted}"
            )

        parse_error = self._parse_error(quoted)
        if parse_error:
            # Every check below reads the parse tree, so there is nothing
            # further to say until this is fixed.
            findings.append(f"Does not parse against the catalog: {parse_error}")
            return findings

        degenerate = detect_degenerate_sql(quoted, self.dialect)
        if degenerate:
            findings.append(
                f"Placeholder query that does not answer the question: {degenerate}"
            )

        vacuous = detect_vacuous_group_by(quoted, self.dialect, self.database_name)
        if vacuous:
            findings.append(f"Aggregation is a no-op: {vacuous}")

        missing_agg = detect_missing_aggregation(
            quoted, self.dialect, self.database_name
        )
        if missing_agg:
            findings.append(f"GROUP BY is unsafe: {missing_agg}")

        findings.extend(self._wording_findings(quoted))

        return findings

    def _wording_findings(self, sql: str) -> list[str]:
        """Columns the question names more plainly than the ones *sql* uses.

        Every other check here asks whether the query is valid against the
        database. None asks whether it answers the question that was put, which
        is where the construction actually goes wrong: a query reading
        ``trans.balance`` for "own an amount of money greater than 3000USD"
        parses, executes and returns rows, so it clears every gate while
        answering something else. ``trans.amount`` is described in the catalog
        as "amount of money, Unit：USD" -- the question's own words.

        Matched on description text rather than on the semantic layer's curated
        entity names, because the names are what missed: the layer knew that
        column as "Balance After Transaction" and had no label resembling
        "amount of money" to match the question against.

        Reported as a prompt to re-read, never as a verdict. A question often
        does mean the sibling column, and the gate must not push a correct
        query off a column it chose deliberately.
        """
        question = (self.question or "").lower()
        if not question:
            return []

        candidates: list[tuple[int, str, str, str, str]] = []
        for table in self.relevant_tables:
            table_name = table.get("name") or "?"
            for column in table.get("columns") or []:
                if not isinstance(column, dict):
                    continue
                name = (column.get("name") or "").strip()
                description = (column.get("description") or "").strip()
                if not name or not description or _uses_column(sql, name):
                    continue
                overlap = _description_overlap(description, question)
                if len(overlap.split()) >= _WORDING_MIN_PHRASE_WORDS:
                    candidates.append(
                        (len(overlap.split()), table_name, name, description, overlap)
                    )

        if not candidates:
            return []

        _, table_name, name, description, overlap = max(candidates)
        return [
            f'The question says "{overlap}". {table_name}.{name} is described '
            f'as "{description}" and your query does not use it. Re-read the '
            "question against the columns you selected, and keep yours only if "
            "it is the better reading."
        ]

    def diagnose_sql(self, sql: str) -> str:
        """Probe the live database for the failure modes that survive static checks.

        Covers wrong filter literals, wrong numeric scale, truncating integer
        division, non-existent JSON key paths, and joins that do not follow a
        real relationship. Costs several probes, so it is worth running once
        on a near-final draft rather than on every revision.
        """
        findings = self.probe_findings(sql)
        result = (
            "\n\n".join(findings)
            if findings
            else "No value, JSON-path or join-path problems found."
        )
        return self._record("diagnose_sql", {"sql": sql}, result)

    def probe_findings(self, sql: str) -> list[str]:
        """Live-probe problems with *sql*, empty when it looks sound."""
        findings: list[str] = []
        known_types = _known_types(self.relevant_tables)

        value_mismatches = self._safely(
            "literal check",
            lambda: find_literal_mismatches(self.executor, self.dialect, sql)
            + find_numeric_scale_mismatches(self.executor, self.dialect, sql)
            + find_integer_division_mismatches(
                self.dialect, sql, known_types=known_types
            ),
            [],
        )
        if value_mismatches:
            findings.append(build_value_repair_error(value_mismatches))

        jsonb_mismatches = self._safely(
            "jsonb path check",
            lambda: find_jsonb_path_mismatches(
                self.executor, self.dialect, sql, known_types=known_types
            ),
            [],
        )
        if jsonb_mismatches:
            findings.append(build_jsonb_path_repair_error(jsonb_mismatches))

        join_mismatches = self._safely(
            "join path check",
            lambda: find_join_path_mismatches(
                self.executor, self.dialect, sql, self.database_name
            ),
            [],
        )
        if join_mismatches:
            findings.append(build_join_path_repair_error(join_mismatches))

        case_dirty = self._safely(
            "case-dirty join check",
            lambda: find_case_dirty_join_mismatches(
                self.executor, self.dialect, sql, self.database_name
            ),
            [],
        )
        if case_dirty:
            findings.append(build_case_dirty_join_repair_error(case_dirty))

        return findings

    def preview_result(self, sql: str) -> dict[str, Any]:
        """Execute *sql* and return the raw probe entry.

        Used by the submission gate: a query that errors or comes back empty
        is almost always wrong, and that is far cheaper to learn here than
        after the graph has already executed and scored it.
        """
        return self.executor.run(sql, purpose="submission preview")

    def elaboration_findings(
        self, sql: str, preview: Optional[dict[str, Any]] = None
    ) -> list[str]:
        """Ways *sql* answers a different question from the one asked.

        The variant checks each build the simpler query and run it, reporting
        only when the two disagree — so a ``SUM`` over a single row or a ``>``
        whose boundary row does not exist costs one query and says nothing.
        The join and duplicate-row checks need no query of their own: one is
        pure analysis, the other reads *preview*, which the gate has already
        executed.
        """
        findings: list[str] = []

        for build in (
            lambda: elaboration.find_gratuitous_aggregation(
                sql, self.question, self.dialect
            ),
            lambda: elaboration.find_gratuitous_distinct_count(
                sql, self.question, self.dialect
            ),
            lambda: elaboration.find_strict_boundary(sql, self.dialect),
            lambda: elaboration.find_relationship_as_filter(sql, self.dialect),
        ):
            variant = self._safely("elaboration check", build, None)
            if variant is None:
                continue
            submitted = self.executor.run(sql, purpose="elaboration: submitted")
            simpler = self.executor.run(
                variant.sql, purpose=f"elaboration: {variant.label}"
            )
            if not elaboration.rows_differ(submitted, simpler):
                continue
            findings.append(
                f"{variant.message}\n\n"
                f"The query {variant.label} is:\n{variant.sql}\n\n"
                f"and it returns {simpler['row_count']} row(s):\n"
                f"{_render_rows(simpler['rows'])}"
            )

        inert = self._safely(
            "inert join check",
            lambda: elaboration.find_inert_joins(
                sql, self.dialect, self._has_direct_link
            ),
            [],
        )
        if inert:
            findings.append(elaboration.render_inert_joins(inert))

        if preview is not None:
            repeats = self._safely(
                "duplicate row check",
                lambda: elaboration.find_undeduplicated_rows(
                    sql, self.dialect, preview
                ),
                None,
            )
            if repeats:
                findings.append(repeats)

        return findings

    def _has_direct_link(self, table_a: str, table_b: str) -> bool:
        """Whether the relationship graph knows a single-hop route between two tables.

        This is what tells a detour apart from a legitimate bridge: a
        many-to-many's two ends have no direct edge, so going through the
        bridge is the only way to connect them at all. Two tables that *do*
        have a direct edge can be reached the long way round as well, and
        that longer path arrives at the same place with different rows.
        """
        anchor = self._any_column_id(table_a)
        dest = self._any_column_id(table_b)
        if not anchor or not dest:
            return False
        # Tried both ways round for the same reason the join-path probe does:
        # find_join_path walks outgoing FK edges only, so a real edge written
        # parent-first comes back empty on the forward call alone.
        hops = find_join_path(anchor, dest) or find_join_path(dest, anchor)
        return len(hops) == 1

    def _any_column_id(self, table_name: str) -> Optional[str]:
        """A catalog column id for *table_name*, to stand for the table itself.

        ``find_join_path`` only uses its endpoints to identify the tables, so
        which column is passed does not matter.
        """
        match = next(
            (
                t
                for t in self.relevant_tables
                if (t.get("name") or "").lower() == table_name.lower()
            ),
            None,
        )
        if match is None:
            return None
        for col in match.get("columns") or []:
            name = col.get("name") if isinstance(col, dict) else col
            if not name:
                continue
            column_id = find_column_id_by_table_and_name(
                table_name,
                str(name),
                match.get("database_name") or self.database_name,
            )
            if column_id:
                return column_id
        return None

    def inspect_table(self, table_name: str) -> str:
        """Show one table's full schema plus a few real rows.

        The prompt's schema block is trimmed to fit, so a column the model
        needs may be summarised away; this is how it gets the whole thing.
        """
        match = next(
            (
                t
                for t in self.relevant_tables
                if (t.get("name") or "").lower() == table_name.lower()
            ),
            None,
        )
        if match is None:
            available = ", ".join(
                sorted((t.get("name") or "?") for t in self.relevant_tables)
            )
            return self._record(
                "inspect_table",
                {"table_name": table_name},
                f"No table named {table_name!r} is in scope. Available: "
                f"{available or '(none)'}. Use search_tables to look for others.",
            )

        schema_text = format_tables_for_prompt(
            [match], target_db=self.database_name, dialect=self.dialect
        )
        # Qualified the same way the prompt tells the model to write it, so
        # the sample comes from the table the model will actually query.
        qualified = qualify_table(
            match.get("database_name") or self.database_name or "",
            match.get("schema_name") or "",
            match.get("name") or "",
            self.dialect,
        )
        sample = self.executor.run(
            f"SELECT * FROM {qualified} LIMIT 5",
            purpose=f"sample rows from {qualified}",
        )
        if sample["ok"]:
            schema_text += f"\n\nSample rows:\n{_render_rows(sample['rows'])}"
        else:
            schema_text += f"\n\nSample rows unavailable: {sample['error']}"
        return self._record("inspect_table", {"table_name": table_name}, schema_text)

    def search_tables(self, query: str) -> str:
        """Find tables beyond the ones already in scope.

        The same VDB search ``reconstruct_sql`` runs when it decides an error
        was caused by missing data — reachable here before a wrong query has
        been written rather than after.
        """
        if self.data_retriever is None:
            return self._record(
                "search_tables",
                {"query": query},
                "Table search is unavailable in this run; work with the tables "
                "already in scope.",
            )

        hits = self._safely(
            "table search",
            lambda: get_relevant_tables(
                self.data_retriever,
                query,
                k=_SEARCH_K,
                database_name=self.database_name,
            ),
            [],
        )
        if not hits:
            return self._record(
                "search_tables", {"query": query}, "No matching tables found."
            )

        existing_ids = {
            str(t.get("id")) for t in self.relevant_tables if t.get("id") is not None
        }
        new_ids = [
            str(t["id"])
            for t in dedupe_merge_relevant_tables(hits)
            if t.get("id") is not None and str(t["id"]) not in existing_ids
        ][:_MAX_DISCOVERED_TABLES]
        if not new_ids:
            return self._record(
                "search_tables",
                {"query": query},
                "Nothing beyond the tables already in scope.",
            )

        enriched = self._safely("table enrichment", lambda: fetch_tables_by_ids(new_ids), [])
        if not enriched:
            return self._record(
                "search_tables",
                {"query": query},
                "Nothing beyond the tables already in scope.",
            )

        # Merged into scope, not just shown: a table the model is about to
        # join has to reach path_state too, or validate_sql_query will fail to
        # resolve it against the catalog.
        self.relevant_tables.extend(enriched)
        self.discovered_tables.extend(enriched)

        rendered = format_tables_for_prompt(
            enriched, target_db=self.database_name, dialect=self.dialect
        )
        names = ", ".join(t.get("name", "?") for t in enriched)
        return self._record(
            "search_tables",
            {"query": query},
            f"Added {len(enriched)} table(s) to scope: {names}\n\n{rendered}",
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _parse_error(self, sql: str) -> str:
        """Catalog parse failure for *sql*, or ``""`` when it parses."""
        try:
            if self._schemas is None:
                # Fetched once per question: the catalog does not change
                # mid-construction, and the model may call check_sql on every
                # revision it makes.
                self._schemas = get_schemas_by_ids(fetch_all_schema_ids())
            schemas = self._schemas
            dialects = [self.dialect] if self.dialect else []
            query_obj = parse_query_single(sql=sql, schemas=schemas, dialects=dialects)
        except Exception as exc:  # noqa: BLE001 — surfaced to the model as text
            return str(exc)
        if query_obj is None:
            return (
                "it references no table the catalog knows about; check the table "
                "names against the schema above"
            )
        return ""

    def _safely(self, label: str, fn, fallback):
        """Run a check, downgrading any failure to *fallback*.

        A check that breaks must not cost the question — the agent simply gets
        no finding from it, exactly as if it had found nothing.
        """
        try:
            return fn()
        except Exception:  # noqa: BLE001
            logger.warning("SQL agent %s failed", label, exc_info=True)
            return fallback

    def _record(self, name: str, args: dict[str, Any], result: str) -> str:
        self.calls.append({"tool": name, "args": args, "result": result})
        logger.info(
            "sql_agent tool %s(%s) -> %d chars",
            name,
            ", ".join(f"{k}={str(v)[:80]!r}" for k, v in args.items()),
            len(result),
        )
        return result


__all__ = ["SQLAgentToolbox"]
