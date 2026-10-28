# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the notes that tell near-duplicate tables apart."""

from unittest.mock import patch

import pandas as pd

from auto_ontology.dal.datasources import with_sibling_note
from auto_ontology.semantic import sibling_tables
from auto_ontology.semantic.sibling_tables import (
    TableSpec,
    find_sibling_families,
    format_family_facts,
    measure_family,
    write_family_notes,
    write_sibling_table_notes,
)


def _table(name: str, *columns: str, types: dict[str, str] | None = None) -> TableSpec:
    types = types or {}
    return TableSpec(
        id=f"id-{name}",
        name=name,
        schema_name="dw",
        columns=[{"name": c, "data_type": types.get(c, "varchar")} for c in columns],
    )


SUBJECTS = ("SUBJECT_ID", "SUBJECT_TITLE", "TERM_CODE", "WAREHOUSE_LOAD_DATE")
# The rest of a warehouse: unrelated tables that all carry the load timestamp,
# enough of them that the timestamp counts as plumbing while a column a few
# siblings share does not. BUILDINGS and ROOMS share one real column.
OTHERS = [
    _table("BUILDINGS", "BUILDING_KEY", "BUILDING_NAME", "WAREHOUSE_LOAD_DATE"),
    _table("ROOMS", "ROOM_KEY", "BUILDING_KEY", "WAREHOUSE_LOAD_DATE"),
    *(
        _table(f"UNRELATED_{i}", f"KEY_{i}", f"VALUE_{i}", "WAREHOUSE_LOAD_DATE")
        for i in range(20)
    ),
]


def test_related_names_with_shared_columns_form_one_family() -> None:
    offered = _table("SUBJECT_OFFERED", *SUBJECTS, "SECTION_ENROLLMENT_NUMBER")
    catalog = _table("COURSE_CATALOG_SUBJECT_OFFERED", *SUBJECTS, "GIR_ATTRIBUTE")
    library = _table("LIBRARY_SUBJECT_OFFERED", *SUBJECTS, "NUM_ENROLLED_STUDENTS")

    families = find_sibling_families([offered, catalog, library, *OTHERS])

    assert [[t.name for t in f] for f in families] == [
        ["COURSE_CATALOG_SUBJECT_OFFERED", "LIBRARY_SUBJECT_OFFERED", "SUBJECT_OFFERED"]
    ]


def test_a_shared_load_timestamp_does_not_make_tables_siblings() -> None:
    # BUILDINGS and ROOMS share BUILDING_KEY, but their names are unrelated and
    # a single real column is not heavy overlap.
    assert find_sibling_families(OTHERS) == []


def test_heavy_column_overlap_links_unrelated_names() -> None:
    columns = ("A_ID", "A_NAME", "A_CODE", "A_TYPE", "A_STATUS")
    families = find_sibling_families(
        [
            _table("TIP_THING", *columns),
            _table("LIBRARY_STUFF", *columns, "EXTRA"),
            *OTHERS,
        ]
    )

    assert [[t.name for t in f] for f in families] == [["LIBRARY_STUFF", "TIP_THING"]]


def test_an_oversized_chain_is_split_into_real_families() -> None:
    # BUILDING_ADDRESS chains to BUILDING by name and two shared columns, which
    # glues the building and address copies into one group of seven.
    tables = list(OTHERS)
    for entity, columns in {
        "BUILDING": ("BUILDING_KEY", "BUILDING_NAME", "SITE", "AREA", "HEIGHT"),
        "BUILDING_ADDRESS": ("BUILDING_KEY", "BUILDING_NAME", "STREET", "CITY", "ZIP"),
        "FLOOR": ("FLOOR_KEY", "FLOOR_NAME", "LEVEL", "FLOOR_AREA", "FLOOR_USE"),
    }.items():
        for prefix in ("FAC_", "FCLT_"):
            tables.append(_table(prefix + entity, *columns))
        tables.append(_table(f"FCLT_{entity}_HIST", *columns, "VALID_FROM"))
    tables.append(
        _table(
            "FCLT_BUILDING_HIST_1",
            "BUILDING_KEY",
            "BUILDING_NAME",
            "SITE",
            "AREA",
            "HEIGHT",
        )
    )

    with patch.object(sibling_tables, "_MAX_FAMILY_SIZE", 4):
        families = find_sibling_families(tables)

    assert sorted(tuple(t.name for t in f) for f in families) == [
        ("FAC_BUILDING", "FCLT_BUILDING", "FCLT_BUILDING_HIST", "FCLT_BUILDING_HIST_1"),
        ("FAC_BUILDING_ADDRESS", "FCLT_BUILDING_ADDRESS", "FCLT_BUILDING_ADDRESS_HIST"),
        ("FAC_FLOOR", "FCLT_FLOOR", "FCLT_FLOOR_HIST"),
    ]


class _FakeConnector:
    """Answers the two queries measure_family sends, from canned values."""

    dialect = "mysql"

    def __init__(
        self, ranges: dict[tuple[str, str], tuple[object, object]], rows: dict[str, int]
    ):
        self.ranges = ranges
        self.rows = rows
        self.sql: list[str] = []

    def qualify(self, schema: str, table: str) -> str:
        return f"`{schema}`.`{table}`"

    def execute(self, sql: str) -> pd.DataFrame:
        self.sql.append(sql)
        table = sql.split("`dw`.`")[1].split("`")[0]
        if sql.startswith("SELECT COUNT(*)"):
            return pd.DataFrame([[self.rows[table]]])
        column = sql.split("MIN(`")[1].split("`")[0]
        low, high = self.ranges.get((table, column), (None, None))
        return pd.DataFrame([[low, high]])


def test_measure_family_reports_period_coverage_and_distinctive_columns() -> None:
    offered = _table(
        "SUBJECT_OFFERED",
        *SUBJECTS,
        "IS_OFFERED_FALL_TERM",
        "TERM_DESCRIPTION",
        "LAST_ACTIVITY_DATE",
        "SECTION_ENROLLMENT_NUMBER",
    )
    summary = _table(
        "SUBJECT_SUMMARY", *SUBJECTS, "SCHOOL_NAME", types={"TERM_CODE": "varchar"}
    )
    connector = _FakeConnector(
        ranges={
            ("SUBJECT_OFFERED", "TERM_CODE"): ("2003JA", "2015SU"),
            # Text that does not lead with the year sorts alphabetically, not by time.
            ("SUBJECT_OFFERED", "LAST_ACTIVITY_DATE"): ("01-APR-02", "31-MAR-03"),
            ("SUBJECT_SUMMARY", "TERM_CODE"): ("1995FA", "2025FA"),
        },
        rows={"SUBJECT_OFFERED": 10_000, "SUBJECT_SUMMARY": 144},
    )

    facts = measure_family(
        [offered, summary], connector, generic={"WAREHOUSE_LOAD_DATE"}
    )

    by_name = {f.table.name: f for f in facts}
    assert by_name["SUBJECT_OFFERED"].periods == {"TERM_CODE": ("2003JA", "2015SU")}
    assert by_name["SUBJECT_SUMMARY"].periods == {"TERM_CODE": ("1995FA", "2025FA")}
    assert by_name["SUBJECT_SUMMARY"].row_count == 144
    assert "SECTION_ENROLLMENT_NUMBER" in by_name["SUBJECT_OFFERED"].only_here
    assert by_name["SUBJECT_SUMMARY"].only_here == ["SCHOOL_NAME"]
    # A flag and a label match the period pattern but are never measured.
    assert not any(
        "IS_OFFERED_FALL_TERM" in sql or "TERM_DESCRIPTION" in sql
        for sql in connector.sql
    )
    text = format_family_facts(facts)
    assert "TERM_CODE ranges from 2003JA to 2015SU" in text
    assert "Has columns most similar tables lack: SCHOOL_NAME" in text


def test_notes_are_keyed_by_table_id_and_cleaned() -> None:
    facts = measure_family(
        [
            _table("ACADEMIC_TERMS", "TERM_CODE"),
            _table("ACADEMIC_TERMS_ALL", "TERM_CODE"),
        ],
        connector=None,
        generic=set(),
    )
    reply = sibling_tables._SiblingNotes(
        notes=[
            sibling_tables._SiblingNote(
                table="academic_terms", note="Terms  from | 1995FA."
            ),
            sibling_tables._SiblingNote(table="NOT_IN_FAMILY", note="Ignored."),
        ]
    )
    with (
        patch.object(sibling_tables, "get_llm_client"),
        patch.object(
            sibling_tables, "invoke_with_structured_output", return_value=reply
        ),
    ):
        notes = write_family_notes(facts)

    assert notes == {"id-ACADEMIC_TERMS": "Terms from / 1995FA."}


def test_every_table_is_rewritten_so_stale_notes_are_cleared() -> None:
    rows = [
        {
            "id": t.id,
            "name": t.name,
            "schema_name": "dw",
            "description": "",
            "columns": t.columns,
        }
        for t in [
            _table("ACADEMIC_TERMS", "TERM_CODE", "TERM_STATUS"),
            _table("ACADEMIC_TERMS_ALL", "TERM_CODE", "TERM_STATUS"),
            *OTHERS,
        ]
    ]
    with (
        patch.object(
            sibling_tables, "fetch_tables_for_sibling_notes", return_value=rows
        ),
        patch.object(sibling_tables, "_resolve_connector", return_value=None),
        patch.object(
            sibling_tables,
            "write_family_notes",
            return_value={
                "id-ACADEMIC_TERMS": "Recent terms.",
                "id-ACADEMIC_TERMS_ALL": "All terms.",
            },
        ),
        patch.object(sibling_tables, "store_table_sibling_notes") as store,
    ):
        written = write_sibling_table_notes("dw")

    assert written == 2
    stored = store.call_args.args[0]
    assert stored["id-ACADEMIC_TERMS"] == "Recent terms."
    assert stored["id-BUILDINGS"] is None


def test_off_switch_skips_the_pass() -> None:
    with (
        patch.object(sibling_tables, "SIBLING_NOTES_ENABLED", False),
        patch.object(sibling_tables, "fetch_tables_for_sibling_notes") as fetch,
    ):
        assert write_sibling_table_notes("dw") == 0
    fetch.assert_not_called()


def test_the_note_follows_the_description() -> None:
    assert with_sibling_note("Academic terms.", "Covers 1995FA on.") == (
        "Academic terms. Versus similar tables: Covers 1995FA on."
    )
    assert (
        with_sibling_note(None, "Covers 1995FA on.")
        == "Versus similar tables: Covers 1995FA on."
    )
    assert with_sibling_note("Academic terms.", None) == "Academic terms."
