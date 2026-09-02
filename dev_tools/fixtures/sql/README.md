# Fixture databases

Build everything with one command (Postgres must be up):

```bash
docker compose up -d postgres
uv run --no-sync python -m dev_tools.fixtures.seed_fixtures
```

Idempotent — re-running skips anything already loaded. To rebuild a Postgres
fixture from scratch, drop the database first; for SQLite, delete the
`.sqlite`.

| Fixture | Engine | Source | Licence |
|---|---|---|---|
| `pagila` | Postgres | [devrimgunduz/pagila] `pagila-v3.1.0`, trimmed | MIT |
| `pagila_analytics` | Postgres | GSF-authored addendum to `pagila` | — |
| `chinook` | SQLite | [lerocha/chinook-database], unmodified | MIT |

[devrimgunduz/pagila]: https://github.com/devrimgunduz/pagila
[lerocha/chinook-database]: https://github.com/lerocha/chinook-database

## Why these, and not a small hand-written schema

The catalog tests are graded almost entirely on fidelity: does the code find
the same tables, columns, types, and foreign keys? A hand-written toy schema —
a couple of tables, one FK, no views — cannot answer that. Pagila can, because
it has the shapes the code special-cases:

| Feature | What it exercises |
|---|---|
| 7 views + 1 materialized view (`rental_by_category`) | `TableTypes` — `base table` / `view` / `materialized view`. Previously only base tables were ever tested. |
| `payment` partitioned into 22 monthly children | Partitions surface as tables in `information_schema`. Whether the catalog should show them is a real question this fixture forces us to answer. |
| `film.special_features text[]`, `mpaa_rating` enum, `fulltext tsvector` | Column `data_type` handling for non-scalar types. |
| 2 domains, incl. `bıgınt` (Turkish dotless i) | Non-ASCII identifiers surviving the round trip. |
| `film_actor`, `film_category` junction tables | Bridge-table detection (`gsf/semantic/bridge_tables.py`). |
| `customer → address → city → country` | Multi-hop `find_join_path` traversal. |
| Two schemas — `public` and `analytics` | The `Schema` tier, and schema-scoped zones. |
| 2 cross-schema FKs | Join paths that cross a schema boundary. |
| 1000 films / 4581 inventory rows | `store_column_sample_values` and `store_column_uniqueness` are meaningless on a handful of rows. |

Chinook exists to make the set **multi-database and multi-dialect**. Several
behaviours are untestable with only one database: cross-database rejection in
`find_join_path` (`gsf/dal/attributes.py:404`), `delete_by_database` scoping,
per-database pgvector collection resets, and zones spanning databases. It also
drives `gsf/connectors/sqlite.py` and a second sqlglot dialect through the
parsers, and carries a self-referencing FK (`Employee.ReportsTo`) that Pagila
has nowhere.

## How `pagila.sql` was produced

Not a verbatim upstream file. Reproduce with
`dev_tools/fixtures/sql/build_pagila.sh` (see below):

1. **Pinned to `pagila-v3.1.0`, not `master`.** Upstream master targets
   PostgreSQL 18 — it uses `uuidv7()` defaults and `VIRTUAL` generated columns,
   neither of which exists in PG17. GSF runs `pgvector/pgvector:pg17`. `v3.1.0`
   is the newest tag that loads on PG17 **unmodified**, and it still has the
   partitioned table, both domains, the enum, and all 8 views. Pinning beats
   patching: no local edits to a third-party fixture to keep re-applying.
2. **Trimmed.** `DELETE FROM payment WHERE rental_id > 600`, then
   `DELETE FROM rental WHERE rental_id > 600` — in that order, so the FK never
   breaks. Everything else is untouched. Cuts the dump from ~13 MB to ~1.2 MB
   while keeping every table, view, type and constraint. Verified zero orphaned
   rows afterwards.
3. **Dumped as `INSERT`s, not `COPY`.** `pg_dump`'s default `COPY ... FROM
   stdin` needs the psql copy protocol; the seed script uses psycopg's
   `cur.execute()`, which cannot run it. `--rows-per-insert=200` keeps the file
   a fraction of the size of one `INSERT` per row.
4. **`\restrict` / `\unrestrict` stripped.** `pg_dump` 17.6+ emits these psql
   meta-commands; they are not SQL and psycopg rejects them.

## Corrections to earlier assumptions

Recorded because the refactor plan asserted these before they were checked, and
three were wrong:

- Pagila has **no self-referencing foreign key**. There is no `staff.reports_to`
  — that is Sakila/Northwind. Chinook's `Employee.ReportsTo` covers the
  same-schema case; `analytics.category_rollup.parent_rollup_id` covers it in a
  non-public schema.
- Upstream Pagila is **single-schema**. The second schema is
  `pagila_analytics.sql`, authored here — see that file's header for why the
  `Schema` tier and the `-Schema` exclusion in `find_join_path` are untestable
  without it.
- It is **22 base tables, not 15** — the `payment` partitions account for the
  difference.
