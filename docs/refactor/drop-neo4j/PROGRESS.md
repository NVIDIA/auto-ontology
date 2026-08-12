# Drop Neo4j — progress log

Append-only. Newest entries at the bottom. **Never rewrite an earlier entry** —
if one turns out to be wrong, add a correcting entry that says so.

Each entry carries: date, phase, what landed, files touched, tests added,
Done-criteria met or explicitly not met, and what's next.

See [PLAN.md](PLAN.md) for the phase definitions and their Done criteria, and
[DECISIONS.md](DECISIONS.md) for anything that deviated from the plan.

---

## Blocked / needs a human

> Open questions go here rather than being resolved by guess. Anything touching
> zone scoping or access control belongs here by default.

Nothing open.

### Resolved

**Two "unit" tests silently require a live Neo4j.** *(found 2026-08-11,
**resolved** 2026-08-11 — skipped until Phase 10)*
`test_export_model_filters_by_database_id` and
`test_export_model_all_databases_uses_empty_filter` in
`gsf/server/model_interchange/tests/test_model_interchange.py` patch
`dal.validate_database_ids`, `dal.fetch_export_rows` and
`dal.resolve_sql_column_ids`, but `service.export_model` also calls
`_dialect_by_database_name()` (`service.py:56`), which calls `list_connections()`
→ `MATCH (db:Database) RETURN properties(db)`. That call is unpatched, so both
fail with connection refused on :7687 unless Neo4j happens to be up.
Pre-existing; not caused by this refactor.

Now `@pytest.mark.skip` with a reason that states the unpatched call path, so
the next reader doesn't have to re-derive it. **Phase 10 must un-skip them** and
rewrite them against the Pagila/Chinook fixture — they are the only coverage
`export_model`'s database filtering has.

---

## 2026-08-11 — Phase 0 — docs landed

**What landed:** `docs/refactor/drop-neo4j/` created with `PLAN.md` (the
approved plan, verbatim apart from the self-reference in "This plan lives in
the repo"), plus `PROGRESS.md`, `DECISIONS.md`, and `SCHEMA.md` stubs. Pointer
added to `README.md`; `CLAUDE.md` gained a line directing future sessions here
before touching `gsf/dal/` or `gsf/catalog/`.

**Files touched:** `docs/refactor/drop-neo4j/{PLAN,PROGRESS,DECISIONS,SCHEMA}.md`,
`README.md`, `CLAUDE.md`.

**Tests added:** none — documentation only.

**Done criteria:** `docs/refactor/drop-neo4j/` exists and is linked from
`README.md` and `CLAUDE.md` — **met**. The other two Phase 0 criteria
(`reserved_words` severed, surface test green) are **not yet met**; they are
the remaining Phase 0 work.

**Next:** create `gsf/catalog/constants.py` and repoint the ~29
`reserved_words` imports.

---

## 2026-08-11 — Phase 0 — `reserved_words` severed

**What landed:** new `gsf/catalog/` package with `constants.py` holding
GSF-owned `Labels` / `TableTypes` / `Edges` / `Props`, forked verbatim from
`nemo_retriever.tabular_data.ingestion.model.reserved_words`. All 30 import
sites across 29 modules repointed to `gsf.catalog.constants`. No shim and no
re-export — a re-export would have kept a live import edge to the package being
dropped.

**Files touched:** `gsf/catalog/{__init__,constants}.py` (new),
`gsf/catalog/tests/test_constants.py` (new), and the import line in 29 modules
under `gsf/{dal,server,retrieval,connectors}/` (mechanical `sed`, no other
content change).

**Tests added:** `gsf/catalog/tests/test_constants.py` — 5 tests asserting the
fork is attribute-for-attribute identical to the library, plus a guard that no
vocabulary class was missed. Verified 21 attributes across the 4 classes match
exactly. **This test is deleted in Phase 11** once nothing imports the library's
ingestion package.

**Done criteria:** `grep -rn reserved_words gsf` returns only the docstring
reference in `gsf/catalog/constants.py` — **met**. `ruff check` and
`ruff format` clean.

**Full suite:** 180 passed, 2 failed, 1 skipped. Both failures are the
pre-existing live-Neo4j dependency recorded under *Blocked / needs a human*
above — unrelated to this change, confirmed by reading the unpatched call path.

**Next:** `gsf.dal.close_store()` lifecycle hook, then the DAL surface snapshot
test.

---

## 2026-08-11 — Phase 0 — `close_store()` lifecycle hook

**What landed:** `gsf.dal.close_store()`, called from the `__main__.py` lifespan
in place of poking `neo4j_connection._conn`. Imports inside it are deferred so
`import gsf.dal` stays cheap and carries no import-time driver dependency.
Phase 3 repoints this at the SQLAlchemy engine; nothing in `__main__.py` has to
change again.

**Bug fixed along the way:** `gsf/dal/neo4j_tx.py` opens its *own* driver
(`_driver`) for explicit write transactions, and nothing ever closed it — the
old shutdown path only closed the library's singleton. `close_store()` closes
both, and closes the second even if the first raises, since leaking the driver
because the singleton misbehaved is strictly worse than logging.

**Files touched:** `gsf/dal/__init__.py`, `gsf/server/__main__.py`,
`gsf/dal/tests/test_close_store.py` (new).

**Tests added:** 4 — both connections closed and globals cleared; idempotent
across repeat calls; the driver still closes when the shared connection raises;
a driver opened without the singleton is still closed. Verified `create_app()`
builds (61 routes) and the entrypoint no longer references
`neo4j_connection`.

**Correction to PLAN.md.** The plan claimed this drops the heavy
`nemo_retriever` import from the API entrypoint. Measured: it does not.
`nemo_retriever.tabular_data.neo4j` still loads transitively via the ten
`get_neo4j_conn` imports in `gsf/dal/*.py`; only the direct import at
`__main__.py:13` goes away. `PLAN.md` § Phase 0 has been amended with the
correction in this same commit. The real benefit is the public seam, not import
weight — import weight only improves in Phase 11.

**Next:** the DAL surface snapshot test, which closes out Phase 0.

---

## 2026-08-11 — Phase 0 — DAL surface frozen — **Phase 0 complete**

**What landed:** `gsf/dal/tests/test_dal_surface.py` plus the generated
`dal_surface.json` snapshot — **15 modules, 143 public callables**, each with
its signature. Regenerate deliberately with
`uv run python -m gsf.dal.tests.test_dal_surface`.

Two design calls worth recording:

- **Functions only, not constants.** Several DAL module constants
  (`TABLE_COUNTS_SUBQUERY`, the `cypher_fragments` output) are Cypher, which is
  precisely what this refactor deletes. Freezing them would fight the work.
- **Classes recorded by their bases**, not a signature — the public classes are
  exceptions and a dataclass, and what callers depend on is
  `except UnknownDatabaseIdsError` still catching the same thing. Several
  exceptions have no introspectable signature at all, which is what forced the
  distinction.

`test_postgres_and_neo4j_surfaces_match` is written and **skips** until
`gsf/dal/pg/` and `gsf/dal/neo4j/` exist side by side. From Phase 3 it is the
check that makes the `GSF_STORE` flip safe.

**Files touched:** `gsf/dal/tests/test_dal_surface.py`,
`gsf/dal/tests/dal_surface.json` (both new).

**Tests added:** 18 (15 parametrised per-module + 3 guards). Verified
non-vacuous by renaming `delete_data_layer`'s `database_name` keyword to
`db_name` — the suite failed with the exact diff — then reverting.

**Done criteria — all three met:**
- `docs/refactor/drop-neo4j/` exists, linked from `README.md` and `CLAUDE.md` ✅
- `grep -rn reserved_words gsf` returns only the docstring reference in
  `gsf/catalog/constants.py` ✅
- surface test green ✅

**Phase 0 totals:** 3 commits, **27 tests added** (5 vocabulary + 4
`close_store` + 18 surface), 1 of which skips until Phase 3. Full suite:
**201 passed, 2 failed, 2 skipped** — the 2 failures are the pre-existing
live-Neo4j dependency recorded under *Blocked / needs a human*, unchanged from
before this phase. `ruff check` and `ruff format` clean.

**Next:** Phases 1, 2 and 3 run in parallel and touch disjoint files. The
sequencing constraint is that Phase 2's fixture seed script (Pagila + Chinook)
must land before Phase 1 can verify its fork-equivalence criterion, so start
there.

Before Phase 4, the **ERD review gate** at the end of Phase 3 needs a human in
the room — polymorphic `zone_target`, the edge-property-carrying
`sql_query_table`, and the `join_edge` view shape decide whether Phases 5–10
are mechanical or a rewrite.

---

## 2026-08-11 — Phase 0 — follow-ups from review

**Merge from `origin/main`:** `origin/main` is at `ce62e52`; this branch was
already 4 ahead / 0 behind, so nothing to merge. Recorded because
[DECISION-001](DECISIONS.md) now requires a merge at every phase boundary —
`git fetch origin && git merge origin/main` before any code, logged here with
the SHA and whether it conflicted.

**What landed:**
- The two live-Neo4j `test_export_model_*` tests are now `@pytest.mark.skip`
  with a reason naming the unpatched `_dialect_by_database_name()` →
  `list_connections()` call path. **Phase 10 must un-skip them** — they are
  `export_model`'s only database-filtering coverage.
- DECISION-001 added, and `PLAN.md` § *This plan lives in the repo* gained the
  per-phase merge rule.

**Files touched:** `gsf/server/model_interchange/tests/test_model_interchange.py`,
`docs/refactor/drop-neo4j/{PLAN,PROGRESS,DECISIONS}.md`.

**Suite:** `gsf/server/model_interchange/tests/` — 12 passed, 2 skipped. The
two failures that stood at the end of Phase 0 are gone; the suite is green.

---

## 2026-08-11 — Phase 2 — fixture databases

**Merge:** `origin/main` at `ce62e52`; branch 6 ahead / 0 behind. Nothing to
merge, no conflicts.

**What landed:** Pagila (Postgres) and Chinook (SQLite) replace a 16-line,
2-table `testdb.sql` as the fixture the whole refactor is graded against.

- `dev_tools/sql/pagila.sql` — 1.2 MB, from upstream `pagila-v3.1.0`, trimmed.
- `dev_tools/sql/pagila_analytics.sql` — GSF-authored second schema.
- `dev_tools/sql/chinook.sql` — upstream, unmodified; the `.sqlite` is
  generated and gitignored.
- `dev_tools/sql/build_pagila.sh` — regenerates the dump reproducibly.
- `dev_tools/sql/README.md` — provenance and rationale.
- `dev_tools/{seed_fixtures,build_sqlite_fixtures}.py`, and
  `seed_local_postgres.py` reworked around a `Fixture` dataclass with a
  sentinel-table check so re-seeding is genuinely idempotent (the old
  docstring claimed idempotence but `pg_dump` output is not re-runnable).
- `THIRD_PARTY_NOTICES.md` — both fixtures, both MIT.

**Four things that were not anticipated**, all now in
`dev_tools/sql/README.md` and [DECISION-002](DECISIONS.md):
upstream Pagila master needs PG18 (`uuidv7()`, `VIRTUAL` columns) and GSF runs
pg17, so it is pinned to v3.1.0 rather than patched; master's data is 13 MB not
~3 MB, hence the trim; `pg_dump`'s `COPY ... FROM stdin` cannot run through
psycopg's `execute()`, hence `--inserts`; and `pg_dump` 17.6+ emits
`\restrict`/`\unrestrict` psql meta-commands that had to be stripped.

**Three plan claims were wrong and are corrected in `PLAN.md`:** Pagila has no
self-referencing FK (no `staff.reports_to` — that's Sakila/Northwind); it is
single-schema, so the second schema is GSF-authored; and it is 22 base tables,
not 15, because `payment` is partitioned. Pagila is MIT, not BSD.

**Tests added:** `dev_tools/tests/test_fixtures.py` — 12 tests asserting every
shape claim (views, matview, partitions, two schemas, cross-schema FKs,
self-reference, array/enum/tsvector columns, the multi-hop join chain, zero
orphans after the trim, row counts). 12 pass against a live Postgres; without
one it degrades to 3 passed / 9 skipped. Verified the seeder end-to-end from a
clean database and confirmed a second run skips everything.

**Deviation from the plan's letter:** the plan said to extend
`seed_local_postgres.py`'s loop. SQLite shares nothing with Postgres seeding, so
Chinook got its own module and `seed_fixtures.py` runs both — recorded in
DECISION-002.

**Flagged for Phase 4:** partitioned `payment` means ingestion will discover 22
base tables where a reader expects 1. Whether partition children belong in the
catalog is a product question, not a porting detail; Phase 4 must decide it
deliberately rather than inherit whatever the current code happens to do.

**Not yet done in Phase 2:** the golden capture itself. It needs a live Neo4j
plus ingestion and `/semantic/compile` (which needs model credentials), so it
is the next piece of work.

**Also fixed the stale `CLAUDE.md` dev command** (the second item that had been
under *Blocked*). It documented `uv run uvicorn gsf.server.main:app`, but
`gsf/server/main.py` does not exist — the factory is `create_app()` in
`gsf/server/__main__.py`. Corrected to the `--factory` form, and added the two
commands that were missing entirely: `python -m gsf.server` (what the
Dockerfile runs) and `python -m gsf.ingestion_service`. All three verified to
resolve, and `gsf/server/main.py` confirmed absent.

---

## 2026-08-11 — Phase 2 — connector bug found by the fixture, and fixed

**What happened:** ingesting Pagila into Neo4j for the golden capture wrote
**21 of the 30 relations** in `public`. Missing: the partitioned table
`payment`, its 7 partition children, and the materialized view
`rental_by_category`. Two independent pre-existing bugs in
`gsf/connectors/postgres.py`, both fixed here — see
[DECISION-003](DECISIONS.md) for the full reasoning:

- `get_tables` allowed `relkind IN ('r','v','m','f')`. Partition *children* were
  correctly hidden by `relispartition = false`, but the parent is `'p'` and was
  not allowed through, so an entire queryable table was invisible.
- Both `get_tables` and `get_columns` drove from `information_schema`, which
  does not list materialized views **or their columns**. `TableTypes.MATERIALIZED_VIEW`
  had been unreachable dead code since it was written.

**Why fix it now rather than defer:** `gsf/connectors/` sits above the DAL and
is not part of the port, so this changes the Neo4j and Postgres paths
identically and cannot complicate Phases 5–10. Deferring was the worse option —
the goldens are captured from current behaviour and become the oracle for the
whole refactor, so capturing first would have frozen the bug into the oracle and
every later phase would faithfully reproduce a catalog with missing tables.

**Third finding, unplanned:** driving from `pg_catalog` also changed `data_type`
for 5 of 143 previously-ingested Pagila columns — `USER-DEFINED` →
`mpaa_rating`, `ARRAY` → `text[]`, `integer` → `year`. Strictly better for
SQL generation, but it is a real change: those columns' descriptions and
embeddings differ on next ingest. `format_type(atttypid, NULL)` was chosen
specifically because it reproduces `information_schema`'s unqualified spelling,
so the other 138 columns are byte-identical — verified, not assumed.

Recorded as behaviour change **B3**.

**Files touched:** `gsf/connectors/postgres.py`,
`gsf/connectors/tests/test_postgres.py` (new), `docs/refactor/drop-neo4j/DECISIONS.md`.

**Tests added:** 8, covering partitioned parent present, children absent,
matview present and correctly typed, all three `TableTypes` values reachable,
every catalogued relation has columns, declared types not placeholders, system
schemas excluded, `is_nullable` spelling preserved. All pass against the
fixture; skip without it.

**Verified after the fix:** `public` catalogues 23 relations — 15 base tables
(incl. `payment`), 7 views, 1 materialized view — and 0 partition children.

**Next:** the golden capture itself, now that the catalog it captures is
correct.

**Full suite after the connector fix:** **221 passed, 4 skipped, 0 failed**
(was 201/2/2 at the end of Phase 0). Runtime also fell from 224s to 60s — the
two skipped `test_export_model_*` tests were spending ~3 minutes retrying Neo4j
connections on every run.

---

## 2026-08-11 — Phase 2 — golden capture — **Phase 2 complete**

**What landed:** the fidelity oracle for Phases 5–10. **122 DAL reads** recorded
from the live Neo4j graph, replayed by `gsf/dal/tests/test_golden.py`
(128 tests: 122 comparisons + 6 structural/invariant checks).

- `dev_tools/seed_graph_fixture.py` — builds the canonical graph: both catalogs
  plus a hand-authored semantic layer (5 terms with varied certification flags,
  8 column attributes, 3 semantic FKs, 4 SQL attributes across all four `source`
  values, 2 custom analyses, 1 PQL analysis, 3 zones — one schema-scoped, one
  spanning both databases, one disabled).
- `dev_tools/capture_dal_golden.py` — captures and normalises.
- `gsf/dal/tests/golden/dal_reads.json` — 183 KB, 0 captures raising.

Resulting graph: 12 node labels (including `disableZone`, the label-swap case)
and 10 relationship types — every type the DAL uses except `IS_A`/`ROLE`/`UNION`,
which are declared but unused in GSF.

**Semantic layer is hand-authored, not compiled.** `/semantic/compile` drives an
LLM: it needs credentials and returns something slightly different every run,
neither of which is acceptable for an oracle. Writing the same artifacts through
`gsf.dal` is reproducible and exercises the same write paths the port must
preserve. Embedding is stubbed to a no-op so building the fixture needs no
embedding endpoint.

**Determinism was the hard part.** Verified by tearing the graph down and
rebuilding it: **0 of 122 captures differ.** Four separate sources of noise had
to be handled, three of which were bugs in the capture rather than the DAL:

1. *Ids as dict keys.* `normalise` rewrote values but not keys, and several
   reads return maps keyed by node id.
2. *Sets.* Some reads return `set`; `default=str` stringified them at
   serialisation — *after* normalisation — leaving raw uuids inside a string.
3. *DataFrames.* A few reads return them, and `default=str` captured a truncated
   repr with `...` eliding most columns and raw uuids in what remained.
4. *Ingest timestamps* (`created`), redacted rather than dropped so a port that
   stops populating them still fails.

**A genuine finding, not a capture bug:** exploration edge direction flips
between rebuilds. Both `fetch_data_exploration_edges`
(`WHERE source.id < target.id`) and `fetch_semantic_exploration_graph`
(`tuple(sorted((a, b)))`) canonicalise undirected edges by comparing **uuids**,
which are random — so the direction is stable for a given database but arbitrary
across rebuilds. The capture re-orients by token instead, swapping the paired
`source_column`/`target_column` fields in tandem, since swapping ends without
them would describe an edge that doesn't exist.

**Second finding:** `fetch_sorted_tables` orders only by `query_count DESC`, and
nearly every table ties at 0 — so a function whose name promises sorting returns
rows in arbitrary order. Not fixed: it is `gsf/dal/datasources.py`, which Phase 6
ports, and adding a tiebreaker is a behaviour change that should be deliberate.
**Phase 6 should decide.**

**Third finding, out of scope:** `parse_query_single` resolves tables only via
column references, so `SELECT count(*) FROM film` is rejected as "doesn't
reference any table known to the catalog" even when `film` is catalogued — a
misleading message for a user authoring a custom analysis. It lives in the
NeMo-Retriever parser; noted, not fixed.

**Known limitation, deliberate:** list order is **not** covered. Captures sort
lists canonically, because most DAL queries lack a total `ORDER BY` and freezing
an arbitrary observed order would fail the port for behaviour Neo4j never
guaranteed. Ordering that *is* contractual — paging stability across
`skip`/`limit` — needs its own tests in the invariants suite.

**Tests added:** 128. Beyond the per-read comparisons: the golden file must be
non-empty (so a missing file fails loudly rather than making every comparison
vacuous), every DAL module must be represented, no capture may raise, no read
may be captured-but-unrecorded or recorded-but-uncaptured, **scoped access must
return strictly less than admin access** (a port that silently drops the zone
filter would still match most goldens while handing every user the admin view),
and a disabled zone must stay visible to admins.

**Phase 2 Done criteria — met.** Fixture databases landed, semantic layer
seeded, every `fetch_*`/`get_*`/`list_*` snapshotted under all three zone modes.

**Next:** Phase 1 (fork the write path) and Phase 3 (Postgres foundations), which
are independent of each other. Phase 1's fork-equivalence check can now run,
since it compares against exactly this fixture.

---

## 2026-08-11 — Phases 1 + 3 — merge from `origin/main`

**Merge:** `origin/main` `ce62e52` → **`0a3ed24`**, 4 commits, **no conflicts**.
Brings in API tokens for machine-to-machine access (#176) and conversation
follow-up (#177). 55 files, +2691/−288.

**DAL surface unchanged** — `test_dal_surface.py` green, so nothing upstream
touched a `gsf/dal` signature. Goldens still replay clean (159 passed in
`gsf/dal/tests` + `gsf/catalog/tests`). Full suite **361 passed, 4 skipped**,
up from 349 as upstream added 12 tests.

**Worth noting for Phase 3:** upstream added `gsf/server/chat/conversation_dal.py`,
a new *Postgres* data-access module living outside `gsf/dal/`. Phase 3 should
check whether it builds its connection through `gsf/infra/postgres.py` or opens
its own, and whether it should share the pooled engine this phase introduces —
a second connection-management pattern landing while we build the first one is
exactly the kind of drift that is cheap to fix now and annoying later.

---

## 2026-08-11 — Phase 1 — the catalog write path is GSF-owned — **Phase 1 complete**

**Merge:** logged in the previous entry — `origin/main` `ce62e52` → `0a3ed24`,
no conflicts. Nothing further landed upstream during this phase.

**What landed:** ~2,500 lines forked out of
`nemo_retriever.tabular_data.ingestion` into `gsf/catalog/`, still writing to
Neo4j. Phase 4 rewrites `gsf/catalog/store/` for Postgres; **everything above
`store/` forks once and is done.**

| New module | Forked from |
|---|---|
| `extract.py` | `ingestion/extract_data.py` |
| `normalize.py` | `ingestion/utils.py` |
| `write.py` | `ingestion/write_to_graph.py` |
| `sql_parse.py` | `ingestion/services/queries.py` |
| `services/schema.py` | `ingestion/services/schema.py` |
| `model/{node,schema,query}.py` | `ingestion/model/{neo4j_node,schema,query}.py` |
| `parsers/{sqlglot_extractor,query_comparator,schemas_parser}.py` | `ingestion/parsers/*` |
| `store/{db,schemas,queries,edges}.py` | `ingestion/dal/{db,schemas,queries,utils}_dal.py` |
| `store/indexes.py` | `ingestion/indexes.py` |
| `store/connection.py` | `neo4j/neo4j_connection.py` |
| `ingest.py` | **new** — replaces `TabularSchemaExtractOp` |

`Neo4jNode` → `CatalogNode` throughout. Public surface is the four entry points
the plan specified: `gsf.catalog.ingest_catalog`,
`gsf.catalog.sql_parse.parse_query_single`, `gsf.catalog.store.queries.add_query`,
`gsf.catalog.model.Schema` / `CatalogNode`.

**Rewired off the library** — 17 GSF modules plus 3 dev tools. `get_neo4j_conn`
in all 11 `gsf/dal/*.py`, `add_query` and `CatalogNode` in
`model_interchange.py` / `sql_attributes/service.py` /
`custom_analyses/service.py`, `parse_query_single` in `sql_utils.py` /
`sql_parse_validation.py`, `CatalogNode` + `Schema` in `graph_schemas.py`,
`close_store()`, and `ingestion_service/ingest.py`. `grep -rnE
"^\s*(from|import)\s+nemo_retriever\.tabular_data\.(ingestion|neo4j)" gsf
dev_tools` now returns exactly one line — `test_constants.py`, the Phase-0 pin
test that imports the library on purpose to compare against. The remaining
textual hits are provenance docstrings inside `gsf/catalog/` and the two pin
tests.

**`TabularSchemaExtractOp` deleted, not moved**, as the plan required.
`gsf/ingestion_service/ingest.py` is now three straight-line calls:
`ingest_catalog(connector)` → `TabularFetchEmbeddingsOp(...)(pair)` →
`batch_embed(rows, params)`. The middle two stay on the library and are called
directly, which is exactly what `Graph` did — it calls `operator.run(data)` and
`AbstractOperator.__call__` *is* `run`. That equivalence does **not** hold for
`_BatchEmbedActor`: it is an *archetype* operator resolved to a CPU or GPU
variant only during graph execution, so `gsf.utils.embedding.batch_embed` keeps
a one-node `Graph()` inside it. That is the one place in GSF that names the
private symbol, and a test enforces it.

**Fork equivalence — the Done criterion — met, byte-identically.** Procedure:
capture a baseline, fork, re-seed from scratch, re-capture.

- **Golden:** `dev_tools.seed_graph_fixture --reset` +
  `dev_tools.capture_dal_golden` reproduce `gsf/dal/tests/golden/dal_reads.json`
  with **zero diff** — all 122 DAL reads, across a full teardown and rebuild.
- **Node counts per label:** identical, all 12 labels (209 Column, 37 Table,
  8 ColumnAttribute, 6 Sql, 5 Term, 4 SqlAttribute, 3 Schema, 2 Database,
  2 CustomAnalysis, 2 Zone, 1 PqlAnalysis, 1 disableZone).
- **Relationship counts per type:** identical, all 10 types (249 CONTAINS,
  30 FOREIGN_KEY, 25 SQL, 12 PROPERTY_OF, 8 HAS_ATTRIBUTE, 6 HAS_SQL,
  6 ZONE_OF, 5 REPRESENTS, 3 JOIN, 3 SEMANTIC_FK).
- **Sorted property dump** of every node and every edge, natural-keyed and with
  random uuids and per-run timestamps normalised: identical.
- **Incremental re-ingest:** running `run_ingest` again over Chinook through the
  forked path left the graph and the golden unchanged, so `write.py`'s diffing
  survived the move intact.

**Tests added: 26.**
- `gsf/catalog/tests/test_fork_parity.py` (18) — the fork is compared to the
  library **AST-for-AST**, per module, after applying exactly the mechanical
  rewrite (import paths + `Neo4jNode` → `CatalogNode`). It catches an accidental
  edit during the move *and* upstream drift the fork would otherwise miss.
  Formatting and the provenance docstring are excluded because neither is
  behaviour. Plus a guard that no module can be added to `gsf/catalog/` without
  declaring provenance, and one that nothing above `store/` imports `neo4j` —
  the invariant that keeps Phase 4 contained. Verified non-vacuous by editing a
  forked function body and watching the right module fail.
- `gsf/catalog/tests/test_ingest.py` (5) — the three empty paths and the
  cross-schema concat that was `TabularSchemaExtractOp`'s real contract.
- `gsf/utils/tests/test_embedding.py` (3) — empty short-circuit, plus the two
  that matter: no other GSF file mentions `_BatchEmbedActor`, and the wrapper
  confines it to one import statement.

**Full suite: 387 passed, 4 skipped** (was 361/4). `ruff check` and
`ruff format` clean.

**Three behaviour changes, all in [DECISION-004](DECISIONS.md):**
- **B4** — `_shared_connection` now wraps extraction only, not extraction plus
  the embed HTTP round trip. Strictly narrower and strictly better on Databricks
  (where opening a connection is the slowest and flakiest step), but it changes
  when the source database sees a disconnect, so it is named rather than
  absorbed.
- **B5** — `SemanticEmbedder.embed_graph` is gone; the one-node graph is built
  per call. Microseconds, next to an HTTP call in the same function.
- `extract_tabular_db_data` takes a **connector** instead of a library
  `TabularExtractParams`, and `store_relational_db_in_neo4j` is deleted. Both
  keep library types out of code that is meant to survive Phase 4 untouched.
  `extract.py` is therefore the one module `test_fork_parity.py` lists as
  deliberately diverged rather than pinning.

**Deliberately not renamed:** `gsf/catalog/store/connection.py` keeps
`get_neo4j_conn` and its `_conn` singleton. Renaming would touch 11 modules for
a name Phase 11 deletes, and `gsf.dal.close_store()` is already the seam Phase 3
repoints.

**Phase 1 Done criteria:**
- Forked path produces an identical Neo4j graph to the pre-fork path — node
  counts per label, relationship counts per type, sorted property dump ✅
- Golden captured (Phase 2 had already captured it; this phase proved the fork
  reproduces it byte for byte) ✅
- `uv run pytest` green ✅
- No live import edge to `nemo_retriever.tabular_data.{ingestion,neo4j}` outside
  the two deliberate pin tests ✅

**Next:** Phase 3 (Postgres foundations) is unblocked and independent. Phase 4
then rewrites `gsf/catalog/store/*.py` and nothing else — `test_fork_parity.py`
should be deleted or narrowed at that point, since `store/` stops being a copy.

Two things Phase 4 should know:
- `store/indexes.py` disappears entirely (Alembic owns indexes), as planned.
- `write.py`'s incremental diffing — existing-DB detection, schema
  add/update/delete, the `pulled` timestamp — is the part with no test coverage
  beyond the end-to-end re-ingest above. It deserves its own tests before it is
  rewritten, not after.

## 2026-08-11 — Phase 3 — Postgres foundations

**What landed:** the schema and connection machinery Phases 4–10 build on.
**23 tables + 1 view**, created by Alembic revision `75bdf1cdf36d`. Nothing
writes to them yet — Phase 4 ports the catalog write path onto this.

- `gsf/dal/pg/session.py` — pooled SQLAlchemy engine, plus `store()` and
  `write_transaction()` mirroring `gsf/dal/neo4j_tx.py`'s contract exactly, so
  the ~180 call sites and every `with write_transaction():` block port without
  changing shape.
- `gsf/dal/pg/schema.py` — the full `MetaData(schema="gsf")`. Single source of
  truth; Alembic autogenerates from it.
- `alembic/` + `alembic.ini`, `make migrate` / `migrate-check` /
  `migrate-revision`, a `gsf-migrate` compose service, and
  `helm/gsf/templates/backend-migrate-job.yaml`.
- New deps: `sqlalchemy>=2.0`, `alembic`.
- `docs/refactor/drop-neo4j/SCHEMA.md` regenerated from what actually landed.

**Three things that needed solving, none of them anticipated:**

1. **SQLAlchemy resolves `postgresql://` to psycopg2**, which this repo does not
   install. Added `sqlalchemy_url()`, which pins the psycopg 3 driver — the same
   rewrite `gsf/vdb/postgres.py` already does for asyncpg.
2. **`alembic check` reported drift on all 14 id columns** of a perfectly
   up-to-date database. Postgres stores `gen_random_uuid()::text` as
   `(gen_random_uuid())::text`, and `compare_server_default` compares the text.
   Fixed by writing the default in its normalised form rather than by turning
   the comparison off — a drift check that has to be disabled is not a check.
3. **The image could not run migrations at all.** The Dockerfile's dispatcher
   entrypoint only accepted `server` and `ingestion_service`, and neither
   `alembic/` nor `alembic.ini` was copied into the image. Added a `migrate`
   mode and the COPY lines.

**The destructive failure mode is tested, not assumed.** Alembic's autogenerate
sees any table not in its metadata as removable, so without the `include_object`
filter in `alembic/env.py` it would emit `DROP TABLE public."user"`. Verified by
planting Prisma-shaped tables in `public` plus a `vdb` schema, then running
`alembic check` (no drift), autogenerating a revision (body is `pass`), and
cycling `downgrade base` → `upgrade head`: both schemas and their rows survive
intact.

**Tests added:** 10 in `gsf/dal/pg/tests/test_session.py`, against a **real
database** rather than a `MagicMock` — mocking would prove only that
`.execute()` was called, where what has to hold is that a failed block leaves no
rows. Ports all five Neo4j transaction cases (commit, rollback, nesting reuses
the outermost, autocommit outside a scope, contextvar cleared after failure) and
adds four the pool makes newly possible to get wrong: concurrent threads get
genuinely separate transactions (a sibling's rollback must not discard committed
work), 48 concurrent reads do not exhaust the pool, `dispose_engine` is
idempotent and the engine rebuilds after it, and the URL is pinned to psycopg 3.

**Verified:** `upgrade head` on a blank database → 23 tables + `join_edge`;
`alembic check` clean; `downgrade base` leaves only `alembic_version` (the
schema is deliberately *not* dropped — the version table lives in it);
`upgrade head` again → 23; `join_edge` queryable; version table in `gsf`, not
`public`; `helm template` renders the new Job.

**Done criteria — met**, with one carried forward: the plan also asks that
`prisma db push` still succeed afterwards. Verified structurally (Alembic
provably cannot touch `public`, tested above) but **not** by running Prisma
itself, which needs the frontend toolchain. Worth doing once during the Phase 11
end-to-end pass.

**Not done here:** wiring `dispose_engine()` into `gsf.dal.close_store()`.
`gsf/dal/__init__.py` is being edited concurrently by the Phase 1 fork, and
touching it from two directions would conflict for no benefit. Fold it in when
Phase 1 merges.

> **ERD review gate.** PLAN.md marks this the highest-leverage checkpoint in the
> refactor: the three shapes below decide whether Phases 5–10 are mechanical or
> a rewrite. They are recorded here rather than left implicit, so disagreeing
> with one is cheap now and expensive later.
>
> - **`zone_target` polymorphism** — three nullable FKs +
>   `CHECK (num_nonnulls(...) = 1)`, not a `(kind, id)` pair. A pair cannot
>   carry a foreign key, so deleting a table would leave a dangling grant
>   behind: an access-control bug, not untidiness.
> - **`column_attribute_link`** — `HAS_ATTRIBUTE` and `SEMANTIC_FK` in one table
>   discriminated by `kind`, not two tables. `find_join_path` traverses them
>   together, and the `join_edge` view is far simpler over one table.
> - **`join_edge` view** — `SEMANTIC_FK` emitted in one direction only. That
>   asymmetry is why the Cypher needed `apoc.path.expandConfig`; here it is
>   simply a row that is not emitted.

**Next:** Phase 4 (catalog writes on Postgres) is the gate for all read phases,
but it depends on Phase 1's fork landing first.

---

## 2026-08-11 — Phase 3 — ERD review corrections

Two objections at the ERD gate, both upheld. See [DECISION-005](DECISIONS.md).

**1. `HAS_ATTRIBUTE` and `SEMANTIC_FK` are now separate tables** —
`column_has_attribute` and `column_semantic_fk` — rather than one table
discriminated by `kind`. They assert different things, their cardinality already
differs, and a discriminator inside the primary key blocks any constraint that
applies to only one of them.

**2. The claim that `SEMANTIC_FK` is "one direction only" was wrong**, and this
was the more serious of the two. It is *stored* one way (Column →
ColumnAttribute) but *read* both ways: `fetch_attr_column_contexts`
(`gsf/dal/attributes.py:151`) binds a ColumnAttribute and finds the columns
referencing it. The outgoing-only rule belongs to `find_join_path`'s traversal,
not to the edge. Left as written, anyone reusing that view for another
traversal would have silently lost half the edges.

The view is renamed `join_edge` → **`join_path_edge`**, after the single
function it serves, and its docstring now separates the storage direction from
the traversal restriction and points at `column_semantic_fk` for reverse reads.

**Migration regenerated** as `96b629fa2ae5` (replacing `75bdf1cdf36d`, deleted).
Regenerated rather than superseded by a follow-up revision because it had never
been applied outside a throwaway database — there was nothing to migrate from.

**Re-verified:** 24 tables + `join_path_edge`; `alembic check` clean;
`downgrade base` → 1 table; `upgrade head` → 24; view queryable; `public`
untouched.

---

## 2026-08-11 — Phases 1 + 3 merged

Phase 1 was executed in a separate worktree, in parallel with Phase 3, and
merged here. Conflicts were confined to `PROGRESS.md` and `DECISIONS.md` — both
append-only logs where each side had appended — and were resolved by keeping
both, Phase 1's entries first. No code conflicts: the two phases touched
disjoint files, which is why `dispose_engine()` was deliberately left unwired
from `close_store()` in Phase 3.

**Phase 1's claims verified independently after the merge**, not taken on
report:
- `grep -rnE "^\s*(from|import)\s+nemo_retriever\.tabular_data\.(ingestion|neo4j)" gsf dev_tools`
  returns exactly one line: the Phase-0 `test_constants.py` pin test, which
  imports the library on purpose.
- `TabularSchemaExtractOp` survives only in docstrings; no code references it.
- `_BatchEmbedActor` is named in exactly two files — `gsf/utils/embedding.py`
  and its test.
- The `gsf/catalog/` layout matches PLAN.md § Phase 1 exactly.
- **Fork equivalence holds:** rebuilding the fixture graph through the forked
  write path and replaying the goldens gives **128 passed** — all 122 recorded
  DAL reads reproduce byte-identically.

**Full suite: 397 passed, 4 skipped** (was 371 before the merge; +26 from
Phase 1). Ruff clean.

**Still open from Phase 3:** `dispose_engine()` is not yet called from
`gsf.dal.close_store()`. Now that both phases have landed there is no longer a
conflict risk — fold it in with Phase 4.

---

## 2026-08-11 — Phase 4 — write path behind a selector, and a silent data-loss bug

**Merge:** `origin/main` at `0a3ed24`, branch 0 behind. Nothing to merge.

### The write path is now backend-selected

Rewriting `store/` in place would have broken `GSF_STORE=neo4j` the moment
ingestion started writing to Postgres while the DAL still read Neo4j. So the
write path gets the same treatment the plan specifies for the DAL: `store/neo4j/`
holds the unchanged implementation, `store/<mod>.py` is a selector with an
explicit import list, and `store/pg/` lands alongside it.

- `gsf/infra/store.py` holds the `GSF_STORE` switch — not `gsf/dal/_store.py` as
  the plan sketched, because `gsf.catalog.store` selects on it too and
  `gsf.catalog` must not depend on `gsf.dal`.
- `store/connection.py` is **deleted** rather than made a selector: there is no
  Postgres counterpart to select between, that role belonging to
  `gsf/dal/pg/session.py`, whose pooled engine is a different shape. Its eleven
  importers name the Neo4j package directly now.
- Two implementation modules were importing siblings *through the selector*,
  which under `GSF_STORE=postgres` would have made the Neo4j path call the
  Postgres one. Fixed to direct sibling imports; the parity check normalises the
  two now-legitimate paths.

### The bug: no column change has ever survived a re-ingest

Phase 1 asked for incremental-diff tests before the rewrite. Writing them found
two stacked bugs — full reasoning in [DECISION-006](DECISIONS.md):

1. The column diff merges on `["database", "schema", "table_name", "column_name"]`.
   Measured: **neither** frame has `schema` (both call it `table_schema`) and
   **only** the graph side has `database`. Every merge raised `KeyError`.
2. `populate_db` runs schema updates through `executor.map(...)` and never
   consumes the iterator, so the exception sat in its future and was discarded.

Together: a column added to an existing table never reached the catalog, a
dropped column left a ghost the SQL generator would keep writing queries
against, a changed type stayed stale — silently, on a 24-hour schedule
reporting success. Table add/delete appeared to work only because it happens
*before* the raise.

Both bugs are upstream NeMo-Retriever's; the fork is verbatim.

**Fixed in the Neo4j implementation rather than reproduced in Postgres.**
Preserving behaviour means preserving what the system is meant to do, not
porting silent data loss so both stores can be wrong identically. Recorded as
**B6**: the next re-ingest of an existing database will apply every column
change accumulated since it was first catalogued, which for a long-lived
database is a large one-time diff. Consuming the executor's results also turns a
silent partial success into a loud failure — intended, but it will surface
sources that were quietly failing.

`gsf/catalog/store/neo4j/db.py` and `gsf/catalog/write.py` leave the
verbatim-fork set and are declared in `DIVERGED`.

**Tests added:** 12 in `gsf/catalog/tests/test_incremental_ingest.py`, each over
a throwaway source database it may freely mutate. Unchanged re-ingest is
compared on **node counts**, not names, because a duplicate-creating bug
preserves every name. Type change asserts the column keeps its **id**, since
recreating it would dangle anything referencing it.

**Suite:** 407 passed, 4 skipped (was 397). Goldens still replay clean —
the fix does not touch first-ingest behaviour, which is what they capture.

**Phase 4 is not complete.** What remains is the actual Postgres implementation
of `store/pg/*`. The scaffolding, the spec tests, and the ERD it will target are
in place; `edges.py` is the hard part, since its node/edge helpers are
label-generic and need a label→table mapping to work against a relational
schema.

---

## 2026-08-11 — Phase 4 — node shapes measured, schema corrected

Before writing `store/pg/*`, measured what the write path actually produces —
node labels, match properties and property bags — rather than inferring it from
the Cypher. Three findings, one of which corrects the Phase 3 schema.

**`is_nullable` is a string, not a boolean.** The graph stores `'YES'`/`'NO'`
(measured on the fixture: 99 / 110), and the DAL returns it raw.
`catalog_column.is_nullable` was declared `Boolean` in Phase 3; it is now
`Text`. A boolean column would hand callers `True`/`False` where they have
always received `'YES'`/`'NO'` — a read-contract change wearing the costume of a
type fix. It *should* become a boolean, but as a deliberate change with its
consumers updated, not as a side effect of the port. Migration regenerated as
`731be6d6d249`.

**`ordinal_position` is a genuine INTEGER** in the graph (209 of 209 columns),
even though the parser hands the write path the string `"3"` — something
coerces on the way in. The Phase 3 declaration was already right.

### Finding for Phase 10: every column exports as nullable

`gsf/dal/model_interchange.py:391` reads that string with
`is_nullable=bool(row.get("is_nullable", True))`. **`bool("NO")` is `True`**, so
the model export reports every column as nullable — `NOT NULL` columns
included — and an import of that document would then assert the wrong
constraint.

Not fixed here. It belongs to `model_interchange`, which Phase 10 ports, and
fixing it now would change export output with no export test to catch a
mistake — the two tests that cover `export_model` are the ones skipped since
Phase 0. **Phase 10 must fix this and un-skip those tests together.**

This is the third bug this refactor has surfaced without looking for bugs: the
first two were partitioned tables and matviews missing from Postgres catalogs
(B3) and column diffs never running (B6). All three were found by measuring
what the system does rather than reading what it says it does.

**Suite:** 407 passed, 4 skipped. `alembic check` clean; upgrade/downgrade/
re-upgrade verified against the corrected schema.

**Phase 4 remains incomplete** — `store/pg/*` is still to be written. The
foundations are in place: the selector scaffolding, the 12 incremental-diff
spec tests, the corrected ERD, and now the measured node shapes the registry
has to map. The next step is the label→table registry and the generic
`edges.py` upsert layer, which is where the `CONTAINS`-as-FK-column decision
stops being a schema detail and becomes code: `add_edges` must *update a child's
parent column* for `CONTAINS` while *inserting a row* for every other edge type.

---

## 2026-08-11 — Phase 4 — the generic write layer on Postgres

`store/pg/` now has its foundation: the label→table registry and the node/edge
upsert primitives everything else in the write path sits on. `db.py`,
`schemas.py`, `queries.py` and `indexes.py` are still to come.

- `store/pg/registry.py` — which table a label lives in, how identity is
  decided per label, and which relationships are rows versus parent columns.
- `store/pg/nodes.py` — `upsert_node` / `resolve_id`, the stand-in for
  `apoc.merge.node.eager`.
- `store/pg/edges.py` — the public surface `store/neo4j/edges.py` exposes,
  function for function, asserted by a parity test in the same file.

**20 tests**, run directly against the migrated schema rather than through the
selector, since the other modules have no Postgres side yet.

### Three things the property-graph assumption cost

**`CONTAINS` is not an edge.** It is the child's parent foreign key, so writing
one is an `UPDATE` of the child while every other relationship is an `INSERT`.
`add_edges` receives edges generically and has to branch on it. This is the bill
for the schema decision that turned `reset.py`'s APOC cascade into
`ON DELETE CASCADE`, and it comes due in exactly one place.

**Identity is not uniform.** `Table` and `Column` match on a pre-generated
`id`, `Schema` matches on `(database_name, name)` where `database_name` is not a
column on the row at all, and `Database` matches on `name`. Each shape is
handled explicitly. The natural keys are *parent-scoped* — a key that forgot the
parent would collapse `public` from every connected database into one row, which
is now a test.

**The stored id cannot be overwritten** — a deliberate divergence, and the one
worth reading twice. The Cypher replaces a matched node's id with the parser's
(`apoc.merge.node.eager(..., {id: $props.id})`). Harmless in a property graph,
where `id` is an ordinary property and relationships bind to internal nodes.
Here `id` is the primary key with other rows referencing it, and overwriting it
raises a foreign-key `IntegrityError` — found by a test, not by reasoning.

Nothing is lost by not reproducing it: the parser resolves ids out of the store
before writing, so for `Table` and `Column` the incoming id already *is* the
stored one. Only a `Sql`, matched by statement text, arrives with a fresh id —
and there keeping the stored one is correct rather than merely safe. Both
behaviours are pinned by tests.

**One Phase 3 decision came back:** `sql_query` is unique on
`md5(sql_full_query)`, not on the column, because statement text can exceed the
btree row limit. `ON CONFLICT (sql_full_query)` therefore matches no index and
raises. The registry now carries an optional expression conflict target.

**Suite:** 427 passed, 4 skipped.

**Still to do in Phase 4:** `store/pg/{db,schemas,queries,indexes}.py`, then
running the 12 incremental-diff tests and the fixture ingest under
`GSF_STORE=postgres`. The registry answers *where things go*; those modules are
the diffing and bulk-write logic on top of it.

---

## 2026-08-11 — Phase 4 — the Postgres store stops talking about nodes and edges

Review point: the Postgres implementation was carrying graph vocabulary into a
relational store. Correct, and worth separating into the part that was a choice
and the part that is a constraint.

**A choice, and the wrong one — now fixed.** `nodes.py`, `upsert_node`,
`LabelSpec`, `EdgeSpec`, `EDGES` were all internal to `store/pg/` with no caller
depending on them. Renamed:

| was | now |
|---|---|
| `store/pg/nodes.py` | `store/pg/rows.py` |
| `upsert_node` | `upsert_row` |
| `LabelSpec` / `LABELS` | `EntitySpec` / `ENTITIES` |
| `EdgeSpec` / `EDGES` | `LinkSpec` / `LINKS` |
| `label_spec` / `edge_spec` | `entity_spec` / `link_spec` |

**"Link", not "relation"** — in relational vocabulary a *relation* is a table,
so `RelationSpec` for an association would have meant the opposite of what it
described.

**A constraint, deliberately left.** `store/pg/edges.py` keeps `add_edges`,
`prepare_edge`, `prepare_node` and `get_node_properties_by_id`, because
`write.py`, `queries.py` and `schemas_parser.py` call them by those names and
the selector only works while both implementations match. Renaming half of a
two-sided contract is worse than either end of it. **Phase 11 renames the whole
surface in one change**, once the Neo4j side is deleted and the callers can move
with it.

The boundary is now explicit rather than incidental: graph words stop at
`store/pg/edges.py`, which says so in its docstring; `registry.py` and `rows.py`
behind it speak only of tables, rows, columns and links. The graph words that do
remain in the registry are the *keys* of the translation — the labels arriving
from callers — not descriptions of how anything is stored.

**Suite:** 427 passed, 4 skipped. No behaviour change; renames and docstrings
only.

---

## 2026-08-11 — Phase 4 — schema reads and writes on Postgres

`store/pg/schemas.py` and `store/pg/indexes.py` land. `db.py` (the incremental
diff) and `queries.py` remain.

**`indexes.py` is a no-op**, and that is the whole point of moving to a schema'd
store: the Neo4j version creates a uniqueness constraint and two indexes per
label on *every ingest*, because there is nowhere else to put them. Here they
come from a migration. Kept as a function only because `write.py` calls it
unconditionally and that call site is storage-agnostic code the fork does not
edit; Phase 11 removes both.

**`schemas.py` is where the frames have to match exactly.** Its readers feed
`update_diff_from_existing_schema`, which merges them against freshly parsed
frames — so the column names must line up, including the `database` column only
the stored side carries. Getting that wrong is not a type error, it is the diff
silently doing nothing, which is precisely the bug found earlier this phase
(DECISION-006). Verified: `columns_df` comes back with `database`,
`table_schema`, `table_name`, `column_name`, `id`, `data_type`, `is_nullable`
plus what `normalize_columns` adds — the same shape the Cypher produced.

**Schema change: `column_foreign_key` gains `last_seen`.** The Neo4j FK edge
carries it, and `delete_old_fks` uses it to remove keys an ingest did not see —
without it there is no way to distinguish a foreign key the source dropped from
one simply not re-asserted, and stale keys accumulate forever. Migration
regenerated as `fdafddfc335a`.

**`delete_old_fks` is scoped to the database being ingested.** The Cypher
matched from the `Database` node down, so a global delete would have dropped
another database's keys whenever two ingests overlap. Easy to miss when the
Cypher's scoping is implicit in the pattern rather than in a `WHERE`.

**Verified end to end** against the migrated schema: primary keys append (a
composite key arrives as several rows, so `add_pks` appends and `reset_pks`
clears first, matching `t.pk + [col.name]`); foreign keys upsert without
duplicating on re-ingest; a later stamp removes stale keys;
`load_schema_from_graph` round-trips.

**Suite:** 427 passed, 4 skipped.

**Left in Phase 4:** `store/pg/db.py` — the incremental diff, the largest and
most consequential of the modules, now specified by the 12 tests written before
the rewrite — and `store/pg/queries.py`. Then the whole thing runs under
`GSF_STORE=postgres` against the fixture.

---

## 2026-08-11 — Phase 4 — the re-ingest diff is now shared, not duplicated

`store/pg/db.py` lands. Only `store/pg/queries.py` remains before the whole path
can run under `GSF_STORE=postgres`.

**The diff was extracted rather than reimplemented.**
`update_diff_from_existing_schema` and the two `accumulate_*` helpers are pandas
over two frames plus calls back into the store — storage-agnostic, and now in
`gsf/catalog/diff.py`, with both backends re-exporting them. Its store calls go
through the selector, imported inside the function because the selector imports
the implementations which import the diff, and a top-level import would close
that circle.

Duplicating it was the obvious alternative and the wrong one: it is the subtlest
code in the write path, it was already silently broken for every re-ingest
(DECISION-006), and a second copy means the next fix has to be made twice or the
two drift. `store/neo4j/db.py` loses ~140 lines and keeps only its primitives.

Verified the extraction changed nothing: the 12 incremental-diff tests pass and
the fixture goldens still replay.

**Three places the schema shows through in the primitives:**

- `delete_table` and `delete_schema` no longer name their children. The Cypher
  had to `DETACH DELETE table, col`; here columns cascade.
- `db_exists` returns "has anything hanging off it". The Cypher counted
  *relationships* on the node; relationally the only thing that can hang off a
  database is a schema, so that is what is counted.
- `update_properties_in_graph_batch` keeps its `coalesce` on `description`, and
  that is the point rather than an incidental detail: a curated description has
  to survive a re-ingest that would otherwise overwrite it with whatever the
  source reports, which is usually nothing.

**One rough edge, handled explicitly:** `write.py` calls `update_node_property`
with the literal string `"db"`, not `Labels.DB`. The registry is keyed by label,
so `_spec_for` resolves case-insensitively with an alias — an unmapped label
would have silently skipped the `pulled` timestamp that marks an ingest
complete, and silence is the failure mode this phase has already been bitten by
twice.

**Suite:** 427 passed, 4 skipped.

**Left in Phase 4:** `store/pg/queries.py` (6 functions), then the fixture
ingest and the 12 incremental tests re-run under `GSF_STORE=postgres`. That last
step is the actual Done criterion — everything so far is tested against the
Postgres schema directly, not through the ingest pipeline end to end.

---

## 2026-08-11 — Phase 4 — `queries.py` needs a schema decision first

Reading `store/neo4j/queries.py` to port it surfaced a gap that is a design
question, not a translation. Stopping here rather than guessing at it.

**`sql_query` is missing five columns and one whole concept.** The module reads
and writes `nodes_count`, `join_count`, `union_count`, `total_counter`,
`last_query_timestamp` and `deleted`; the table has none of them. Those five are
ordinary additions.

**The concept is monthly counters.** `update_counters_and_timestamps_for_query_and_affected_data`
increments *dynamically named* properties — `count_2026_01`, `count_2026_02`,
one per month a query was seen — and `get_sql_counters` finds them by scanning
for the `count_` prefix. That is a schemaless pattern with no direct relational
form, and it needs a deliberate choice:

- **A child table** `sql_query_month_count(sql_query_id, month, count)`. Properly
  relational, queryable ("usage since March"), and the shape someone would pick
  from scratch. Changes the read shape, so `get_sql_counters` returns rows
  rather than a `count_*` dict — and its caller has to move with it.
- **A `jsonb` column.** Preserves the `{month: count}` shape exactly, so nothing
  above the store changes. But it reproduces the schemaless pattern inside the
  relational store, which is most of what this refactor exists to stop.

I would take the child table, with `get_sql_counters` returning the same dict it
does today so the change stays inside the store. But it is a schema decision
with a consumer-visible edge, so it belongs in a DECISIONS record made
deliberately rather than at the end of a long session.

**Two functions should not be ported at all** — they are storage-agnostic and
belong beside the diff in `gsf/catalog/diff.py` or a sibling:
`get_candidate_sql_ids` is pure pandas filtering, and `get_sql_counters` only
reads a node object. Duplicating them repeats the mistake the diff extraction
just corrected.

**Also worth flagging:** `update_counters_...` walks `apoc.path.subgraphNodes`
from a `Sql` node to stamp `last_query_timestamp` on every Table and Column it
touches, skipping any marked `deleted`. Nothing in the GSF schema writes
`deleted`, so that filter may be dead — worth confirming before carrying it
over, since a dead filter copied faithfully becomes a permanent puzzle.

### State at this stopping point

Working tree clean, everything committed, **427 passed / 4 skipped**.

Landed in Phase 4: the backend selector, `registry.py`, `rows.py`, `edges.py`,
`schemas.py`, `indexes.py`, `db.py`, and the shared `diff.py`. Plus two
pre-existing bugs found and fixed (DECISION-006) and one recorded for Phase 10.

Remaining: `store/pg/queries.py` behind the decision above, then the fixture
ingest and the 12 incremental tests under `GSF_STORE=postgres` — which is the
real Done criterion. Everything so far is tested against the Postgres schema
directly, not through the ingest pipeline end to end.

---

## 2026-08-11 — Phase 4 — the suggester was never ranking anything

Deciding how to store the per-month query counters turned into finding a fourth
bug. Full reasoning in [DECISION-007](DECISIONS.md); summary here.

The counters *are* read — `sql_attribute_suggester` ranks candidate SQL
expressions by them, which decides what the semantic layer proposes as
SqlAttributes. But the read has never worked: `_latest_3month_score` matched
keys against `^count_monthly_(\d{4})_(\d{2})$`, while `Query.__init__` writes
`count_{month}_{year}` — `count_8_2026`. Nothing has ever written a name that
regex accepts.

Verified directly: the scorer returns `0.0` for a real property bag from the
fixture and `42.0` only for a hand-made key nothing produces. So every
expression scored 0.0, sorting equal values changed nothing, and the ranking has
always been dict insertion order.

Two faults were latent behind it, and a faithful port would have inherited both:
the sort key was `(month, year)` read as `(year, month)`, so December 2025 would
have outranked January 2026 had the names ever matched; and
`fetch_terms_with_sqls`'s docstring documents the broken name rather than the
written one, which is probably where the mistake started.

**Now ranked by `total_counter`** — maintained on the same paths, a real
non-zero number, one integer column. Recorded as **B7**. It trades recency for
all-time usage, which the old code *intended* but never delivered; if recency
matters, `last_query_timestamp` is already a column and does not require
bringing back dynamic property names.

**This settles the schema question that started it:** no child table, no
`jsonb`, no per-month counters. `sql_query` needs five scalar columns and
nothing more.

**Tests added:** 5 in `gsf/semantic/tests/test_usage_score.py`, asserting
against a **real property bag**. Nothing caught this for the life of the feature
because no test used real key names — the test that would have caught it is the
one that asserts a non-zero score.

**PLAN.md** now carries a "Bugs this refactor has surfaced" table, since four is
no longer incidental. Three of the four are silent — no error, no log, no
failing test — which is the pattern: a schemaless store lets a name mismatch sit
undetected indefinitely, because nothing declares what a name is supposed to be.

---

## 2026-08-11 — Phase 4 — correction: the suggester fix is withdrawn

The previous entry recorded ranking by `total_counter` as landed behaviour
change **B7**. **That is withdrawn.** The fix was written, then deliberately
reverted on review: turning ranking on changes which SqlAttributes the semantic
layer proposes, and that is a product behaviour change in a component this
refactor is meant to leave untouched. It belongs in its own change with its own
tests, not as a side effect of a port.

`_usage_score` now returns a literal `0.0` — today's behaviour exactly — with
the cause, the two latent faults behind it, and the one-line fix written out in
its docstring. A constant is honest where a plausible-looking scorer that
silently does nothing is not.

**This is the opposite call to DECISION-006, and the distinction is the point.**
That bug silently lost data: columns vanished from the catalog, so reproducing it
faithfully would have meant porting data loss. This one produces an arbitrary
ordering of suggestions — wrong, but not destructive, and safe to leave until it
can be changed on purpose. The test for whether to fix a bug mid-port is whether
preserving it destroys something.

`gsf/semantic/tests/test_usage_score.py` pins the inert behaviour rather than
the fix, and says so: every assertion in it inverts when ranking is turned on,
which is the signal it worked rather than a regression.

**What still stands from the investigation:** per-month counters are dropped
from the schema entirely. Nothing can read them, and `total_counter` — the
column a fix would use — is already there. `sql_query` needs five scalar
columns, no child table, no `jsonb`.

**Suite:** 431 passed, 4 skipped.

---

## 2026-08-11 — Phase 4 — **the catalog now ingests into Postgres**

`store/pg/queries.py` lands, and with it the whole write path runs under
`GSF_STORE=postgres`. A full ingest of both fixture databases produces a catalog
whose counts match the Neo4j graph exactly:

| | Neo4j | Postgres |
|---|---|---|
| databases | 2 | 2 |
| schemas | 3 | 3 |
| tables | 37 | 37 |
| columns | 209 | 209 |
| foreign keys | 30 | 30 |
| `base table` / `view` / `materialized view` | 28 / 8 / 1 | 28 / 8 / 1 |

Zero orphans at every tier. `film.pk` is `['film_id']`, `film.rating` is
`mpaa_rating` (the B3 fix), `ordinal_position` is an integer.

### Three structural mismatches the ingest exposed

None was visible from reading the code; all three only appeared under a real
ingest.

**Labels arrive as lists.** `merge_schema_nodes` passes `["Table"]`, not
`"Table"` — `apoc.merge.node.eager` takes a list because a node can carry
several labels. `entity_spec` now accepts either and takes the first, matching
`labels(n)[0]` as used everywhere else.

**Rows were being created before their parents existed.** `add_schema` creates
every Table and Column node first and links them afterwards, because a property
graph lets a node exist with no relationships. Relationally it cannot:
`schema_id` and `table_id` are `NOT NULL`, so a table with no schema is not a
row that can be written at all.

Rather than make the parent columns nullable — trading a real integrity
guarantee for the convenience of matching the old call order —
`merge_schema_nodes` now *holds* each row's properties and `merge_schema_edges`,
where the parent id finally appears, does the insert. Stateful, and documented
as such: the properties genuinely arrive before the parent does, and something
has to bridge that.

**`Sql` points at columns as well as tables.** 16 `Sql -[:SQL]-> Column` edges
against 9 to tables, so `sql_query_column` joins `sql_query_table`. Two
association tables rather than one polymorphic one, because the two are always
read separately — `load_sqls_to_tables` returns them as distinct lists and
deduplication compares them as distinct sets.

### Two things deliberately not reproduced

**Per-month counters.** Nothing can read them (DECISION-007), so writing them
would mean reviving dynamic property names in a relational store to hold data no
reader can parse. `total_counter` is kept and is what a fix would use.

**The `deleted` filter.** Both Cypher reads guard with
`coalesce(node.deleted, false) = false`, and nothing in GSF writes `deleted` —
verified across the codebase. The guard is always true, so dropping it preserves
behaviour, where carrying a column no writer sets would leave a permanent
puzzle.

**Also shared, not duplicated:** `get_sql_counters` and `get_candidate_sql_ids`
are pure, so they moved to `gsf/catalog/query_stats.py` alongside the same
treatment given to the diff.

**Suite:** 430 passed, 4 skipped.

### What Phase 4 still owes

The 12 incremental-diff tests run against Neo4j only — they assert by querying
the graph directly. Re-running them under `GSF_STORE=postgres` needs them
parameterised over the backend, and that is the remaining gap between "a first
ingest is correct" and "re-ingest is correct", which is where the bugs were.

---

## 2026-08-11 — Phase 4 — **the gap is closed: re-ingest verified on both backends**

The 12 incremental-diff tests now run against **whichever backend the process
was started with**, and pass on both:

```
== GSF_STORE=neo4j ==      13 passed
== GSF_STORE=postgres ==   13 passed
```

That closes the distance between "a first ingest is correct" and "re-ingest is
correct" — and re-ingest is where every bug in this phase was.

**How.** Assertions go through `gsf/catalog/tests/catalog_inspector.py`, which
answers the same question of either store — *what does the catalog contain* —
in the vocabulary of the catalog rather than in labels-and-nodes or rows-and-
joins. The test file is therefore one specification, not two that could drift.

**Why it needs two processes.** `GSF_STORE` is read once at import, deliberately:
a value that could change mid-process would let one request read Neo4j and the
next read Postgres. So one process exercises one backend, and covering both
means running twice. `make test-stores` does it, and the test module says so in
its docstring: **CI has to run both — one passing does not imply the other.**

A 13th test asserts the configured backend is a known one, so a run that
silently exercised the default twice fails rather than reporting false
coverage.

### Phase 4 Done criteria — met

- `/ingest` of both fixture databases populates `gsf.*` ✅ (2 databases, 3
  schemas, 37 tables, 209 columns, 30 foreign keys — identical to Neo4j)
- row counts match the Phase-2 node counts per label ✅
- views and the materialized view carry the right `table_type` ✅ (28 / 8 / 1)
- re-ingest is idempotent ✅ (asserted on counts, both backends)
- added / renamed / dropped table and column are diffed correctly ✅ (both
  backends)
- delete cleans up one database via cascade without touching the other ✅

**Suite:** 431 passed, 4 skipped.

### What Phase 4 leaves for later, explicitly

- `dispose_engine()` is still not wired into `gsf.dal.close_store()`. It was
  deferred in Phase 3 to avoid conflicting with the Phase 1 fork; there is no
  longer a conflict.
- The **reads** are untouched — `gsf/dal/*` still queries Neo4j under both
  settings. `GSF_STORE=postgres` currently means "write the catalog to
  Postgres", not "run on Postgres". Phases 5–10 close that, and the goldens are
  the oracle for it.

---

## 2026-08-11 — Phase 5 — zone scoping ported, and the filter contract settled

**Merge:** `origin/main` at `0a3ed24`, branch 0 behind. Nothing to merge.

`gsf/dal/users.py` and `gsf/dal/zones.py` move to `gsf/dal/neo4j/`, and
`gsf/dal/<domain>.py` becomes a selector with an explicit import list — the
structure the plan specifies, now applied to the first two domains.
`gsf/dal/pg/users.py` is implemented; **`gsf/dal/pg/zones.py` is not yet.**

### The filter contract

Settled as [DECISION-008](DECISIONS.md): the Postgres `resolve_table_filter`
takes a SQLAlchemy column and returns a predicate, where Neo4j returns a Cypher
`WHERE` string. What made that safe to decide was measuring the callers — all
six are inside `gsf/dal` and are ported per-phase, so the two shapes never meet.
`resolve_accessible_catalog_ids`, which *is* called from the service layer,
returns plain id sets and is unchanged.

Returning a SQL string would have preserved the signature exactly and thrown
away the reason Core was chosen over raw SQL.

### Four round trips become one query

`get_accessible_catalog_ids_for_zones` was four sequential Neo4j reads —
direct grants, then expand down from databases, then from schemas, then up from
tables. It is now one statement with three CTEs.

**The expansion is one level from each direct grant, and not transitive**, which
is the part worth getting right: granting a table admits its schema so the tree
can be drawn, but must *not* then admit that schema's other tables. Every CTE
branch reads from `granted` rather than from another CTE, which is what keeps it
non-recursive — and there is a test whose failure message says a sibling table
leaked.

### Tests

13 in `gsf/dal/pg/tests/test_users.py`, over a fixture of two databases × two
schemas × two tables — enough shape that "everything" and "the right subset" are
different answers, where a single-table fixture would pass a filter that matched
everything. Each expansion rule is asserted separately rather than in aggregate,
because this is the access-control boundary: a mistake here is not a wrong
answer, it is one user seeing another's data.

Two of them pin the trap named in DECISION-008: `zone_ids=None` is *unscoped*
and `[]` is *nothing granted*, and conflating them turns a locked-down user into
an administrator.

### Two fixes to the surface freeze

- **Selectors appeared to have no public surface.** `_public_surface` filtered
  on `__module__`, which for a re-exported function points at the implementation
  — so the freeze silently stopped guarding every module it was written for the
  moment that module got a selector. Now `__all__` counts too.
- **`test_postgres_and_neo4j_surfaces_match` went live** for the first time and
  immediately caught DECISION-008's divergence, which is the test doing its job.
  It now compares only domains present in *both* backends, and carries a
  narrow `BACKEND_SPECIFIC` allowlist where annotations may differ but
  **parameter names may not**.

**Suite:** 445 passed, 3 skipped — up from 431/4, the difference being that the
backend-parity test now runs instead of skipping.

**Next:** `gsf/dal/pg/zones.py`, then a cross-backend equivalence check for both
against the same fixture.

---

## 2026-08-12 — Phase 5 complete — zones ported

`gsf/dal/pg/zones.py` lands, finishing Phase 5.

### A wrong constraint, caught by porting

`zone.name` was declared `UNIQUE` in Phase 3. Reading the Cypher's
`_zone_name_exists` showed that is wrong in **both** directions at once:

- the rule is **case-insensitive on the trimmed name** (`toLower(trim(z.name))`),
  so a `UNIQUE` column would *admit* "Sales" alongside "sales", which the
  application rejects;
- the rule is **scoped to zones sharing a database** with the zone's items, so a
  `UNIQUE` column would *reject* the same zone name in two unrelated databases,
  which the application allows.

A constraint wrong in both directions is worse than none, so it is removed and
the check stays in application code exactly as today. Expressing it in the
database would need a partial index on `lower(trim(name))` *per database*, which
zone membership — many-to-many through `zone_target` — cannot express. An index
on `lower(name)` supports the lookup. Migration regenerated as `9ceb971cfdcb`.

Both directions are now tests: one asserts `"  NAME  "` collides with `"name"`,
another asserts the same name in a different database does not.

### Polymorphic targets

Callers pass a flat list of ids without saying whether each is a database,
schema or table — a graph never needed to know, and three nullable foreign keys
very much do. `_classify_items` resolves each id to its tier before writing it
to the right column, and unknown ids are dropped rather than rejected, matching
the Cypher's `MATCH` finding nothing and moving on.

### Two distinctions that would be easy to flatten

- **`item_ids=None` vs `[]` on update.** `None` means "not editing membership";
  `[]` means "clear it". Collapsing them would strip a zone's items on any
  metadata-only edit.
- **A disabled zone stays visible.** The label swap is a boolean column now, but
  `list_zones` still returns disabled zones — they are administered, just not
  granting.

### Tests

15 in `gsf/dal/pg/tests/test_zones.py`, plus the 13 for users. One asserts that
deleting a table removes its zone membership by cascade — the reason
`zone_target` uses three real foreign keys rather than a `(kind, id)` pair,
since a dangling grant is an access-control bug rather than untidiness.

**Suite:** 460 passed, 3 skipped.

### Phase 5 Done criteria — met

`resolve_accessible_catalog_ids` / `resolve_table_filter` are ported and the
filter contract is settled once ([DECISION-008](DECISIONS.md)), which is what
the phase existed to do. Zone CRUD is ported. The four round trips in
`get_accessible_catalog_ids_for_zones` collapse to one query.

**Still owed, carried since Phase 3:** `dispose_engine()` is not yet called from
`gsf.dal.close_store()`.

**Next:** Phase 6, `datasources` — the largest blast radius after
`model_interchange`, and where `fetch_schemas_by_ids` feeds `Schema` and
therefore every SQL validation.

---

## 2026-08-12 — Phase 6 — scaffolding, and the carried-over engine disposal

**Merge:** `origin/main` at `0a3ed24`, branch 0 behind. Nothing to merge.

**`dispose_engine()` is finally wired into `gsf.dal.close_store()`** — deferred
since Phase 3 to avoid conflicting with the Phase 1 fork, and then simply
carried. All three connections are now closed **unconditionally, whatever
`GSF_STORE` says**: the setting decides which store is *used*, not which
connections a process has opened, and a run that touched both would otherwise
leak one backend's sockets. Closing what is open is cheaper than reasoning about
what should be.

`gsf/dal/datasources.py` moves to `gsf/dal/neo4j/` behind a 26-function
selector. **`TABLE_COUNTS_SUBQUERY` is deliberately not re-exported**: it is
Cypher, and the selector carries only the backend-neutral function surface.
`gsf/dal/exploration.py` imports it from the Neo4j implementation directly until
Phase 9 ports that module and the constant goes with it.

Three test files patched `gsf.dal.datasources.graph`, which the selector does
not have; they now patch the implementation. Worth noting because it is the
predictable cost of the selector pattern — a test that patches *where a function
lives* breaks when the module becomes a re-export, and the fix is always to
patch the implementation rather than the seam.

**Suite:** 461 passed, 3 skipped.

**Next:** `gsf/dal/pg/datasources.py` — 26 functions, the largest blast radius
after `model_interchange`, and the one where `fetch_schemas_by_ids` feeds the
`Schema` model and therefore every SQL validation.

---

## 2026-08-12 — Phase 6 — dependency analysis before implementing

Measured which of `datasources`' 26 functions can be *verified* now, rather than
discovering it partway through writing them.

**21 need only the catalog tier**, which Phase 4 already populates in Postgres:
`fetch_databases`, `fetch_schemas_for_database`, `fetch_all_schema_ids`,
`fetch_schema_ids_for_database`, `fetch_schemas_by_ids`, `fetch_sorted_tables`,
`fetch_table_by_id`, `fetch_table_by_name`, `fetch_tables_by_ids`,
`fetch_join_neighbors`, `fetch_join_edges`, `count_columns_for_table`,
`fetch_parent_table_id_for_column`, `fetch_table_context`,
`fetch_col_table_contexts`, `store_column_sample_values`,
`store_column_uniqueness`, `apply_metadata_batch`, `patch_catalog_node`,
`fetch_node_properties_by_id`, `fetch_item_by_id`.

**5 join to the semantic tier** — `Term`, `ColumnAttribute`, or the description
fallback through them: `fetch_tables_for_schema`, `fetch_all_tables_without_term`,
`fetch_columns_for_table`, `fetch_tables_and_columns_by_node_ids`,
`fetch_bridge_table_candidates`.

Those five are *implementable* now — the tables exist from Phase 3 — but they
are **not verifiable** until Phase 7 writes terms and attributes, because until
then every semantic join returns empty and a wrong join reads exactly like a
correct one against no data. Their golden captures are non-trivial precisely
because the fixture *has* semantic data.

**This is worth flagging against PLAN.md's ordering.** The plan sequences 6
before 7 and notes only that 6 has "the largest blast radius after
`model_interchange`". It does not record that a fifth of the module cannot be
checked until 7 lands. Two workable options, and the choice belongs to whoever
picks this up:

1. **Implement all 26 now, verify 21, and re-verify the remaining 5 during
   Phase 7.** Keeps the plan's order; costs a second verification pass and
   leaves five functions carrying unearned confidence in between.
2. **Implement the 21, and move the 5 into Phase 7** where their data and their
   goldens arrive together. Phase 6 then genuinely completes, at the cost of
   `datasources` being finished across two phases.

I would take (2): a phase that ends with everything it produced actually checked
is worth more than one that ends on schedule. But it changes the phase
boundaries the plan defines, so it wants a DECISIONS record rather than a
quiet reinterpretation.

**Nothing implemented this session beyond the scaffolding** — stopping at a
clean point rather than writing 700 lines that could not be verified before the
session ended.

**Suite:** 461 passed, 3 skipped. Working tree clean.

---

## 2026-08-12 — Phase 6 — correction: the split is 20/6, not 21/5

The previous entry and DECISION-009 both said 21 catalog-tier functions and
five semantic-dependent ones. **`fetch_tables_by_ids` belongs in the second
group**, so it is 20 and six. Both documents are corrected in place, with the
reasoning kept in DECISION-009 rather than erased.

`_FETCH_TABLES_BY_IDS` calls `column_description_expr`, the ColumnAttribute
description fallback — so the function reads the semantic tier and, like the
other five, cannot be verified until Phase 7 writes it.

**How two analyses agreed on the wrong answer**, which is the part worth
keeping. The first script resolved each function's query constants with a regex
that stopped at the first line beginning with a letter; every one of these
constants is a Cypher string whose second line is `MATCH`, so it read one line
of each and found nothing. The second recursed properly but guarded
self-reference with "the constant's text does not start with its own name" —
true of every assignment, so the recursion never ran and it reproduced the first
answer exactly.

Two independent bugs producing the same number is precisely how a wrong figure
survives a second look. What caught it was reading the actual Cypher while
starting to port it, not the analysis.

**Nothing implemented yet.** The corrected boundary is what Phase 6 builds
against.

---

## 2026-08-12 — Phase 6 — the catalog reads run on Postgres

`gsf/dal/pg/datasources.py` implements the 19 catalog-tier functions. The seven
that reach the semantic tier raise `NotImplementedError` naming Phase 7 and the
reason — a loud failure rather than a query that quietly returns nothing.

**Phase 6's Done criterion is met**: a `Schema` map built from Postgres resolves
real SQL. `validate_sql` succeeds on `SELECT film_id FROM film` and on
`SELECT customer_id, amount FROM payment` — the second being the partitioned
table that was invisible to the catalog entirely until B3.

Verified against the ingested fixture: `fetch_databases` → chinook (1 schema),
pagila (2); `fetch_schemas_for_database` → analytics 3 tables, public 23;
`fetch_schemas_by_ids` → 209 column rows; `fetch_sorted_tables` → 37. Writes
and `patch_catalog_node` round-trip.

### Four shapes that needed deciding rather than translating

**Empty parents disappear.** `MATCH (db)-[:CONTAINS]->(s)` is an inner join, so
a database with no schemas produced no row at all rather than a row with zero.
Reproduced with inner joins; `fetch_schemas_for_database` still returns `None`
for a database with no tables, which is what its caller treats as absent.

**`fetch_schemas_by_ids` reads an empty id list as *everything*.** The Cypher's
`WHERE size($schema_ids) = 0 OR ...`. Reading it as "nothing" would leave every
query unresolvable — and silently, since an empty catalog map produces "no table
known" rather than an error.

**`apply_metadata_batch`'s `coalesce` direction is the point.** `coalesce(new,
existing)` means a curated description survives a batch that has nothing to say
about it. Tested by running a second batch with `description: None` and
asserting the first one is still there.

**`catalog_database` has no `description` column.** The Cypher read
`db.description`, a property nothing writes, so the value was always `None`. The
key stays in the returned shape because callers read it; there is nothing to
read it from.

### One deliberate improvement

`fetch_sorted_tables` now breaks ties on `name`. The Cypher ordered by
`query_count DESC` alone and nearly every table ties at zero, so a function whose
name promises an order returned rows in whatever order the store felt like —
differing between calls. A tiebreaker cannot break a caller that was already
receiving an arbitrary order. This settles the finding raised during the Phase 2
golden capture, which asked Phase 6 to decide.

`fetch_table_by_name` gets the same treatment for the same reason: it takes
`LIMIT 1` over a name that is not unique across schemas or databases, so it now
orders by id. Which row wins is still arbitrary; it is at least the same
arbitrary row each time.

**Suite:** 461 passed, 3 skipped.

**Next:** Phase 7 — the semantic core, plus the seven inherited functions. Port
the description fallback (`column_description_expr` / `table_description_expr`
→ `sql_fragments`) first: five of the seven need nothing else.

---

## 2026-08-12 — Phase 7 — the description fallback

`gsf/dal/pg/sql_fragments.py` — the Postgres counterpart to
`cypher_fragments.py`, and the thing five of the seven functions Phase 6 handed
forward (DECISION-009) were actually waiting on.

**Two of the four Cypher helpers have no counterpart, deliberately.**
`and_condition` existed because the Cypher builders returned either `""` or a
whole `WHERE ...` clause, so a caller narrowing a query further could not
concatenate; SQLAlchemy composes predicates natively, so the problem does not
arise. `paging_clause` becomes `.offset()` / `.limit()`. Its docstring's warning
survives the port unchanged and unsolved: without a total `ORDER BY`,
consecutive pages both repeat and drop rows. That hazard is identical in SQL and
belongs to each paged query, which must order by something unique — it is not
something this module can fix on the caller's behalf.

**`head([...])` becomes `ORDER BY id LIMIT 1`.** The Cypher took whichever
element came first from a list with no defined order, so a column with two
attributes could be described differently between two calls. Which attribute
wins is still arbitrary; it is now at least *stably* arbitrary. Same reasoning
as `fetch_sorted_tables` in Phase 6.

### The bug the tests caught

The first cut took a column **id** rather than a table alias, and read the
column's own description back out with a second `SELECT` on `catalog_column`.
That correlates the table with itself: `WHERE catalog_column.id =
catalog_column.id` is a tautology matching every row. Postgres raised
`CardinalityViolation` — loudly wrong rather than quietly wrong, which is the
good outcome, but only because 11 of the 12 tests exercised it. Both helpers now
take the table or alias the caller is already selecting from.

### Tests — `gsf/dal/pg/tests/test_sql_fragments.py`, 12 of them

Pinning three things that each fail silently:

* **precedence** — own description, then `HAS_ATTRIBUTE`, then `SEMANTIC_FK`.
  Not alphabetical: an attribute a column *is an instance of* describes it
  better than one it merely *references*. One test asserts the order directly by
  attaching both and checking which wins.
* **blank counts as missing** — `trim(x) <> ""` in the Cypher. An empty string
  is what a UI leaves behind when someone deletes text, so treating it as
  present would make the fallback useless for exactly the rows a user has
  touched. Covered on the column, on the attribute, and on the table.
* **correlation** — a sibling column's attribute is not borrowed. This is the
  test that would have caught the cardinality bug as a wrong answer had Postgres
  been willing to return one.

**Suite:** 461 passed, 3 skipped.

### Two corrections to the record

**Phase 6's entry says 461 passed; it was 449.** Measured at `HEAD~1`, before
this commit. The 461 written there was never true — the 12 tests that make the
number 461 today are the ones added *above*. The coincidence is unlucky enough
to be worth naming, because it would otherwise read as "Phase 7 added no tests".

**How to actually run the suite.** Both facts below were reconstructed from
`docker inspect` rather than read anywhere, which cost more time than writing
them down would have:

```
docker run -d --name gsf-golden-neo4j -p 7475:7474 -p 7688:7687 \
    -e NEO4J_AUTH=neo4j/goldenpass ...
docker run -d --name gsf-golden-pg -p 55432:5432 ...

export NEO4J_URI=bolt://localhost:7688 NEO4J_USERNAME=neo4j \
       NEO4J_PASSWORD=goldenpass
export POSTGRES_HOST=localhost POSTGRES_PORT=55432 POSTGRES_USER=postgres \
       POSTGRES_PASSWORD=goldenpass POSTGRES_DATABASE=gsf_alembic
uv run pytest gsf/ -q
```

Without these the suite still exits green — at **323 passed, 141 skipped**. The
141 are the golden replays and the incremental-ingest tests skipping themselves
because they cannot reach a store. A green run that has quietly stopped
exercising the thing being refactored is the failure mode to watch for here;
check the skip count, not just the exit code.

---

## 2026-08-12 — Phase 7 — the seven inherited functions

`gsf/dal/pg/datasources.py` is complete: 26 of 26. The seven Phase 6 deferred
(DECISION-009) now run on Postgres, and `_PHASE_7` and its `NotImplementedError`
stubs are gone.

They were held back for want of terms and attributes to join to, and that was
the right call — three of the findings below are things that look correct
against an empty semantic tier.

### A missing column, found by porting rather than by reading

**`description_certified` does not exist on `catalog_table` or
`catalog_column`.** The UI's certification checkbox PATCHes it through
`patch_catalog_node`, which deliberately drops properties with no column. So on
Postgres the request returns 200, the checkbox ticks, and the flag is gone by
the next read — no exception, no log line, nothing red in CI.

Added by migration `325e4825d9ee`, `NOT NULL DEFAULT false`. Recorded as
[DECISION-010] / B8, including why the initial migration was not amended: it is
already applied everywhere, and rewriting an applied migration makes `upgrade
head` a silent no-op for anyone who ran it.

Worth generalising: **a property only a user can create is invisible to a
fixture nobody has used.** Nothing in the ingest writes this, so it is absent
from Pagila, absent from the golden capture, and absent from the write path the
schema was built by reading. Checked the neighbours while there —
`description_suggestion` and the other `*_certified` flags do have columns.

### The bug the term count was hiding

`_terms_count` has to count the *deduplicated union* of two routes to a Term: a
table `REPRESENTS` one directly, and reaches others through its columns'
attributes. Written as a `UNION` of id lists wrapped in `.subquery()` to be
counted, it silently stopped correlating — two levels between the leg predicates
and `catalog_table` is one too many for SQLAlchemy's auto-correlation — and
**every table reported the same total**.

Rewritten as "count Terms reachable by any route", with `.correlate()` spelled
out. The explicit calls are load-bearing for the same underlying reason:
auto-correlation only considers the immediately enclosing `SELECT`, and that one
selects from `term` alone, so left alone it adds a second unconstrained
`catalog_table` to each `EXISTS`.

The failure was invisible on the first fixture, where the one table under test
happened to have the same count either way. It is caught now by asserting a
table with **no** terms alongside one with two — the assertion that makes an
uncorrelated subquery impossible to pass.

### `fetch_bridge_table_candidates`, and why Pagila cannot test it

It returns **nothing** on Pagila. That is correct: `film_actor` and
`film_category` each carry a `last_update` column, so no table has a foreign key
on every column. The golden capture agrees — Neo4j recorded `[]` too.

Two empty lists agreeing is not evidence. The bridge fixture is therefore built
by hand, and covers the parts that would otherwise be asserted by nothing: the
`SEMANTIC_FK` direction (Column → ColumnAttribute ← Column, so the *owning*
column is the join target — read backwards, every bridge points at itself),
self-referential bridges, the `source = 'bridgeTable'` filter that stops it
re-proposing its own output, and the FK that resolves to nothing because its
target table was never ingested.

That last one is why the Cypher checked "every column resolves" twice, in two
different ways. The checks are not redundant — `ALL(c IN cols ...)` asks whether
each column has an outgoing edge, `size(fk_pairs) = size(cols)` asks whether each
edge lands somewhere real. Both are kept.

### Smaller things, each preserved deliberately

* **`fetch_columns_for_table` runs two queries, not one.** The header decides
  whether the table exists; the page decides which columns come back. A single
  join with `OFFSET` would return nothing for a page past the last column, and
  the caller would read "page 3 of a 2-page table" as "no such table".
* **Page order gets `id` as a tiebreaker.** `ordinal_position` is nullable and
  not unique, so on its own it is not a stable page order: two columns sharing a
  position can swap between pages, showing one twice and the other never.
* **`fetch_tables_and_columns_by_node_ids` uses the description fallback for
  columns but not for tables.** Asymmetric in the Cypher. Preserved, because
  these frames feed embeddings and widening what gets embedded would move
  retrieval results with nothing failing.
* **`fetch_tables_by_ids` still drops a table with no columns** — the Cypher's
  second `MATCH` was an inner join. A column-less table is a symptom worth
  seeing where it originates, not something to paper over in a read.
* **`fetch_tables_for_schema` ignores its `database_name` argument**, as before:
  schema ids are globally unique, so it can only agree or contradict. Pinned by
  a test, because the next reader is otherwise right to assume it filters.

### A finding about the grading mechanism itself

`test_golden.py` describes itself as the fidelity oracle — "Phases 5-10 are
graded by this file". **It cannot currently grade Postgres at all.**

Run with `GSF_STORE=postgres`, 49 of 128 fail, including functions that Phase 6
landed and verified. The cause is not the DAL:
`capture_dal_golden._fixture_ids()` resolves every fixture entity with
hardcoded Cypher, so under Postgres it hands *Neo4j* ids to the Postgres DAL and
essentially every read returns nothing. A second leak shows up in the same run —
`gsf/dal/neo4j/datasources.py` formats a zone filter into a Cypher string, and
under `GSF_STORE=postgres` it receives the Phase 5 predicate object and
interpolates `None` into the query.

Neither breaks production, where one backend is selected and the facade
dispatches to it. Both break the parity check, which is the thing the plan is
counting on.

Nobody noticed because every phase so far was verified by hand-written tests
instead — which is why the count of those matters and why this entry is not
claiming parity it has not measured. **Porting `_fixture_ids` to be
backend-aware is a prerequisite for the Phase 11 gate**, and it cannot be
finished before the semantic tier is seeded in Postgres, since it also resolves
terms, zones, and analyses. Added to PLAN.md as explicit Phase 11 work rather
than left as a surprise.

### Tests — `gsf/dal/pg/tests/test_datasources_semantic.py`, 34 of them

A hand-built catalog (`World`) rather than the Pagila ingest, shaped so each
join has something to get wrong: a Term reachable by both routes, a column whose
description comes from an attribute, a genuine junction table, and an FK whose
target is deleted out from under it.

Two of the three initial failures were the tests being wrong, and one is worth
repeating: **`ORDER BY name` was compared against Python's `sorted()`**, which
disagrees with Postgres. `order_tag` sorts before `orders` by codepoint and after
it under a collation that ignores punctuation at the first level. The contract is
the database's order, so the test now compares against the database.

**Suite:** 495 passed, 3 skipped (was 461).

---

## 2026-08-12 — Phase 7 — attributes, and the join-path spike

`gsf/dal/pg/attributes.py` — 8 functions, including `find_join_path`, which
PLAN.md carved out as its own 2-3 day spike. `gsf/dal/attributes.py` becomes a
selector and the Cypher moves to `gsf/dal/neo4j/attributes.py`.

The move fixed a backend leak on the way past: the Neo4j module imported
`fetch_col_table_contexts` from the **selector**, so under
`GSF_STORE=postgres` the Cypher implementation would have called the Postgres
one. It now imports from `gsf.dal.neo4j.datasources` directly. This is the same
class of bug as the `resolve_table_filter` leak recorded in the previous entry —
worth checking for in every module still to be ported.

### `find_join_path`

Level-at-a-time BFS from Python over the `join_path_edge` view, with the
recursive CTE kept beside it as `JOIN_PATH_CTE_SQL` and used as a test oracle —
exactly as the plan called for. The CTE is correct and is *not* what APOC did:
its visited array is per-path, so it re-expands every distinct route to a node,
where `uniqueness: 'NODE_GLOBAL'` visits each node once. On a hub attribute
referenced by hundreds of columns that difference is the whole ballgame. A test
counts queries to pin it.

**The depth bound was nearly a silent regression.** `maxLevel: 30` looks
generous when you know real join paths are 2-4 hops, so 10 seemed like a safe
tightening that would make runaway traversals fail fast. It is not: **a hop is
four edges, not one.** Getting from one FK column to the next runs
`Column -SEMANTIC_FK-> ColumnAttribute -HAS_ATTRIBUTE-> Column -CONTAINS->
Table -CONTAINS-> Column`, so an n-hop path is `4n - 2` edges and an ordinary
4-hop path is 14. A ceiling of 10 would have returned "no path" for it, silently
and only on the longer paths. Caught by the depth-bound test failing on a
3-table chain, which was the assertion doing its job in the least convenient
way. `MAX_PATH_DEPTH` is 30, unchanged from the Cypher.

Also fixed while writing the oracle: `:anchor::text` inside a SQLAlchemy
`text()` is a syntax error, because `::` collides with bind-parameter parsing.
`CAST(:anchor AS text)`.

### A filter that belonged in the JOIN, not the WHERE

`fetch_attr_column_contexts` takes an optional `database_name`. Written as a
`WHERE` over the outer join, it **deleted the attribute's row entirely** when
its column belonged to another database. The Cypher put the condition on an
`OPTIONAL MATCH`, so the attribute survived with a null database and the
empty-string defaults took over.

The difference is invisible to a caller that uses `.get`, and the function
swallows exceptions and returns `{}` — so the first version failed by returning
an empty dict rather than by raising. It is caught here by asserting the key is
present, not by asserting the values.

### One divergence, left to surface

`update_column_attribute` renames an attribute, and `name` is part of the 5-part
merge key — a **real unique constraint** in Postgres where Cypher had only a
`MERGE` pattern. The graph let a later `SET` produce two attributes identical on
all five properties, which the next merge would then match arbitrarily. Renaming
onto an existing key now raises `IntegrityError`.

Not caught, deliberately: the constraint is describing a genuine conflict, and
swallowing it would put the ambiguity back. Documented on the function.

### Tests — `gsf/dal/pg/tests/test_attributes.py`, 31 of them

The one that matters most: **two columns referencing the same attribute must not
join to each other.** Traversing `SEMANTIC_FK` backwards walks up from one FK
column and down another, producing a join between two `customer_id` columns that
have nothing to do with each other — a wrong answer that reads as obviously
right. The view simply does not emit the reverse row, and a test says so.

Alongside it: the `-Schema` exclusion (two tables in one schema, unconnected,
must not find each other), cycle termination, cross-database rejection through a
shared attribute, hop pairing on a real 2-hop path, and agreement with the CTE
oracle on every path case.

A test-hygiene note worth keeping: the fixture cleaned up by name prefix, and
`update_column_attribute` **renames** rows — so a renamed row stopped matching
the prefix, survived teardown, and collided with the next run on the merge key.
Cleanup is by id now. A fixture that tests a rename cannot identify its rows by
name.

**Suite:** 526 passed, 3 skipped (was 495).

---

## 2026-08-12 — Phase 7 — SqlAttribute

`gsf/dal/pg/sql_attributes.py` — 21 functions plus the three exception types.
`gsf/dal/sql_attributes.py` becomes a selector; the Cypher moves to
`gsf/dal/neo4j/sql_attributes.py` and, as with `attributes`, stops importing
`resolve_accessible_catalog_ids` from the *selector* — a third instance of the
same cross-backend leak.

**The selector re-exports the exception types, and that is not cosmetic.**
`service.py` catches `SqlAttributeNameConflict` imported from here. If each
backend defined its own class the `except` would stop matching under one of
them, and a name conflict would surface as a 500 instead of a 409 — a
correctness bug with no failing test anywhere, because both classes exist and
both are importable.

### Two rules that fail quietly

**A SqlAttribute is scoped by the tables its own SQL references**, not by its
parent Term's tables. The two genuinely differ; using the Term's would show a
viewer an attribute querying data they cannot see.

**The check is all-or-nothing**, and phrased in the negative for a reason: not
"does it touch an allowed table" but "does it touch a disallowed one". An
attribute joining an in-zone table to an out-of-zone one passes the positive
form and must fail. The fixture builds exactly that attribute, because it is the
one case where a wrong implementation still looks right on every other row.

`_has_sql` is repeated across the list, both counts, and the retrieval reads for
a related reason: the Cypher's `HAS_SQL` match was not optional, so an attribute
with no statement is invisible. A count that included them would render a badge
promising rows the list cannot produce.

### `except Exception: return []` hid a hard SQL error, again

`fetch_tables_from_sql_attributes` combines `SELECT DISTINCT` with an `ORDER BY
ordinal_position` that was not in the select list — which Postgres rejects
outright. The function caught it, logged a warning, and returned `[]`, so the
test failed as "0 tables instead of 1" rather than as a syntax error.

That is the second time this session (after `fetch_attr_column_contexts`) that a
deliberate catch-and-degrade turned a hard error into a plausible empty result.
The catches are correct — losing retrieval context beats losing the answer — but
they mean **a test asserting emptiness proves nothing**. Every test here asserts
content.

### Notes on individual functions

* **`find_attr_by_expression` still compares in Python.** The normalisation is
  `" ".join(x.split())`, which collapses runs of *any* whitespace, and no SQL
  expression reproduces it — `regexp_replace` on `\s+` is close but differs on
  the Unicode whitespace `str.split` accepts. One full scan of a single term's
  attributes is worth matching the old behaviour exactly.
* **`update_sql_attribute` types its `source` parameter.** With every argument
  `None` — an empty PATCH body — Postgres cannot infer a type for a bare `NULL`
  inside `coalesce` and rejects the statement. Pinned by a test that calls it
  with nothing set.
* **`delete_sql_attribute_node` relies on `ON DELETE CASCADE`** rather than
  deleting links itself. That is `DETACH DELETE` moved into the schema, so a
  link table added later cannot be forgotten here.
* **`detach_existing_sql_edges` leaves the Sql rows alone**, as before — they
  are shared with query history and with other attributes.
* **The embedding doc text is reproduced literally.** It is what gets embedded,
  so a changed separator silently invalidates every stored vector for these
  attributes and nothing downstream fails — retrieval just quietly degrades.
  Pinned character for character, including the blank description being omitted
  rather than rendered as an empty clause.

### One snapshot regeneration

`dal_surface.json` gained `SqlAttributeSqlError` for `sql_attributes`. Not new
API — it has always been importable from this module — but the freeze
identifies a module's surface by `__module__`, and `SqlAttributeSqlError` is an
alias for `SqlParseError` defined elsewhere, so it was invisible until the
selector's `__all__` made it explicit. The snapshot now records what callers
could always import.

**Suite:** 559 passed, 3 skipped (was 526).

---

## 2026-08-12 — Phase 7 complete — Terms

`gsf/dal/pg/terms.py` — 28 functions, the largest module in the DAL. With it,
**Phase 7 is done**: `datasources` (26), `attributes` (8), `sql_attributes` (21)
and `terms` (28) all run on Postgres.

`gsf/dal/terms.py` becomes a selector and the Cypher moves to
`gsf/dal/neo4j/terms.py` — which was importing `fetch_column_attribute_columns_map`
and `resolve_accessible_catalog_ids` from the **selectors**. That is the fourth
and fifth instance of the cross-backend leak; every ported module has had at
least one. Worth checking for first, not last, in Phases 8-10.

### Three rules, each held in one place

The Cypher had already been through a round of consolidation here, and the
comments explaining *why* were the most valuable thing in the file. They are
preserved as the reason each helper exists:

* **What counts as a Term** — semantic, and represented by at least one table.
  `fetch_all_terms`, `count_terms` and `term_is_in_scope` read it from
  `_semantic_terms`, so a total can never describe a different set than the list
  it pages.
* **All-or-nothing visibility** (`_in_scope`) — a term representing *any*
  out-of-zone table is hidden entirely. Phrased in the negative, like the
  SqlAttribute check, and for the same reason.
* **What "related" means** (`fetch_term_table_pairs`) — three paths from table to
  term, counted once. Spelling them out separately is what once let a card's
  badge, the "Relationships" column and the length of the list behind them
  report three different numbers.

### The certification rollup

`_certification` replaces a Cypher list-comprehension that built a `flags` array
and measured it. In SQL it is four correlated counts — total and certified, for
column and sql attributes — plus the Term's own two booleans, compared as
`certified == total`.

The zone-scoped variant applies **each attribute kind's own visibility rule**:
plain `table_id` membership for ColumnAttribute, all-or-nothing over the SQL's
tables for SqlAttribute. That asymmetry is real and not an oversight — a
ColumnAttribute is owned by exactly one table, so it has no "straddles the
boundary" case to protect against. A test asserts a badge reads `certified` for
a scoped viewer and `partial` for an admin, off the same term.

### The rename that silently empties a list

`update_term` rewrites `term_name` on every ColumnAttribute of the Term. It
looks like housekeeping on a denormalised copy; it is not optional.
`fetch_column_attributes_by_term_id` matches attributes to their term *through
that column*, so a stale copy leaves the rows in place and the list empty, with
nothing raising. Pinned by a test that renames and then re-reads the list.

### Smaller notes

* **`merge_term` overwrites the description** where `merge_column_attribute`
  coalesces it. So a semantic rebuild discards a hand-edited term description.
  That is what the Cypher did; preserved, tested, and flagged here rather than
  quietly improved — changing it alters what a rebuild does.
* **`fetch_terms_with_sqls` excludes attribute- and analysis-owned statements**,
  so the suggester cannot learn from its own output.
* **`props` is now the `sql_query` row.** In Cypher it was `properties(sql)`,
  including the `count_monthly_YYYY_MM` counters — measured as unread in Phase 6
  and absent from the schema, so there is nothing to carry.
* **`find_column_attribute_by_column_id` is exported by both `terms` and
  `attributes`**, as it was in Cypher. `pg/terms` delegates rather than
  reimplementing; the duplication is a caller contract, not a mistake to fix.

### `fetch_table_zones_map` moved, deliberately

Neo4j keeps it in `exploration.py` (Phase 9). Postgres needs it now, so it lands
once in `pg/zones.py` and Phase 9 re-exports it — [DECISION-011], with a note
added to PLAN.md at the Phase 9 entry.

`test_postgres_and_neo4j_surfaces_match` would read that as pg gaining two
functions. It is the check that makes the `GSF_STORE` flip safe, so rather than
weaken it, a `RELOCATED` allowlist covers exactly those two names with their
reasons — a relocation is not an addition, since the function exists on both
backends and only its module differs.

### Test-suite fix

`gsf/semantic/tests/test_neo4j_dal_merge.py` patched `gsf.dal.terms.get_neo4j_conn`,
which stopped existing when that module became a selector. Retargeted at
`gsf.dal.neo4j.terms`, which is what it was always testing.

### Tests — `gsf/dal/pg/tests/test_terms.py`, 51 of them

Each of the three rules gets a test that fails for a *different reason* than "the
number is wrong": the list/total/scope-check triple asserted together, a term
straddling the zone boundary, and the related-count matched against the related
list. The certification section is deliberately dense, because every wrong
version of that rollup still returns one of three valid strings.

**Suite:** 610 passed, 3 skipped (was 559).

**Next:** Phase 8 — analyses, candidates, connections, reset.

---

## 2026-08-12 — Phase 8 complete — analyses, candidates, connections, reset

Five modules, 31 functions and 6 exception types, all now on Postgres:
`connections` (3), `candidates` (1), `reset` (3 + `ResetResult`),
`custom_analyses` (10 + 3), `pql_analyses` (8 + 2). Each moves to
`gsf/dal/neo4j/` behind a selector.

`custom_analyses` had the same cross-backend leak as every other module —
`resolve_accessible_catalog_ids` imported from the selector. Six now.

### `reset.py` is where the schema pays for itself

`_delete_nodes_in_batches`, `apoc.periodic.iterate` and `apoc.path.subgraphNodes`
are all gone. `delete_data_layer` is one `DELETE` on `catalog_database`; schemas,
tables, columns, foreign keys, joins, statement links and zone targets follow by
`ON DELETE CASCADE`. That is exactly what modelling `CONTAINS` as a parent FK
was for, and it means a child table added later cannot be forgotten here — the
batching existed only because the traversal could return more nodes than one
transaction could hold.

**B1 — the deletes are narrower, and that is a fix.** `subgraphNodes` followed
*any* relationship, so from a `Database` it reached that database's Terms and,
through a shared Term, a **different database's tables**. Resetting one database
could delete another's data. Foreign keys only point downward within one
database, so it cannot happen now. The scoped semantic delete resolves each
entity by its own path — a Term through the tables representing it, a
SqlAttribute or CustomAnalysis through the tables its SQL touches — and deletes
it only when *every* table it touches is inside this database.

**B2 — the scoped reset still does not reach `PqlAnalysis`.** It is never
attached to a database, so the traversal never found it. Preserved: a scoped
reset silently wiping every predictive analysis in the deployment is a worse
surprise than the current gap. The unscoped reset does remove them, matching the
Cypher's label-only match.

`delete_all_data` still deletes semantic before data, and the docstring now says
why in terms that survive the port: the semantic rows are identified *through*
the catalog, so deleting the catalog first would leave the semantic pass with
nothing to scope and every Term standing.

### I deleted the shared fixture, and it took a test to notice

`test_an_unscoped_reset_clears_both_collections` called `delete_all_data(None)`
against `gsf_alembic` — the database holding the ingested Pagila, Chinook and
testdb catalogs every other suite reads. It wiped all three: 3 databases, 39
tables, 218 columns, gone.

Restored by re-running `ingest_catalog` against all three connectors, and
verified against the pre-existing counts. The cause is fixed rather than worked
around: the unscoped variants now run inside `_rolled_back()`, which drives
`write_transaction` and raises a sentinel at the end of the block so the
transaction unwinds. The assertions still see the deletion — same transaction —
and one test additionally asserts the row is *back* afterwards, so the
containment itself cannot silently stop working.

The general lesson is worth stating: **a test for an unscoped destructive
function has no fixture boundary by construction.** "Scoped to a test prefix"
protects nothing when the function under test means "everything".

### `candidates.expand_info` — one `apoc.case`, five functions

The Cypher was a single `apoc.case` with three sub-queries inlined as strings.
Here each label gets a function and a dict dispatches on it; the output shape is
unchanged. Two asymmetries in the original are preserved and tested, because
both look like oversights until you see what they do:

* a **SqlAttribute with no statement does not appear at all** (the match was not
  optional) — it has nothing to contribute to a prompt, and the caller reads a
  missing id as "no context";
* a **CustomAnalysis with no statement does** appear, with `sql: ""` — that
  branch used `OPTIONAL MATCH`.

`sample_values` is emitted only when non-empty, which the Cypher was explicit
about. An empty list rendered into a prompt reads as "this column has no
values" — a different claim from "we never profiled it". Likewise
`toString(coalesce(c.data_type, ""))`: callers concatenate it, so a null must
become `""` rather than the word "None".

### Smaller notes

* **`delete_custom_analysis_node` deletes the statement too**, reproducing
  `DETACH DELETE ca, sql` rather than leaving it to the cascade — which would
  only drop the link row. An analysis's statement is not shared: it is parsed
  from text typed into that analysis, so leaving it behind accumulates
  unreachable rows that still surface in query-history reads.
* **`find_analysis_by_sql` matches exact text**, where `find_attr_by_expression`
  normalises whitespace and case. Inherited, not chosen — loosening it would
  start rejecting saves that succeed today.
* **`insert_connection` creates the database row if absent.** A connection is
  normally configured *before* anything is ingested, so there is usually nothing
  to attach to yet.
* **`list_connections` decodes a stored string as well as jsonb**, so a row
  written before the column was typed stays readable.
* **The `neo4j:` prefix in embedding row keys stays.** It is a stored key, not a
  reference to the store; changing it would orphan every embedding already
  written. A deliberate re-embed can migrate it later.

### Tests — 56 across three files

`test_reset.py` (14) asserts what **survives** in every case, not only what
goes: a delete that removes too much passes any "is it gone?" assertion. B1 gets
a shared-Term test with a single-database control beside it, so "survives"
cannot just mean the delete never matched. `test_analyses.py` (27) and
`test_candidates.py` (15) cover the zone boundary, the two OPTIONAL-MATCH
asymmetries, and both embedding text formats character for character.

**Suite:** 666 passed, 3 skipped (was 610).

**Next:** Phase 9 — `exploration`. Re-export `fetch_table_zones_map` and
`zone_covers_table` from `pg/zones.py` ([DECISION-011]); do not write a second
copy.

---

## 2026-08-12 — Phase 8 follow-up — the `expand_info` asymmetry is now bug 5

The two branches of `candidates.expand_info` that disagree about a candidate
with no statement behind it are recorded in PLAN.md's surfaced-bugs table rather
than only in this log and a docstring — they belong with the other four, and a
finding that lives only in a commit message is a finding nobody reads.

Stated plainly: **a `SqlAttribute` with no SQL is dropped from the enrichment
entirely, while a `CustomAnalysis` with no SQL comes back with `sql: ""`.** The
vector store returns the candidate either way; in the first case the enrichment
silently discards it and the generator never learns it existed.

Which arm is correct is a product question and this refactor does not settle it.
Both are defensible — an attribute with nothing to say contributes nothing to a
prompt, and equally a caller that asked about a specific id is better served by
a visibly blank entry than an absent one. What was not defensible is that the
two rules sat in adjacent branches of one query with nothing recording that the
difference was intended.

PLAN.md's Phase 8 entry now also carries a four-row table of the differences to
preserve while unpicking the `apoc.case` — including the two that are easy to
"tidy" into a bug: `sample_values` must be `None` rather than `[]` when empty
(`[]` in a prompt asserts the column *has* no values), and `data_type` must be
`""` rather than `None` (callers concatenate it, and `None` renders as the word).

No code change; `pg/candidates.py`'s module docstring gains a pointer.

---

## 2026-08-12 — Phase 9 complete — `exploration`

`gsf/dal/pg/exploration.py` — 6 public functions plus
`MAX_EXPLORATION_GRAPH_NODES`. As the plan predicted, most of it composes
helpers that already existed: `fetch_semantic_exploration_graph` is almost
entirely calls into `pg/terms` and `pg/sql_attributes`, and porting it late made
it nearly free.

Only **`gsf/dal/model_interchange.py`** is left on Cypher.

### The degree invariant, and what protects it

A table's `relationship_count` in the graph payload must equal the `total` its
related-nodes page reports. The two are computed by different code — one counts
edges, the other counts neighbours — so they agree only if "related" means
exactly the same thing in both. When they drift the UI draws a node labelled "5
relationships" whose panel lists 4, and nothing fails anywhere.

`_related_tables()` is now the single definition, read by both sides:
a shared statement with text, or a foreign key in **either** direction. The two
FK legs stay separate because a foreign key is stored one way round and
relatedness is not.

Two details that keep it true and are easy to lose:

* **Counts are computed before truncation.** `limit` can leave a neighbour out
  of the payload; counting after would shrink a node's degree to match the
  subset drawn and contradict the page behind it.
* **The catalog path is an outer join** in the related-nodes page. Requiring it
  would drop a related table with no schema or database above it from the page
  while the graph still counted it — the invariant broken by a join, not by a
  count.

The test asserts the invariant for every table in a fixture built to break it:
edges by shared SQL *and* by foreign key, keys in both directions, a pair
connected by both mechanisms at once, and a self-referential key that must count
as nothing. A second test pins the expected degrees outright, so the invariant
cannot pass by comparing zero with zero. Both are re-asserted under zone
scoping — an invariant that only holds for an admin is not worth much.

### `fetch_table_zones_map` is re-exported, and says so

[DECISION-011] held: `pg/exploration` imports it from `pg/zones` rather than
writing a second copy, and a test asserts the two names are literally the same
function object.

That surfaced a mirror image of the Phase 7 surface problem. The freeze resolves
a module's contract by `__module__`, which for a re-export points at the
*definition* — so `pg/exploration` appeared to be missing a function
`neo4j/exploration` has. Fixed by declaring `__all__` on `pg/exploration`, which
is the honest answer: it genuinely is part of that module's public surface.
Better than a second `RELOCATED` entry, which would have recorded an exemption
where there is no divergence.

### A finding: two filters disagree about "blank"

`_non_empty_sql` trims before deciding, so a whitespace-only statement makes no
edge. `fetch_table_exploration_details` drops only *falsy* text, so the same
statement is still listed among the table's queries. The user-visible effect is
a query on the detail panel with no line on the graph beside it.

Preserved. It loses nothing, and a whitespace-only statement should not exist in
the first place — but the two filters differing is now pinned by a test, and
recorded in PLAN.md's Phase 9 entry, so unifying them is a deliberate act rather
than a tidy-up that quietly changes what the panel shows.

### Smaller notes

* **`_count_of` and `_terms_count` are imported across modules, privately.**
  They are the one definition of a table's column/sql/term counts, and the
  Cypher shared them the same way — `TABLE_COUNTS_SUBQUERY`, imported from
  `neo4j/datasources`. A second copy is how the graph's badge and the schema
  tree's badge start disagreeing about the same table.
* **`MAX_EXPLORATION_GRAPH_NODES` is re-exported by the selector.** The router
  reads it to document its own cap; a per-backend value would let the two
  disagree about how large a payload can get.
* **`neo4j/exploration.py` was importing from three selectors** — `sql_attributes`,
  `terms` and `users`. Seven instances of that leak now, one in every module
  ported so far.

**Suite:** 695 passed, 3 skipped (was 666).

**Next:** Phase 10 — `model_interchange`, the last module. It must fix bug 3
(`bool('NO')` is `True`, so every column exports as nullable) and un-skip the
two `test_export_model_*` tests **together**.

---

## 2026-08-12 — Phase 10 complete — `model_interchange`

**Every DAL module now runs on Postgres.** `gsf/dal/pg/` is 15 modules; the only
Cypher left is behind the selectors, waiting for Phase 11 to delete it.

### Bug 3 is fixed, and it was worse than the description

`is_nullable` is stored as the strings `'YES'`/`'NO'` — what
`information_schema` reports — and the reader was `bool(...)`. `bool('NO')` is
`True`, so **every column in every export claimed to be nullable**. Measured on
the fixture: **112 of 218 columns wrong**, and an export is what another
deployment imports as truth.

Nothing ever failed, because a bool is exactly what the schema expects and
`True` is a plausible one. The only way to see it was to look at what was stored
rather than at what was read.

`_is_nullable` parses it properly, and `_nullable_to_stored` writes it back in
the store's own vocabulary on import — a bool there would make a re-ingest diff
see every imported column as changed, so the column would flap on every
subsequent ingest. One test covers both stored values, one covers the parser
directly, and one runs export → import → re-export, which is the only one that
would catch getting the reader right and the writer wrong.

**The two `test_export_model_*` tests are un-skipped**, in the same change, as
the plan required. The thing that had kept them skipped was not the store at
all: `export_model` reaches it twice, and only one of the two paths was patched.
`_dialect_by_database_name()` asks the connector registry and `list_connections()`
what dialect each database speaks. Both are patched now; neither test is about
dialect resolution.

### Two simplifications banked

**`_ensure_import_indexes` is gone.** It created uniqueness constraints and
`imported_id` indexes on every import — a schemaless store had no other way to
guarantee them, and an unindexed `imported_id` turned a several-thousand-column
import into a quadratic crawl. The schema declares them once.

**The split transaction collapses into one.** SQL attributes and custom analyses
used to be applied *outside* the main transaction because `add_query` opened its
own auto-commit Neo4j session and would deadlock against locks the outer
transaction held. The Postgres `add_query` runs on the same connection, so the
whole import is atomic — a failure part-way now leaves nothing behind, where
before it could leave a catalog with no semantics on top. Pinned by a test that
fails an import mid-way and asserts the database row is absent.

Containment being a parent FK also removes a step per level: each level is
created *with* its parent, where the Cypher needed a second batched `MERGE` for
the `CONTAINS` edges.

### Two findings from the round trip

**Bug 6 — an export of one database cannot be imported when a Term spans two.**
`_export_terms` returns *every* table representing a term, including tables
outside the exported set; that is deliberate and keeps a partial export honest
about a term it only partly owns. But the importer resolves each `represents`
entry against the document, so such an export names a table it does not carry
and raises.

Preserved. Silently dropping the unresolvable entry would import the term as
though it belonged wholly to the importing database — quieter, and worse, than
an error naming the table it cannot find. Fixing it properly means deciding what
a partial export *should* say about a shared term, which is a product question.
Recorded in PLAN.md and pinned by a test.

**B9 — the import now meets constraints the graph never had.** `term` is
`UNIQUE(name, source)`, so importing a document whose Term shares a name with an
existing one raises where the graph created a *second* Term node — and
`merge_term` matched on `{name, source}`, so which of the two a later write
found was arbitrary. The duplicate was the bug; the error is the fix. Left to
surface ([DECISION-012]); it does not affect the normal workflow, where an
import targets a deployment that does not already have those names.

### Tests — `gsf/dal/pg/tests/test_model_interchange.py`, 28 of them

The plan's Done criterion — **the round trip is idempotent** — needed a helper
worth naming. A document exported here carries this store's live ids, and the
import matches those against `catalog_table.id`, so re-importing it *adopts* the
rows rather than copying them. That is correct, and has its own test. To
exercise the *create* path the ids have to be foreign, which is what a document
from another deployment has — `_reid` rewrites them, and the names with them,
because otherwise the run collides with the constraints above rather than
testing the import.

**Suite:** 725 passed, 1 skipped (was 695 passed, 3 skipped). The remaining skip
is live Databricks credentials; the two model-interchange skips are gone.

**Next:** Phase 11 — flip `GSF_STORE` to Postgres, delete the Neo4j
implementations and the selectors. Note its prerequisite, recorded in Phase 7:
`capture_dal_golden._fixture_ids()` still resolves every fixture entity with
hardcoded Cypher, so the golden replay cannot grade Postgres until it is ported.

---

## 2026-08-12 — Phase 11, part 1 — the golden replay finally grades Postgres

**128 of 128 goldens, recorded on Neo4j, now pass against Postgres.** The
prerequisite recorded in Phase 7 is met, and the claim this plan has been making
since Phase 5 — that the replay is the fidelity oracle — is true for the first
time.

Before: 49 of 128 failed. Not because the DAL was wrong, but because the harness
could only ever ask Neo4j.

### What had to change

**`capture_dal_golden._fixture_ids()` and `build_id_map()` were hardcoded
Cypher.** Both now resolve through the DAL, so the two backends are asked the
same questions about the same fixture. `build_id_map` matters as much as the
ids: a normalisation token has to describe the *same* entity on both sides or
the comparison compares nothing.

**`seed_graph_fixture` was Cypher in four places** — three id lookups and the
reset. All four go through the DAL now, so one script builds the same fixture on
either backend. The summary it prints is counted through the DAL too, rather
than reintroducing a store-specific query.

**`test_golden`'s skip guard keyed on `NEO4J_URI`.** Under
`GSF_STORE=postgres` it skipped all 128 comparisons — reporting "all green"
while grading nothing. That is the same failure mode as the harness itself, one
level down.

### Two real gaps the seeding surfaced

**`SqlAttribute -[HAS_SQL]-> Sql` was not in the Postgres link registry.** Phase
4 built that registry from the ingest path, where a SqlAttribute never appears —
so creating one on Postgres raised `UnknownLink`. Registered, along with the
`sql_attribute` entity spec.

**`Column -[JOIN]-> Column` had nowhere to go.** Query ingestion emits one edge
per observed join condition, carrying `join_refs`; the schema had only
`table_join` (Table→Table, `join_columns`), which is written by model import and
is a different thing entirely. Nothing writes both. Added `column_join` and
`column_union` in migration `63f6c85ccfce`.

This never fired before because the *catalog* ingest emits no queries — "Total
Added Queries: 0" — so Phase 4's verification could not have caught it. The
semantic fixture is the first thing that parses SQL.

`refs` accumulates on conflict rather than overwriting, which is what the Cypher
did (`coalesce(rel.join_refs, []) + new`) and the point of the column: each
statement adds its own reference. It also **de-duplicates**, which the graph did
not — re-ingesting the same statement grew the array without bound there, and
that is a leak rather than a behaviour worth reproducing.

### One improvement, found by the oracle doing its job

`fetch_attr_column_contexts` returned a different column than Neo4j did. Both
were arbitrary — a row per linked column, last one wins — but the *right* answer
is not arbitrary: **the owning column wins**, the one linked by `HAS_ATTRIBUTE`
rather than one merely pointing at the attribute through `SEMANTIC_FK`. An
attribute describes its owning column, so returning a referencing column's name
and table as the attribute's context was simply wrong whenever the ordering
landed there. The golden happens to agree, which is how it was found.

### One divergence accepted — B10

`fetch_item_by_id` and `fetch_node_properties_by_id` return every column of a
row where the graph returned only the properties something had set. Extra keys
appear; `created` — stamped on every node, read by nothing, verified across
backend and frontend — is gone. [DECISION-013].

Handled by naming the **individual keys** whose presence may differ, per read,
rather than exempting the two reads. Every other key is still compared exactly.
A blanket exemption would have quietly stopped grading two of the 128 — the
exact failure this phase exists to fix.

### Test-hygiene fixes

* **The Phase 5 zone tests leaked ~8 rows per run** into the shared store, named
  outside their fixture's prefix so cleanup never matched them. 32 had
  accumulated. `zones.list_zones` reads what is actually there, so the golden
  caught it — a test asserting real state is worth more than one asserting a
  mock, and this is why.
* **`reset_graph` had to clear zones explicitly.** `delete_all_data(None)` does
  not touch them on either backend — the Cypher's semantic labels never included
  Zone — and the old `MATCH (n) DETACH DELETE n` swept them up incidentally. A
  fixture built on leftover zones is not reproducible.
* **The frozen surface snapshot records one backend's annotations**, so
  `resolve_table_filter` — whose types legitimately differ ([DECISION-008]) —
  tripped it under the other. The `BACKEND_SPECIFIC` allowlist now applies to
  the snapshot comparison as well as the pg/neo4j one, on the same terms:
  parameter *names* must still match, since those are what callers pass.
* **Two `model_interchange` server tests patched Neo4j internals but called
  through the selector**, so under Postgres they patched one backend and ran the
  other. They also imported `UnknownDatabaseIdsError` from the selector while
  calling the Neo4j function — the exception-identity hazard the selector
  docstrings warn about, showing up for real.

**Suite: 725 passed, 1 skipped — on both backends, repeatably.**

**Next:** the flip itself — default `GSF_STORE=postgres`, then delete
`gsf/dal/neo4j/`, the selectors, and the Neo4j infrastructure.

---

## 2026-08-12 — Phase 11 complete — **Neo4j is gone**

The last phase. `origin/main` merged first (one commit), then the flip.

**Suite: 692 passed, 1 skipped.** The count dropped from 725 because tests of
deleted code went with it, not because coverage did — see below.

### What was deleted

`gsf/dal/neo4j/`, `gsf/catalog/store/neo4j/`, `neo4j_tx.py`,
`cypher_fragments.py`, `gsf/infra/store.py` (the `GSF_STORE` switch itself), and
every selector. `gsf/dal/pg/*` and `gsf/catalog/store/pg/*` are promoted a level
— `gsf.dal.terms` is now the implementation rather than a two-branch import.

Also gone: `neo4j>=6.1.0` from `pyproject.toml`, the `neo4j` service and its
four volumes from `docker-compose.yml`, `helm/gsf/templates/neo4j.yaml` and the
`NEO4J_*` values, `.env.example`, `dev_tools/setup_env.sh`, and the README /
DEPLOYMENT sections.

### The health endpoint was reporting a lie

`_check_neo4j()` called `verify_connectivity()` — which, once `connections.py`
was ported, probed **Postgres**. So the endpoint reported `neo4j: ok` while
checking a database Neo4j had nothing to do with, and would have kept doing so
indefinitely. Collapsed to a single `postgres` key, checked through the DAL's
pooled connection rather than a fresh `psycopg.connect`: a raw connection proves
the *server* is reachable, which is not the same as proving this process can get
a usable connection out of its pool, and the pool is what every request needs.

`HealthResponse.neo4j` is removed. No frontend consumer, verified before
deleting rather than assumed.

### A prompt change, called out as one

`semantic_fk.py` labelled its retrieval candidates `neo4j_id`, in **LLM prompt
text**. Renaming it to `column_id` is a prompt change, not a cosmetic one — the
prompt is the interface to the model. Done anyway: leaving a deleted store's
name in a prompt is worse than the small risk of a reworded instruction, and the
label only has to be consistent between the candidate list and the instruction
that reads it. Both were changed together, along with the
`FkHitSelection.selected_id` description.

`check_properties_compatibility_with_neo4j` is now `check_properties_are_flat`.
Its own docstring had said Phase 11 should rename it. The constraint outlived
its reason: these values become table columns, and a dict arriving where a
scalar belongs fails at the insert with a far less useful message.

### Tests: what went, and what did not

Deleted with their subject: `test_neo4j_tx.py`, `test_neo4j_dal_merge.py`,
`test_bridge_tables.py`, the pg/neo4j surface comparison, and six mock-based
`model_interchange` tests.

**Their coverage did not go with them.** Every one of those six had a Postgres
equivalent in `gsf/dal/tests/test_model_interchange.py` asserting the same
behaviour against a live store — which is a better test of "the import is one
transaction" than a mocked `write_transaction` ever was. That is stated in the
remaining file's docstring so the next reader does not mistake the deletion for
a gap.

`test_close_store.py` shrank from four tests to two, because `close_store` now
closes one thing instead of three.

### Two findings from the deletion itself

**The surface snapshot could never have matched itself.** `sql_fragments`
defaults an argument to a `Table`, whose `repr` embeds memory addresses — so the
frozen signature differed every process. It had never been exercised because the
module was new in Phase 7 and the snapshot was regenerated in the same run.
`_describe` now normalises addresses away.

**`SqlAttributeSqlError` vanished from the frozen surface** without being
removed from the code — `router.py` still catches it to return a 422. It is an
alias for `SqlParseError`, so its `__module__` points elsewhere and the freeze
could not see it once the selector's `__all__` was gone. `sql_attributes.py`
now declares its own `__all__`. Worth noting as a shape: a freeze that resolves
by `__module__` is blind to aliases, and an alias is exactly the kind of thing
that gets deleted by accident.

### Fork parity, narrowed rather than deleted

`gsf/catalog/store/` is no longer a fork of anything, so every mapping naming it
is gone from `FORKED` and it sits in `DIVERGED` instead. What remains is the
genuinely storage-agnostic part of the Phase 1 fork — normalisation, SQL
parsing, the model classes, the parsers — which still tracks upstream and where
drift is still worth catching. The upstream→GSF import equivalences had to stay,
though: without them a genuinely verbatim fork reads as drift on its import
lines alone.

**The refactor is complete.** PLAN.md carries a banner saying so; it is kept as
the record of why the schema looks the way it does.
