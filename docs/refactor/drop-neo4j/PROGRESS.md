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
