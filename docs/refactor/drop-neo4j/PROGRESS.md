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

**Also fixed the stale `CLAUDE.md` dev command** (the second item that had been
under *Blocked*). It documented `uv run uvicorn gsf.server.main:app`, but
`gsf/server/main.py` does not exist — the factory is `create_app()` in
`gsf/server/__main__.py`. Corrected to the `--factory` form, and added the two
commands that were missing entirely: `python -m gsf.server` (what the
Dockerfile runs) and `python -m gsf.ingestion_service`. All three verified to
resolve, and `gsf/server/main.py` confirmed absent.
