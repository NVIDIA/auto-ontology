<!--
SPDX-FileCopyrightText: Copyright (c) 2026, AkilsuryaS.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Issue 267: local acceptance validation

The import path now checks database, schema, table, column, term, column
attribute, SQL attribute, and analysis identities before mutation. Conflicts
return HTTP 409 with `detail.code: model_import_conflict`, the proposed ID,
matching live IDs, object path, and export-and-bind instructions. Same-path
objects are diagnostic matches and are never implicitly adopted.

Preflight and replacement use the same deletion plan, so replacing an in-scope
semantic object remains supported while shared terms are protected. Both live
and imported IDs keep existing objects during replacement. Resolution also
rejects ambiguities that appear after preflight. Database integrity failures
are translated at the service boundary after transactional rollback; SQL,
parameters, and driver messages are excluded from the response.

## Acceptance evidence

Validated locally against PostgreSQL 17 with the repository's Alembic migrations.
The fix was refreshed against `main` at `a12fec75` on November 24, 2026.
The full refreshed backend suite passed: **2,968 passed, 2 skipped** (live
Databricks/MySQL tests). The **5 gateway tests** also passed. Ruff, frontend
Prettier/ESLint, production frontend build, API schema generation, and Alembic
schema-drift validation passed; no migration is needed.

The focused identity tests also cover consistent repeated declarations of shared
attributes and exact-ID catalog bindings with edited display labels.

## Relationship to PR 581

This change owns identity binding and fatal import errors (#267, under #250).
PR #581 owns applying replacement edits to already matched terms/attributes,
certification conflicts, update counts, and re-embedding. Preflight leaves
matched semantic name edits to that policy, permits a shared attribute to be
declared once per column, and checks the stored uniqueness keys of matched
catalog rows because catalog-label updates belong to #250.

In a scratch checkout of #581 at `bc7a680b`, overlaying this preflight and
resolver passed **47 existing import tests**. One assertion expecting a raw
uniqueness-constraint exception was excluded because #267 deliberately replaces
that error with a typed conflict. This checks guard compatibility; it does not
claim the two complete branches merge without conflict. Both change the import
module and replacement planning, so their merge order should be coordinated.


| Requirement                              | Validation                                                                                                                                     |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Different IDs on matching physical paths | All seven fixture objects reported together, for both replacement modes                                                                        |
| Complete conflict reporting              | Catalog and all four semantic entity kinds reported in one response                                                                            |
| No partial mutation                      | Real PostgreSQL readback and a write-boundary assertion before any writes                                                                      |
| Exact live IDs and imported IDs          | Successful imports and idempotent reimports, including SQL attributes and analyses                                                             |
| Genuinely new names                      | Full copy into a new database, preserving relationships                                                                                        |
| Ambiguous matches                        | Every imported entity kind, plus late ambiguity after preflight                                                                                |
| Cross-database collisions                | Database/schema/table/column identity boundaries and protected shared terms                                                                    |
| Recovery                                 | HTTP export, bind to exported target IDs, successful HTTP reimport                                                                             |
| Rollback                                 | Real PostgreSQL uniqueness violation after earlier import writes; exact pre/post readback                                                      |
| Gateway behavior                         | Actual import route, permission guard, proxy, FastAPI, and PostgreSQL; raw YAML and multipart 409, bound import 200, viewer 403, anonymous 401 |

Gateway tests stub session resolution to supply admin/viewer/anonymous callers.
They exercise real role permission checks and real backend HTTP/database calls;
they do not test the external login provider or a deployed environment.

## Reproduce

Point `POSTGRES_*` at a disposable PostgreSQL database, then run from the repo
root:

```sh
uv sync --locked --extra kumo
uv run --no-sync alembic upgrade head
uv run --no-sync pytest auto_ontology/dal/tests/test_model_interchange.py auto_ontology/server/model_interchange/tests auto_ontology/dal/tests/test_dal_surface.py -q
```

Install the locked frontend dependencies with `pnpm install --frozen-lockfile`
in `frontend`, then run the gateway acceptance harness from the repo root:

```sh
uv run --no-sync python -m dev_tools.verify_model_import_gateway
```

The harness creates and cleans up its own temporary catalog and local backend
listener. It requires no model or embedding credentials.

For the full backend suite, also seed the deterministic source and graph
fixtures as described in [the fixture guide](../../dev_tools/fixtures/sql/README.md)
and `.github/workflows/ci-python.yml`.
