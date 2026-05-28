# OpenMetadata 1.12.9 — feature evaluation report

**Date:** 2026-05-28  •  **Server:** `docker.getcollate.io/openmetadata/server:1.12.9`  •  **Ingestion:** `docker.getcollate.io/openmetadata/ingestion:1.12.9`  •  **Source:** Snowflake trial account (creds in untracked `.env`; account will be deleted ~3 weeks).

Every check below was driven by `curl` or by the `openmetadata-ingestion` CLI run inside a one-shot container from the official ingestion image. All raw outputs live in this directory.

## Result table

| # | Feature | Result | Evidence file(s) | Notes |
| --- | --- | --- | --- | --- |
| 1 | docker-compose stack up; UI on :8585; API on :8585/api | **PASS** | `00-stack-up.txt`, `00b-auth-check.json` | OM 1.12.9 reports healthy on `/api/v1/system/version`. UI HEAD returns 200 (4.7 KB). Bot JWT verified by hitting `/v1/users/loggedInUser`. |
| — | Stack stability footnote | — | `00-stack-up.txt` + below | The bundled `ingestion` (Airflow) container OOM-killed on a 7.7 GB Docker VM. We stopped it and instead run `metadata ingest` from one-shot containers on the same `app_net` network. Affects ops ergonomics but not feature coverage. |
| 2 | Snowflake **metadata** ingestion (YAML + CLI) | **PASS** | `01-metadata-ingestion.log` | 32 tables ingested across `SNOWFLAKE_SAMPLE_DATA.TPCH_SF{1,10,100,1000}` in 33 s, 100% success. |
| 2 | Snowflake **usage** ingestion (7-day window) | **PASS** | `02-usage-ingestion.log`, `03g-usage-reingest.log` | First pass yielded `Workflow OpenMetadata Summary: Processed records: 0` because the trial account's recent queries didn't touch any OM-tracked tables; after we ran 8 realistic SELECTs against the TPCH tables, the second usage pass landed 9 records → 11 queries visible via API. |
| 3a | Tags API (list, create classification, create tag, PATCH onto column) | **PASS** | `03a-classifications.json`, `03a-classification-create.json`, `03a-tag-create.json`, `03a-column-patch.json`, `03a-column-verify.json` | 17 built-in classifications listed. Created `Sensitivity` → tag `Sensitivity.Internal` (HTTP 201). PATCHed onto `TPCH_SF1.CUSTOMER.C_NAME` with `labelType:Manual,state:Confirmed`. Re-GET confirms persistence. |
| 3b | Auto Classification with PII detection | **PASS** | `03b-pii-autoclass.log`, `03b-people-after-autoclass.json`, `03b-seed-pii.log` | Initial run against TPCH `CUSTOMER` detected nothing — values are synthetic (`Customer#000060001`, `9Ii4zQn9cX`). We seeded a realistic-PII table `WIDEWORLDIMPORTERS.OM_EVAL.PEOPLE` (8 rows) and re-ran. All 6 columns tagged correctly: `FULL_NAME→PII.Sensitive` (SpacyRecognizer 0.85), `EMAIL_ADDRESS→PII.Sensitive` (EmailRecognizer 1.00), `PHONE_NUMBER→PII.NonSensitive`, `SSN→PII.Sensitive` (UsSsnRecognizer 0.85), `DATE_OF_BIRTH→PII.NonSensitive`, `HOME_ADDRESS→PII.NonSensitive`. Tag `state` is `Suggested` by design — the workflow proposes, a human confirms in UI. |
| 3c | Chrome extension | **MANUAL** | `03c-chrome-extension.md` | Extension `pakbbdhbbiclnceabdmnghamabjloofc` published by Collate Inc. on the Chrome Web Store. The extension talks to the same `/api/v1` we already validated programmatically, so the API side is covered; the install step is the only manual piece. |
| 3d | PowerBI / Tableau connectors + lineage | **PARTIAL (schema-verified, not live-tested)** | `03d-bi-connectors.md`, `03d-bi-schemas.json`, `03d-bi-schema-tables.md` | Both plugins ship in the 1.12.9 ingestion image alongside 17 other dashboard sources. Full pydantic JSON Schemas + minimal valid ingestion YAMLs + service-side prerequisites (Tableau Metadata API + role floor; PowerBI service-principal + AAD permissions + tenant settings) documented. No live BI creds → cannot move to PASS without those. |
| 3e | API: list tables/columns/descriptions + flatten to CSV | **PASS** | `03e-tables.json`, `03e-tables.csv` | `GET /api/v1/tables?fields=columns,description&database=snowflake_eval.SNOWFLAKE_SAMPLE_DATA&limit=50` → 32 tables. Flattened to 244-row CSV (`db,schema,table,column,dataType,description`). Description column is empty for these tables because Snowflake's TPCH sample has no column COMMENTs — that's a source-data fact, not an OM defect. |
| 3f | API to run a query | **N/A — feature not in scope** | `03f-run-query.md`, `03f-no-run-query-endpoint.txt` | Probed `/queries/execute`, `/queries/run`, `/sql/execute`, `/query/run` → all 404. OpenMetadata is a catalog, not a query engine. Confirmed via upstream docs (REST API ref + Python SDK lineage page — `add_lineage_by_query` *parses* SQL, doesn't run it). |
| 3g | API: query history (cross-service + per-table) | **PASS** | `03g-queries.json`, `03g-queries-for-people.json`, `03g-table-people-queries.json`, `03g-top-tables.json` | `GET /v1/queries?service=snowflake_eval&fields=query,users,queryDate,queryUsedIn&limit=20` → 11 queries. Per-table filter `GET /v1/queries?entityId=<PEOPLE.id>` → 3 PEOPLE-specific queries with full text + `queryUsedIn` link. Table-level `usageSummary.weeklyStats` populated (PEOPLE: count=4, percentileRank=89). |

## Headline read

| Verdict | Count | Items |
| --- | --- | --- |
| PASS | 7 | stack-up, metadata ingest, usage ingest, tags (3a), auto-classification (3b), tables list+CSV (3e), query history (3g) |
| PARTIAL | 1 | PowerBI/Tableau (3d) — verified at connector-presence + schema level; full lineage test requires live BI creds we didn't have |
| MANUAL | 1 | Chrome extension (3c) — install is a Web Store click |
| N/A (feature not supported) | 1 | Run-a-query (3f) — explicitly outside OM's scope |

## How to reproduce

```bash
cd openmetadata-eval

# 0. one-time
docker compose pull
docker compose up -d
docker compose stop ingestion       # frees ~2 GB; we use one-shot ingestion runs instead
bash scripts/01-wait-for-server.sh
bash scripts/02-mint-jwt.sh         # writes OM_TOKEN into .env

# 1. ingestion
bash scripts/run-ingestion.sh metadata  evidence/01-metadata-ingestion.log
bash scripts/run-ingestion.sh usage     evidence/02-usage-ingestion.log

# 2. feature checks
bash scripts/03a-tags.sh
docker run --rm \
  -e SF_USER=...  -e SF_PWD=... -e SF_ACC=... -e SF_WH=... -e SF_ROLE=... -e SF_DB=WIDEWORLDIMPORTERS \
  -v "$(pwd)/scripts:/work" docker.getcollate.io/openmetadata/ingestion:1.12.9 \
  python /work/03b-seed-pii.py
bash scripts/run-ingestion.sh pii-metadata  evidence/03b-pii-metadata-ingest.log
bash scripts/run-ingestion.sh pii-autoclass evidence/03b-pii-autoclass.log
bash scripts/03b-verify-pii.sh
bash scripts/03e-tables-csv.sh
docker run --rm \
  -e SF_USER=...  -e SF_PWD=... -e SF_ACC=... -e SF_WH=... -e SF_ROLE=... \
  -v "$(pwd)/scripts:/work" docker.getcollate.io/openmetadata/ingestion:1.12.9 \
  python /work/03g-seed-queries.py
bash scripts/run-ingestion.sh usage         evidence/03g-usage-reingest.log
bash scripts/03g-query-history.sh
```

## Setup pain points worth flagging

1. **Memory floor.** The quickstart compose has no per-service `mem_limit`. On a 7.7 GB Docker VM, Elasticsearch (1 GB heap, ~2 GB resident) plus the ingestion/Airflow container (~1.5 GB) plus MySQL plus the OM server collectively pushed the VM into OOM-killing ES. Working around it by skipping the long-lived `ingestion` Airflow container (we run ingestion via `docker run --rm ... metadata <subcmd>` instead) is fine for evaluation but means we never validated the bundled Airflow UI. For prod we'd push for the postgres-flavor compose plus 16 GB minimum.
2. **Stale `docker-compose.yml`** ships with `version:` set, which Docker Compose v2 warns about on every invocation. Cosmetic but noisy.
3. **No baseline admin password hashing in `.env`.** The quickstart admin is `admin@open-metadata.org` / `admin` (b64-encoded on login). Fine for a sandbox; for anything internet-facing you'd switch to JWT/SAML/OIDC and rotate.
4. **`autoClassificationPriority`** on the `PII.None` tag is `false`, on `Sensitive` and `NonSensitive` it's `true` — meaning the auto-classifier never explicitly *labels* a column as "Non-PII" even when nothing matched. Workable, just non-obvious if you build a UI that expects exclusivity.
5. **Auto-classification `state` is `Suggested`** by design — applied tags do not show up as "confirmed" until a human approves them in the UI, and our 3a `state:Confirmed` patch shows the two label types side-by-side. If you build dashboards that filter "PII coverage" you need to decide whether to count `Suggested` or only `Confirmed`.
6. **Usage workflow needs queries to look at.** A freshly-provisioned Snowflake trial has very few queries against OM-tracked tables, so the first usage run will look "empty". You need to either (a) point usage at a busy production warehouse or (b) seed some real queries first. The fix in this eval was a one-time `03g-seed-queries.py` SELECTing eight TPCH joins.
7. **Sample data was not returned by `?fields=sampleData`** on PEOPLE after auto-classify, even though `storeSampleData: true` was set. The PII tags landed correctly but the rows themselves never showed via `GET /v1/tables/{id}/sampleData` (`{sampleData: null}`). Looks like a 1.12.9 quirk worth filing.
8. **`databaseServices/name/<n>` returns a redacted payload** for `name`, `serviceType`, and `connection` when called with the bot JWT — even though the bot is the one that wrote the service. We worked around it via the `databases?service=…` endpoint, but it's a confusing 200 OK shape that takes a minute to debug.
9. **Tableau docs reference an older `siteUrl` field** that was removed in 1.7.x. The newer "Browser Extension" docs page lists *two different* extension IDs (`pakbb…` and `ndjn…`) — flagged in `03c-chrome-extension.md`. Both are upstream-docs hygiene issues.

## Open questions to take back

1. **PII confirm-vs-suggest UX.** Do downstream consumers want to act on `state: Suggested` tags, or only on human-confirmed ones? Choice affects how soon "PII coverage" dashboards become meaningful.
2. **Sample-data field on PEOPLE is null** — bug? config gap? Worth filing an issue or pinning down a workaround before we promise sample-data visibility to internal users.
3. **PowerBI vs Tableau lineage parity** — we documented prerequisites but haven't run them. The Tableau Metadata API path is generally healthier; PowerBI lineage is much harder to drive without admin-API consent. We should pilot one of them end-to-end before committing OM as our BI catalog.
4. **Usage backfill window**. We ran the usage workflow with `queryLogDuration: 7`. For a stale Snowflake account this is fine; for a busy one we'd want to know the practical ceiling (does `30` finish? does it cost-throttle the warehouse?).
5. **Long-term ingestion runtime** — we never validated the bundled Airflow container in this eval because of the memory pinch. If we standardise on running ingestion via one-shot containers or a separate cluster, we should bake that into the OpenMetadata helm chart instead of using the all-in-one compose.
6. **Bot JWT lifecycle.** The ingestion-bot JWT we minted has `JWTTokenExpiresAt: null` (unlimited). Probably fine for an internal deployment behind SSO, but worth confirming the rotation story matches our security policy.
