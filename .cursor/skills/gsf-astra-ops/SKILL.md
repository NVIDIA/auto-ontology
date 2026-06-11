---
name: gsf-astra-ops
description: >-
  Operate and troubleshoot the GSF app on the Astra OpenShift staging cluster
  (gsf-app-deploy). Covers deploying chart/values changes through fusion + the
  GitOps deploy repo, fixing data/chat outages (empty CONNECTION_STRINGS,
  ExternalSecret refresh lag, single-conversation lock), seeding/resetting the
  demo business dataset, and verifying /data and /chat. Use when the user asks
  to check, deploy, fix, seed, re-ingest, reset, or debug the GSF app on Astra,
  its catalog or chat, or its staging deployment.
---

# GSF on Astra (staging) — Ops

Operational runbook for the deployed GSF app. The agent already knows Helm/K8s/git;
this captures the GSF-specific facts and the non-obvious deploy model.

## Key facts

| Thing | Value |
|---|---|
| App URL | `https://frontend-ns-gsf-app-deploy.apps.astrastg01-ocp-pdx04.nvidia.com` |
| Cluster | `astrastg01-ocp-pdx04` · namespace `ns-gsf-app-deploy` |
| Deploy repo (ArgoCD source) | `ssh://git@gitlab-master.nvidia.com:12051/ape-repo/astra-projects/access-gpu-product-all/gsf-app-deploy.git` |
| Deploy repo HTTPS (for fusion) | `https://gitlab-master.nvidia.com/ape-repo/astra-projects/access-gpu-product-all/gsf-app-deploy` |
| Chart path in deploy repo | `deployment/stg/` (Helm chart: `templates/`, `values.yaml`) |
| Vault secret | `fusion/astra/access-gpu-product-all/gsf-app-deploy/stg` |
| K8s secret (from ESO) | `gsf-app-deploy-secrets` (exposes `CONNECTION_STRINGS`, `NVIDIA_API_KEY`, `POSTGRES_*`, `NEO4J_*`, …) |

## Deploy model (read this first)

ArgoCD syncs `deployment/stg/` from the deploy repo. **Two ways to change it:**

- **Values only** → `fusion deploy update` (writes `deployment/stg/values.yaml`, auto-syncs ~3 min). Works even though the repo is archived (fusion-cli commits server-side via API).
- **Templates (new/changed files)** → `git push` over SSH **port 12051**. The repo is often **archived** (git push → *"You can't push code to an archived project"*); ask the user to unarchive it (GitLab → Settings → General → Advanced → Unarchive), push, then they can re-archive. `fusion deploy create --force` does NOT work to update (refuses: repo exists / wants delete).

Auth: `fusion login` (token expires often — re-login on `Token expired`). Git uses SSH key on port 12051.

Trigger a sync / restart a workload: bump a value (e.g. `ingestion.resources.requests.memory` by 1Mi) and `fusion deploy update`. Pods only restart when their manifest changes.

```bash
URL="https://gitlab-master.nvidia.com/ape-repo/astra-projects/access-gpu-product-all/gsf-app-deploy"
fusion login
fusion deploy update -n "$URL" -e stg -f ./values.yaml -m "message"   # values
# templates: edit deployment/stg/templates/*, then:
GIT_SSH_COMMAND='ssh -p 12051' git push origin main
```

## Verify (health, catalog, chat)

```bash
H=frontend-ns-gsf-app-deploy.apps.astrastg01-ocp-pdx04.nvidia.com
curl -s "https://$H/api/datasources/dbs"            # databases in the catalog
curl -s "https://$H/api/schemas/<db_id>"            # schemas for a db
curl -s "https://$H/api/tables/<schema_id>"         # tables for a schema
# chat (slow ~2-3 min; one conversation at a time):
curl -s -m 320 -N -X POST "https://$H/api/chat/completions" \
  -H 'Content-Type: application/json' -d '{"question":"how many customers we have?"}'
```

Chat-slot probe: `POST /api/chat/completions` with `{"question":""}` returns `409`
when a conversation is held, otherwise `422` (slot free).

## Common problems → fixes

- **`/data` empty, `/chat` returns `[DONE]` with no content** → `CONNECTION_STRINGS` is empty/wrong in Vault. The ingestion service exits if it's unset, and the chat worker raises if there are no connectors. Fix:
  ```bash
  fusion vault patch -p fusion/astra/access-gpu-product-all/gsf-app-deploy/stg \
    -s "CONNECTION_STRINGS=postgresql://gsf:<pwd>@postgres:5432/demo"
  ```
  Then restart ingestion (bump memory + `fusion deploy update`).
- **Secret change not taking effect** → ExternalSecret refresh lag. Default `refreshInterval` was `60m`; it's set to `1m` in `templates/fusion-externalsecret.yaml`. Pods read env at **startup only**, so after the secret updates you must restart the workload (memory bump).
- **`{"detail":"Conversation in progress"}` (HTTP 409)** → the app allows **one chat at a time**; another client (often an open browser tab) holds the slot. Wait for it to free or close the tab.
- **Chat loops then returns empty (`SQL could not be constructed`)** → catalog noise. The text-to-SQL agent thrashes when the catalog mixes business tables with the app's own internal tables. Keep the catalog **business-only** (see below).
- **Slow chat (~2-3 min)** → expected: cold-start of the warm-pool worker + multiple model round-trips. Not a bug.

## Demo business dataset (already deployed)

The catalog is a dedicated **`demo`** Postgres database with business tables in `public`
(`customers`, `products`, `employees`, `orders`, `order_items`). It is **isolated** from
the app's operational `gsf` DB so natural questions (e.g. "how many customers do we
have?") resolve cleanly. Implemented via three deploy-repo templates:

- `templates/demo-seed-configmap.yaml` — the demo SQL (idempotent).
- `templates/demo-seed-job.yaml` — PreSync hook: `CREATE DATABASE demo` + load the SQL.
- `templates/demo-reset-job.yaml` — PreSync hook: idempotently delete the stale `gsf`
  catalog subtree from Neo4j and its rows from the pgvector table.

`CONNECTION_STRINGS` points at `…/demo`. To re-run ingestion (e.g. after editing the
seed), bump ingestion memory and `fusion deploy update`; the PreSync hooks run, then
ingestion re-ingests on restart.

## Storage internals (for resets)

- **Catalog** lives in **Neo4j** as `Database`→`Schema`→`Table`→`Column` nodes joined by
  `CONTAINS`. The `Database` node has a `name` property. Re-ingestion MERGEs and never
  deletes stale nodes — delete a stale DB with:
  `MATCH (db:Database {name:'<db>'}) OPTIONAL MATCH (db)-[:CONTAINS*1..]->(c) DETACH DELETE db, c`
  (cypher-shell, creds `NEO4J_URI`/`NEO4J_USERNAME`/`NEO4J_PASSWORD`).
- **Embeddings** live in pgvector table `public.nv_ingest_tabular` in the **`gsf`** DB,
  with a `database_name` column. Ingestion self-cleans its *own* `database_name`; delete
  stale rows: `DELETE FROM nv_ingest_tabular WHERE database_name='<db>'`.

## Changing the catalog source (recipe)

To point the catalog at a different/clean DB without a long wait or stale entries:

1. Ensure the target DB exists and has only the tables you want (use a seed Job like
   `demo-seed-job.yaml`).
2. `fusion vault patch … -s "CONNECTION_STRINGS=…/<newdb>"`.
3. **Phase 1**: `fusion deploy update` (memory bump) to apply templates / `refreshInterval`.
   Wait ~3 min for ESO (1m interval) to pull the new value.
4. **Phase 2**: `fusion deploy update` (another memory bump) so the reset hook clears the
   old DB and ingestion re-ingests the new one (now that the secret has propagated).
5. Poll `/api/datasources/dbs` until it shows only the new DB.

The two-phase split avoids the race where ingestion restarts before ESO has the new
`CONNECTION_STRINGS`.

## Notes

- Commit messages: concise imperative, no `Co-Authored-By`.
- The app source repo is separate from this deploy repo; the demo seed/reset templates
  live only in the deploy repo (staging-demo specific).
