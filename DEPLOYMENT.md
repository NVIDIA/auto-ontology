<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deploying Auto-Ontology on Kubernetes

## Released artifacts

Releases are published to the public NGC Catalog under the `nvidia` org:

| Artifact | Location |
|---|---|
| Backend + ingestion image | `nvcr.io/nvidia/auto-ontology` |
| Frontend (Next.js) image | `nvcr.io/nvidia/auto-ontology-frontend` |
| MCP server image | `nvcr.io/nvidia/auto-ontology-mcp` |
| Helm chart | `auto-ontology` in the `https://helm.ngc.nvidia.com/nvidia` Helm repo (also listed in the NGC Catalog) |

Each release tags its images with the chart version, and the chart pins its
own images to that tag.

## Deploying a Release

To install the latest published release:

1. Add the NGC Helm repo:

   ```bash
   helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
   helm repo update
   ```

   `helm search repo nvidia/auto-ontology --versions` lists the available
   releases.

2. Put the secrets in a values file. Create it in an editor rather than on
   the command line, so they stay out of shell history and process arguments,
   restrict it with `chmod 600`, and keep it out of git:

   ```yaml
   # auto-ontology-secrets.yaml
   defaultModelsApiKey: <API-KEY>
   postgresPassword: <POSTGRES-PASSWORD>
   adminEmail: <ADMIN-EMAIL>
   adminPassword: <ADMIN-PASSWORD>
   connectionStrings: <CONNECTION-STRINGS>   # optional; or add connections in the UI
   ```

3. Install the chart:

   ```bash
   helm install auto-ontology nvidia/auto-ontology -f auto-ontology-secrets.yaml
   ```

   This installs the latest release; add `--version <version>` to pin one.
   `helm list` shows which chart version was installed.

   `adminEmail` and `adminPassword` are required: self-service sign-up is
   disabled, so they create the first admin, and the install fails without
   them. If users will reach the app somewhere other than
   `http://localhost:3000` (the port-forward below), also set `appUrl` to that
   exact origin.

4. Expose the UI:

   ```bash
   kubectl port-forward svc/frontend 3000:3000
   ```

## Using Elasticsearch as the vector store

By default entity vectors live in Postgres (pgvector) and the app embeds them
itself. Set `vectorStore.backend: elasticsearch` to keep them in Elasticsearch
instead: rows go into a `semantic_text` index, and Elasticsearch embeds rows
and queries through an `nvidia` inference endpoint that the app creates from
the same model settings (`defaultModelsApiKey`, `embedModel`). Postgres is
still deployed and still holds the catalog.

Elasticsearch needs to be 9.3+ or Serverless; hybrid search needs 9.5+ or
Serverless. Self-managed clusters need an Enterprise or trial license for the
`nvidia` inference service (on a basic license creating the endpoint fails
with `403 current license is non-compliant for [inference]`). Use a
build.nvidia.com (`nvapi-`) key, or another
endpoint whose embedding model honours `input_type`.

**An existing cluster** (Elastic Cloud, Serverless, ECK or self-managed). Add
to the values file:

```yaml
vectorStore:
  backend: elasticsearch
  searchMode: semantic        # or hybrid
elasticsearch:
  url: https://my-deployment.es.us-east-1.aws.elastic.cloud:443
  apiKey: <ES-API-KEY>        # or username / password
  # For a cluster with a private CA, e.g. ECK's <cluster>-es-http-certs-public:
  # caSecret:
  #   name: <SECRET-NAME>
  #   key: ca.crt
```

The API key needs to manage the inference endpoint and the
`auto-ontology-*` indices:

```json
{
  "cluster": ["manage_inference", "monitor_inference"],
  "indices": [{ "names": ["auto-ontology-*"], "privileges": ["all"] }]
}
```

**A bundled single-node cluster**, for trying it out:

```yaml
vectorStore:
  backend: elasticsearch
elasticsearch:
  bundled:
    enabled: true
```

This runs Elasticsearch next to Postgres with security on and a generated
`elastic` password. It uses a 30-day trial license, so the `nvidia` inference
service, and with it ingest and search, stops when the trial ends. For anything
longer-lived, point `url` at a licensed cluster.

The chart does not restart pods when only these settings change, so after a
`helm upgrade` that switches `vectorStore` or `elasticsearch` values, restart
the app (or bump `podAnnotations`):

```bash
kubectl rollout restart deploy/<release> deploy/ingestion
```

Switching an existing install from Postgres to Elasticsearch does not move its
vectors. Start from a new install, or remove and re-add each connection so it
is ingested into the new index, then reset the semantic layer.

The endpoint keeps the model and key it was created with. To change either,
delete the endpoint (`DELETE _inference/text_embedding/auto-ontology-embed`)
and, for a new model, the `auto-ontology-*` indices, then re-ingest.

### Running it locally

`docker-compose.yml` has the same single-node cluster behind the
`elasticsearch` profile, and `dev_tools/setup_env.sh` starts it when `.env` has
`VDB_BACKEND=elasticsearch`; see [`dev_tools/README.md`](./dev_tools/README.md).
