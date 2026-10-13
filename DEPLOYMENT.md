<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deploying Auto Ontology on Kubernetes

## Released artifacts

Releases are published to the public NGC Catalog under the `nvidia` org:

| Artifact | Location |
|---|---|
| Backend + ingestion image | `nvcr.io/nvidia/auto-ontology:1.1.0` |
| Frontend (Next.js) image | `nvcr.io/nvidia/auto-ontology-frontend:1.1.0` |
| MCP server image | `nvcr.io/nvidia/auto-ontology-mcp:1.1.0` |
| Helm chart | `https://helm.ngc.nvidia.com/nvidia/charts/auto-ontology-1.1.0.tgz` (also listed in the NGC Catalog) |

## Deploying a Release

To install a published release:

1. Fetch the chart from NGC:

   ```bash
   helm fetch https://helm.ngc.nvidia.com/nvidia/charts/auto-ontology-1.1.0.tgz
   ```

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
   helm install auto-ontology auto-ontology-1.1.0.tgz -f auto-ontology-secrets.yaml
   ```

   `adminEmail` and `adminPassword` are required: self-service sign-up is
   disabled, so they create the first admin, and the install fails without
   them. If users will reach the app somewhere other than
   `http://localhost:3000` (the port-forward below), also set `appUrl` to that
   exact origin.

4. Expose the UI:

   ```bash
   kubectl port-forward svc/frontend 3000:3000
   ```
