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
