# Deploying Auto Ontology on Kubernetes

## Deploying a Release

A release is staged in `nvstaging/gsf`, then copied to the public
`nvidia` org by an NGC Catalog publishing merge request.

To install a published release:

1. Fetch the chart from NGC:

   ```bash
   helm fetch https://helm.ngc.nvidia.com/nvidia/charts/auto-ontology-1.0.0.tgz \
     --username='$oauthtoken' \
     --password=<API-KEY>
   ```

2. Install the chart:

   ```bash
   helm install auto-ontology auto-ontology-1.0.0.tgz \
     --set defaultModelsApiKey=<API-KEY> \
     --set postgresPassword=<POSTGRES-PASSWORD> \
     --set connectionStrings=<CONNECTION-STRINGS>
   ```

3. Expose the UI:

   ```bash
   kubectl port-forward frontend 3000:3000
   ```

## Staging / Astra nightly deployment

The live staging demo on Astra (cluster `astrastg01-ocp-pdx04`, namespace
`ns-auto-ontology-demo-deploy`) is GitOps-driven. GitHub and GitLab are **decoupled** and
meet only at the nvcr.io registry:

```mermaid
flowchart LR
  subgraph GitHub["GitHub — NVIDIA/auto-ontology (source + builds)"]
    A[staging-publish-image.yml<br/>backend/ingestion image]
    F[staging-publish-frontend.yml<br/>frontend image]
    C[staging-publish-chart.yml<br/>Helm chart]
  end
  NVCR[(nvcr.io/nvstaging/auto-ontology)]
  subgraph GitLab["GitLab — auto-ontology/auto-ontology-demo-deploy (deploy)"]
    R[ci/roll-stg.sh<br/>nightly schedule]
    V[deployment/stg/values.yaml<br/>umbrella Helm chart]
    W[build-wwi-seed<br/>seed image]
  end
  ARGO[ArgoCD on pdx04]
  A --> NVCR
  F --> NVCR
  C --> NVCR
  W --> NVCR
  NVCR -->|resolve newest stg.* digests| R
  R -->|edit + push main| V
  V -->|auto-sync ~3 min| ARGO
```

### How images are built (all on push, to nvcr.io)

| Image | Built by | Repo / tag |
|---|---|---|
| Backend + ingestion | GitHub `staging-publish-image.yml` | `nvcr.io/nvstaging/auto-ontology/auto-ontology:stg.<ts>` |
| Frontend (Next.js) | GitHub `staging-publish-frontend.yml` | `nvcr.io/nvstaging/auto-ontology/auto-ontology-frontend:stg.<ts>` |
| WWI demo-data seed | GitLab `build-wwi-seed` | `nvcr.io/nvstaging/auto-ontology/auto-ontology-wwi-seed:latest` |

CI validation (`ci-client.yml` runs `pnpm build`) only checks the frontend
compiles — it does **not** publish an image. Production frontend images come
from `staging-publish-frontend.yml` (or, ad hoc, `make publish-frontend`).

### The nightly roll (`roll-stg`)

Lives in the **deploy repo** at `ci/roll-stg.sh` and runs as a GitLab CI job on
a pipeline **schedule** (see that repo's README for the exact schedule + CI
variables). Each night it:

1. Resolves the newest `stg.*` backend and frontend images on nvcr.io + digests.
2. Refreshes the WWI seed digest.
3. Re-pins those in `deployment/stg/values.yaml` and bumps `auto_ontology.podAnnotations.rolledAt`.
4. Commits and pushes to `main` (with `-o ci.skip` so the bot commit starts no
   pipeline). ArgoCD auto-syncs within ~3 min — the app redeploys on the newest
   images and the WWI reset → seed → ingest hooks re-run.

It **builds nothing**; it only re-pins images that already exist on nvcr.io.

### Why the Helm chart is published separately (not "nightly")

The staging deployment **is** Helm-based: the deploy repo holds an umbrella Helm
chart that consumes the portable `auto_ontology` chart (`helm/auto-ontology/`) as an OCI dependency,
and ArgoCD renders/applies it. Two separate lifecycles, deliberately:

- **Chart publish** (`staging-publish-chart.yml`) runs only when the chart
  *structure* changes (push under `helm/auto-ontology/**`) or manually — never on a timer.
- **Nightly roll** (`roll-stg`) never touches the chart; it only edits the image
  pointers *inside* the chart's values.

So `roll-stg` does use Helm — indirectly, via the values ArgoCD renders — and the
chart is not published nightly.