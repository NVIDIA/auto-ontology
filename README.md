# GSF

Generative Semantic Fabric adds the structured-data ontology layer to any partner or NVidia agent harness  interface, like NVIDIA AI-Q Claws, etc

> **Licensing & contributions.** GSF is distributed under the
> [Apache License 2.0](./LICENSE). Third-party open-source components
> bundled, linked, or otherwise used by this project are listed in
> [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md). **This project
> is currently not accepting external contributions.**

## Deployment From NVStaging

1. Fetch the chart from NGC:

   ```bash
   helm fetch https://helm.ngc.nvidia.com/nvstaging/gsf/charts/gsf-0.0.1.tgz \
     --username='$oauthtoken' \
     --password=<API-KEY>
   ```

2. Create the nvcr.io image-pull secret:

   ```bash
   kubectl create secret docker-registry nvcr-creds \
     --docker-server=nvcr.io \
     --docker-username='$oauthtoken' \
     --docker-password=<API-KEY>
   ```

3. Attach the pull secret to the default ServiceAccount so pods inherit it:

   ```bash
   kubectl patch serviceaccount default \
     -p '{"imagePullSecrets":[{"name":"nvcr-creds"}]}'
   ```

4. Install the chart:

   ```bash
   helm install gsf gsf-0.0.1.tgz \
     --set nvidiaApiKey=<API-KEY> \
     --set neo4jPassword=<NEO4J-PASSWORD> \
     --set postgresPassword=<POSTGRES-PASSWORD> \
     --set connectionStrings=<CONNECTION-STRINGS>
   ```

5. Expose the UI:
```bash
kubectl port-forward frontend 3000:3000
```
End-to-end build, install and verification steps live in
[`helm/gsf/README.md`](helm/gsf/README.md). TL;DR for a local Docker Desktop
Kubernetes cluster:

## License

GSF is licensed under the [Apache License, Version 2.0](./LICENSE).
SPDX identifier: `Apache-2.0`.

Each NVIDIA-authored source file in this repository carries an SPDX header
of the form:

```text
SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
```

Third-party open-source components used by GSF are enumerated, with their
upstream licenses and project URLs, in
[`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md).

## Contributing

**This project is currently not accepting contributions.** Issues, pull
requests, and patches submitted from outside the GSF maintainer team will
not be reviewed or merged. Security-relevant reports should follow the
process described in [`SECURITY.md`](./SECURITY.md).
