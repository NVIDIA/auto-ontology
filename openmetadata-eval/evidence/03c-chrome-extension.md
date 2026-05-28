# 3c — Chrome Extension (MANUAL)

The Chrome Web Store extension cannot be installed or driven by code, so this check is documented manually and tagged **MANUAL** in the final report.

## Identity (verified from the Chrome Web Store)

| Field | Value |
| --- | --- |
| Listing name | OpenMetadata |
| Publisher | `collate-openmetadata-ext` (Collate Inc., 200 Middlefield Rd Suite 110, Menlo Park, CA 94025-4003 US) |
| Contact | collate-openmetadata-ext@getcollate.io  •  +1 704-807-1661 |
| Extension ID | `pakbbdhbbiclnceabdmnghamabjloofc` |
| Chrome Web Store URL | https://chromewebstore.google.com/detail/openmetadata/pakbbdhbbiclnceabdmnghamabjloofc |

> The newer "latest" docs reference a different extension id `ndjnpiadedlmgddlpeklbnobebkpkdgb` for the SSO redirect URL. The Web Store listing (`pakbb…`) is the one currently published; the docs appear inconsistent and this should be re-checked once Collate confirms which is the active build.

## What it claims to do

From the Web Store description and `docs.open-metadata.org/v1.12.x/how-to-guides/guide-for-data-users/browser-ext`:

- Surfaces ownership, description, tags, glossary terms, schema, lineage, and custom properties for the data asset under the user's cursor in a third-party tool (e.g. Looker, Tableau, BigQuery console).
- Activity feed view: shows mentions and assigned tasks from the OM instance.
- Lets the user reply on conversation threads from inside the extension.
- Requires the user to provide their OpenMetadata base URL on first launch, then signs in via that instance's auth provider (basic / Google / Okta / Auth0 / SAML / LDAP).

## Install / configure (manual steps — verifier reproduces locally)

1. Open `https://chromewebstore.google.com/detail/openmetadata/pakbbdhbbiclnceabdmnghamabjloofc` in Chrome.
2. Click **Add to Chrome** → **Add extension** in the popup.
3. Pin the extension via the puzzle-piece menu so the OpenMetadata icon is always visible.
4. Click the icon. Enter the base URL — for this eval `http://localhost:8585`. Click **Connect**.
5. Sign in. The default quickstart admin is `admin@open-metadata.org` / `admin`.
6. After auth, the popup renders the user's Activity Feed pulled from `/api/v1/feed`.

## SSO redirect URL (for any future hosted/SSO deployment)

If OM is fronted by SSO, the SSO provider's allowed-redirect list must include:

- `https://pakbbdhbbiclnceabdmnghamabjloofc.chromiumapp.org/auth0` (per `v1.12.x` docs)
- or `https://ndjnpiadedlmgddlpeklbnobebkpkdgb.chromiumapp.org/auth0` (per `latest` docs — see note above)

## Why this check is MANUAL

- Installation requires interacting with the Chrome Web Store UI, which has no public REST/CLI surface.
- The extension talks to the same `/api/v1` endpoints we already validated programmatically in checks 3a, 3b, 3e, 3g — so its features are essentially a thin UI over already-tested APIs. A working API + a published, signed extension on the Web Store is therefore strong indirect evidence that this feature works.

## Open questions for the verifier

1. Confirm with Collate which extension ID is current (`pakbb…` vs `ndjn…`).
2. Confirm that the extension supports the org's SSO provider (Okta / Auth0 / Google Workspace). If we run a hosted OM behind SAML, add the right `chromiumapp.org/auth0` redirect URL to the IdP allowlist before users install.
3. Confirm the data path is direct browser→OM (no Collate-hosted proxy in between) — important for orgs that won't let metadata leave their VPC.
