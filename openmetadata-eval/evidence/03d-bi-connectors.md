# 3d — PowerBI / Tableau lineage (connector presence + config + prerequisites)

No live BI accounts were available for this eval. This check instead verifies:

1. Both connectors ship in the OpenMetadata `ingestion` image we already pulled (`1.12.9`).
2. The connection-config JSON Schemas for each (extracted from the pydantic models in the image).
3. A minimal valid ingestion YAML for each.
4. The end-to-end prerequisites that have to be true on the BI side, *and* in OM, for lineage to land.

## Connector presence (raw evidence)

Listed the dashboard source plug-ins shipped inside `docker.getcollate.io/openmetadata/ingestion:1.12.9`:

```text
domodashboard  grafana  hex  lightdash  looker  metabase  microstrategy
mode  powerbi  qlikcloud  qliksense  quicksight  redash  sigma  ssrs
superset  tableau
```

`powerbi/` and `tableau/` are both present. The full set of plugins above can also be enumerated with `pip show 'openmetadata-ingestion[powerbi,tableau]'` against the same image.

## Config schemas

Auto-extracted from the connection pydantic models inside the ingestion image; full JSON Schema is in `03d-bi-schemas.json`. The compact field-level breakdown follows.

### PowerBI — `PowerBIConnection`

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `PowerBiType` |  | Service type discriminator. |
| `clientId` | string | yes | Azure AD app `client_id`. |
| `clientSecret` | string | yes | Azure AD app client secret. |
| `tenantId` | string | yes | Azure AD tenant ID. |
| `apiURL` | string |  | PowerBI REST host. Default `https://api.powerbi.com`. Override for sovereign clouds. |
| `authorityURI` | string |  | AAD authority URI (e.g. `https://login.microsoftonline.com/`). |
| `hostPort` | string |  | PowerBI service base URL — used to build dashboard hyperlinks. |
| `scope` | array |  | OAuth scopes; default `https://analysis.windows.net/powerbi/api/.default`. |
| `pagination_entity_per_page` | integer |  | Page size for paginated PowerBI API calls. |
| `useAdminApis` | boolean |  | Toggle Admin APIs path (richer metadata + lineage scan) vs non-admin. |
| `displayTableNameFromSource` | boolean |  | Use source-database table names instead of PowerBI renamed names. |
| `pbitFilesSource` | one of `LocalConfig` / `AzureConfig` / `GCSConfig` / `S3Config` |  | Optional .pbit-file source for lineage when the API lineage isn't enough. |
| `dashboardFilterPattern` / `chartFilterPattern` / `dataModelFilterPattern` / `projectFilterPattern` | `FilterPattern` |  | Include/exclude regex filters per entity type. |

### Tableau — `TableauConnection`

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `TableauType` |  | Service type discriminator. |
| `hostPort` | string | yes | Tableau Server / Tableau Cloud URL. |
| `authType` | `BasicAuth` or `AccessTokenAuth` | yes (one of) | Username/password or Personal Access Token (preferred). |
| `siteName` | string |  | Tableau site `contentUrl` (the slug after `/site/` in the URL). Leave blank for default site on Tableau Server. |
| `paginationLimit` | integer |  | Metadata-API page size. |
| `apiVersion` | string |  | Tableau REST API version. Defaults to server version. |
| `proxyURL` | string |  | Public URL used to build hyperlinks back to dashboards. |
| `verifySSL` / `sslConfig` | enums + config |  | TLS verification setup. |
| `dashboardFilterPattern` / `chartFilterPattern` / `dataModelFilterPattern` / `projectFilterPattern` | `FilterPattern` |  | Same filter pattern shape as the rest of the connectors. |

> **Note:** `siteUrl` was *removed* in OM 1.7.4 / 1.7.5 — only `siteName` (a.k.a. `contentUrl`) is accepted now. Older docs that mention `siteUrl` are stale.

## Minimal ingestion YAMLs

### PowerBI (service-principal auth, Admin API path, includes optional db lineage hook)

```yaml
source:
  type: powerbi
  serviceName: prod_powerbi
  serviceConnection:
    config:
      type: PowerBI
      clientId: ${POWERBI_CLIENT_ID}
      clientSecret: ${POWERBI_CLIENT_SECRET}
      tenantId: ${POWERBI_TENANT_ID}
      scope:
        - https://analysis.windows.net/powerbi/api/.default
      useAdminApis: true
      pagination_entity_per_page: 100
      hostPort: https://app.powerbi.com
  sourceConfig:
    config:
      type: DashboardMetadata
      lineageInformation:
        dbServiceNames:
          - snowflake_eval        # name of any already-ingested DatabaseService in OM
      dashboardFilterPattern:
        excludes:
          - ^My workspace$
sink:
  type: metadata-rest
  config: {}
workflowConfig:
  loggerLevel: INFO
  openMetadataServerConfig:
    hostPort: http://openmetadata-server:8585/api
    authProvider: openmetadata
    securityConfig:
      jwtToken: ${OM_TOKEN}
```

### Tableau (PAT auth)

```yaml
source:
  type: tableau
  serviceName: prod_tableau
  serviceConnection:
    config:
      type: Tableau
      hostPort: https://prod-useast-a.online.tableau.com
      siteName: my_site_content_url        # blank for default site on on-prem Server
      authType:
        personalAccessTokenName: ${TABLEAU_PAT_NAME}
        personalAccessTokenSecret: ${TABLEAU_PAT_SECRET}
      apiVersion: "3.21"
      verifySSL: validate
  sourceConfig:
    config:
      type: DashboardMetadata
      lineageInformation:
        dbServiceNames:
          - snowflake_eval
      dashboardFilterPattern:
        includes:
          - ^Finance.*
sink:
  type: metadata-rest
  config: {}
workflowConfig:
  loggerLevel: INFO
  openMetadataServerConfig:
    hostPort: http://openmetadata-server:8585/api
    authProvider: openmetadata
    securityConfig:
      jwtToken: ${OM_TOKEN}
```

## Prerequisites & gotchas

### PowerBI

- **License:** PowerBI **Pro** (or Premium-per-User) for both the service principal and at least one tenant admin who will configure it.
- **Tenant admin settings (Power BI Admin Portal → Tenant settings → Developer settings):**
  - *Allow service principals to use Power BI APIs* — required.
  - *Allow service principals to use read-only Power BI admin APIs* — required for `useAdminApis: true`.
  - *Enhance admin APIs responses with detailed metadata* — needed for column-level lineage.
- **Azure AD app registration:**
  - Create app → record `clientId`, `tenantId`. Generate a client secret → that's `clientSecret`.
  - Under **API permissions** add **PowerBI Service** delegated/app perms:
    - `Dashboard.Read.All` (required).
    - `Dataset.Read.All` (optional — without it, dataset/lineage is skipped).
  - Grant admin consent for those perms.
  - **Authentication** tab → enable *Allow public client flows*.
  - Do **not** grant tenant-wide AAD permissions (the app only needs PowerBI scope).
- **Workspace membership:** the service principal must be added (Admin or Member) to every PowerBI workspace you want ingested. `My workspace` is not supported.
- **OM-side:** the database service for any upstream warehouse (e.g. Snowflake, Synapse) must already be ingested. Then list that service name under `sourceConfig.config.lineageInformation.dbServiceNames` — that's how PowerBI dataset tables get joined to OM tables and lineage edges get drawn.
- **Known limitation:** PowerBI **usage** ingestion is not supported (PowerBI Usage API can't be called by a service principal).
- **Optional:** for `.pbit` lineage, configure `pbitFilesSource` to point at a local path / S3 / GCS / Azure container of `.pbit` exports.

### Tableau

- **Server requirement:** the **Tableau Metadata API** must be enabled. On Tableau Cloud it is on by default; on Tableau Server it must be enabled by an admin via `tsm maintenance metadata-services enable`.
- **Auth:** prefer a **Personal Access Token** (`AccessTokenAuth`) over username/password — PATs respect MFA/SSO and can be rotated. Basic auth fails on any SSO-enforced site.
- **Service-account role:**
  - Minimum *Site role: Viewer* to ingest dashboards/charts/datasources.
  - Minimum *Site Admin Explorer* to also ingest **owner** information.
  - The account must have access to each project/workbook you want to ingest — Tableau permission scoping applies.
- **Site config:** for a **default** site on Tableau Server, leave `siteName` blank. For non-default or Tableau Cloud sites, set `siteName` to the `contentUrl` slug (the bit after `/site/` in the URL, e.g. `acme_finance`). Do **not** use `*`.
- **Removed field:** `siteUrl` was deprecated in 1.7.4/1.7.5 — do not include it.
- **OM-side lineage:** identical pattern to PowerBI — the upstream `DatabaseService` must already be in OM; list it under `sourceConfig.config.lineageInformation.dbServiceNames`. Tableau exposes the SQL behind data sources via the Metadata API; OM parses it with sqlfluff/sqlglot to build column-level lineage.
- **Network:** when running the connector behind a corporate proxy, set `proxyURL` so dashboard hyperlinks generated in OM resolve correctly.

## What would prove this end-to-end (not done here)

1. Provision a Tableau Cloud trial site and a single PowerBI Pro tenant + AAD app.
2. Run both YAMLs above with valid creds.
3. Confirm dashboard/chart entities show up via `GET /api/v1/dashboards?service=<name>` and that the lineage edges show up via `GET /api/v1/lineage/dashboard/name/<fqn>?upstreamDepth=2`.

This is gated on having BI accounts; the connectors and their schemas are otherwise verified to exist and to be wired into the 1.12.9 build.
