## PowerBI — `PowerBIConnection`

**Title:** PowerBIConnection

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `PowerBiType` | null |  | Service Type |
| `clientId` | string | yes | client_id for PowerBI. |
| `clientSecret` | string | yes | clientSecret for PowerBI. |
| `tenantId` | string | yes | Tenant ID for PowerBI. |
| `apiURL` | string | null |  | API URL to call powerbi rest apis to extract metadata. Default to `https://api.powerbi.com`. You can provide youw own in case of different e… |
| `authorityURI` | string | null |  | Authority URI for the PowerBI service. |
| `hostPort` | string | null |  | Dashboard URL for PowerBI service. |
| `scope` | array | null |  | PowerBI secrets. |
| `pagination_entity_per_page` | integer | null |  | Entity Limit set here will be used to paginate the PowerBi APIs |
| `useAdminApis` | boolean | null |  | Fetch the PowerBI metadata using admin APIs |
| `displayTableNameFromSource` | boolean | null |  | Display Table Name from source instead of renamed table name for datamodel tables |
| `pbitFilesSource` | `LocalConfig` | `AzureConfig` | `GCSConfig` | `S3Config` | null |  | Source to get the .pbit files to extract lineage information |
| `dashboardFilterPattern` | `FilterPattern` | null |  | Regex to exclude or include dashboards that matches the pattern. |
| `chartFilterPattern` | `FilterPattern` | null |  | Regex exclude or include charts that matches the pattern. |
| `dataModelFilterPattern` | `FilterPattern` | null |  | Regex exclude or include data models that matches the pattern. |
| `projectFilterPattern` | `FilterPattern` | null |  | Regex to exclude or include projects that matches the pattern. |
| `supportsMetadataExtraction` | `SupportsMetadataExtraction` | null |  |  |

## Tableau — `TableauConnection`

**Title:** TableauConnection

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `TableauType` | null |  | Service Type |
| `hostPort` | string | yes | Tableau Server url. |
| `authType` | `BasicAuth` | `AccessTokenAuth` | null |  | Types of methods used to authenticate to the tableau instance |
| `siteName` | string | null |  | Tableau Site Name. |
| `paginationLimit` | integer | null |  | Pagination limit used while querying the tableau metadata API for getting data sources |
| `apiVersion` | string | null |  | Tableau API version. If not provided, the version will be used from the tableau server. |
| `proxyURL` | string | null |  | Proxy URL for the tableau server. If not provided, the hostPort will be used. This is used to generate the dashboard & Chart URL. |
| `verifySSL` | `VerifySSL` | null |  |  |
| `sslConfig` | `SslConfig` | null |  |  |
| `dashboardFilterPattern` | `FilterPattern` | null |  | Regex to exclude or include dashboards that matches the pattern. |
| `chartFilterPattern` | `FilterPattern` | null |  | Regex exclude or include charts that matches the pattern. |
| `dataModelFilterPattern` | `FilterPattern` | null |  | Regex exclude or include data models that matches the pattern. |
| `projectFilterPattern` | `FilterPattern` | null |  | Regex to exclude or include projects that matches the pattern. |
| `supportsMetadataExtraction` | `SupportsMetadataExtraction` | null |  |  |

