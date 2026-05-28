#!/usr/bin/env bash
# Check 3b verify step: GET TPCH_SF1.CUSTOMER columns + tags, look for any
# tag whose classification is "PII".
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .env

AUTH=(-H "Authorization: Bearer $OM_TOKEN")
TABLE_FQN="snowflake_eval.SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.CUSTOMER"

curl -sS "${AUTH[@]}" "$OM_API/v1/tables/name/$TABLE_FQN?fields=columns,tags" \
  | tee evidence/03b-customer-tags.json \
  | jq '{table: .fullyQualifiedName,
         tableTags: .tags,
         columnTags: [.columns[] | select((.tags // []) | length > 0) | {name, tags}]}'

echo
echo "=== PII tags on any column of any ingested table (PII.* only) ==="
TABLES=$(curl -sS "${AUTH[@]}" "$OM_API/v1/tables?database=snowflake_eval.SNOWFLAKE_SAMPLE_DATA&fields=columns&limit=200" \
  | jq -c '[.data[] | {fqn:.fullyQualifiedName,
                       piiCols:[.columns[] | select((.tags//[])[] | (.tagFQN // "") | startswith("PII.")) | {name, tags: .tags}]}
            | select(.piiCols | length>0)]')
echo "$TABLES" | tee evidence/03b-all-pii-tags.json | jq .
