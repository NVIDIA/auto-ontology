#!/usr/bin/env bash
# Check 3a: classifications & tags.
#  1. GET /classifications        -> evidence/03a-classifications.json
#  2. POST /classifications       create "Sensitivity"
#  3. POST /tags                  create tag "Internal" under Sensitivity
#  4. PATCH column on TPCH_SF1.CUSTOMER.C_NAME -> apply Sensitivity.Internal
#  5. GET the table back and confirm the tag stuck.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .env

AUTH=(-H "Authorization: Bearer $OM_TOKEN")
JSON=(-H 'Content-Type: application/json')
PATCH=(-H 'Content-Type: application/json-patch+json')

EVID=evidence
mkdir -p "$EVID"

echo "== 1. GET classifications =="
curl -sS "${AUTH[@]}" "$OM_API/v1/classifications?fields=termCount&limit=100" \
  | tee "$EVID/03a-classifications.json" | jq '.data[] | {name, description, provider}'

echo "== 2. POST classification Sensitivity =="
# Idempotent: if it already exists, OM responds 409 — accept that.
CREATE_RESP=$(curl -sS -o /tmp/sens-class.json -w '%{http_code}' \
  "${AUTH[@]}" "${JSON[@]}" \
  -X POST "$OM_API/v1/classifications" \
  -d '{"name":"Sensitivity","description":"Eval-created classification for data sensitivity labels.","mutuallyExclusive":false}')
echo "  HTTP $CREATE_RESP"
cat /tmp/sens-class.json | tee "$EVID/03a-classification-create.json" | jq .
echo

echo "== 3. POST tag Sensitivity.Internal =="
TAG_RESP=$(curl -sS -o /tmp/internal-tag.json -w '%{http_code}' \
  "${AUTH[@]}" "${JSON[@]}" \
  -X POST "$OM_API/v1/tags" \
  -d '{"name":"Internal","description":"Restricted to internal employees.","classification":"Sensitivity"}')
echo "  HTTP $TAG_RESP"
cat /tmp/internal-tag.json | tee "$EVID/03a-tag-create.json" | jq .
echo

echo "== 4. Locate target column =="
TABLE_FQN="snowflake_eval.SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.CUSTOMER"
TABLE=$(curl -sS "${AUTH[@]}" "$OM_API/v1/tables/name/$TABLE_FQN?fields=columns,tags")
TABLE_ID=$(echo "$TABLE" | jq -r '.id')
COL_INDEX=$(echo "$TABLE" | jq -r '.columns | to_entries[] | select(.value.name=="C_NAME") | .key')
echo "  table id: $TABLE_ID  C_NAME index: $COL_INDEX"

echo "== 5. PATCH column to add Sensitivity.Internal =="
PATCH_BODY=$(jq -nc --argjson i "$COL_INDEX" '
  [
    { "op":"add",
      "path": ("/columns/"+($i|tostring)+"/tags"),
      "value": [
        { "tagFQN":"Sensitivity.Internal",
          "source":"Classification",
          "labelType":"Manual",
          "state":"Confirmed" }
      ]
    }
  ]')
echo "  patch body: $PATCH_BODY"
PATCH_RESP=$(curl -sS -o /tmp/patch-resp.json -w '%{http_code}' \
  "${AUTH[@]}" "${PATCH[@]}" \
  -X PATCH "$OM_API/v1/tables/$TABLE_ID" \
  -d "$PATCH_BODY")
echo "  HTTP $PATCH_RESP"
cat /tmp/patch-resp.json | tee "$EVID/03a-column-patch.json" \
  | jq '.columns[] | select(.name=="C_NAME") | {name, tags}'

echo "== 6. Re-GET column to confirm tag persisted =="
curl -sS "${AUTH[@]}" "$OM_API/v1/tables/name/$TABLE_FQN?fields=columns,tags" \
  | tee "$EVID/03a-column-verify.json" \
  | jq '.columns[] | select(.name=="C_NAME") | {name, tags}'
