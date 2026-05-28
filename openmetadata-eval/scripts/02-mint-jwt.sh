#!/usr/bin/env bash
# Log in as default admin, fetch the ingestion-bot user, then grab its JWT
# via /users/auth-mechanism/{id}. Persist into .env as OM_TOKEN.
#
# Default basic-auth admin on quickstart: admin@open-metadata.org / admin
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .env

ADMIN_EMAIL="${ADMIN_EMAIL:-admin@open-metadata.org}"
ADMIN_PASSWORD_B64=$(printf '%s' "${ADMIN_PASSWORD:-admin}" | base64)

echo "1. POST ${OM_API}/v1/users/login"
LOGIN_BODY=$(jq -nc --arg e "$ADMIN_EMAIL" --arg p "$ADMIN_PASSWORD_B64" \
  '{email:$e, password:$p}')
LOGIN_RESP=$(curl -s -X POST "${OM_API}/v1/users/login" \
  -H 'Content-Type: application/json' \
  -d "$LOGIN_BODY")
ADMIN_TOKEN=$(echo "$LOGIN_RESP" | jq -r '.accessToken // empty')
if [[ -z "$ADMIN_TOKEN" ]]; then
  echo "FATAL: login failed. Response:" >&2
  echo "$LOGIN_RESP" | jq . >&2 || echo "$LOGIN_RESP" >&2
  exit 1
fi
echo "   got admin accessToken (len=${#ADMIN_TOKEN})"

echo "2. GET ${OM_API}/v1/users/name/ingestion-bot"
BOT=$(curl -s -H "Authorization: Bearer $ADMIN_TOKEN" \
  "${OM_API}/v1/users/name/ingestion-bot")
BOT_ID=$(echo "$BOT" | jq -r '.id // empty')
if [[ -z "$BOT_ID" ]]; then
  echo "FATAL: could not find ingestion-bot user. Response:" >&2
  echo "$BOT" | jq . >&2 || echo "$BOT" >&2
  exit 1
fi
echo "   ingestion-bot user id: $BOT_ID"

echo "3. GET ${OM_API}/v1/users/auth-mechanism/$BOT_ID"
AUTH=$(curl -s -H "Authorization: Bearer $ADMIN_TOKEN" \
  "${OM_API}/v1/users/auth-mechanism/$BOT_ID")
JWT=$(echo "$AUTH" | jq -r '.config.JWTToken // empty')
if [[ -z "$JWT" ]]; then
  echo "FATAL: no JWT on bot. Response:" >&2
  echo "$AUTH" | jq . >&2 || echo "$AUTH" >&2
  exit 1
fi
echo "   got bot JWT (len=${#JWT}, expires=$(echo "$AUTH" | jq -r '.config.JWTTokenExpiresAt'))"

if grep -q '^OM_TOKEN=' .env; then
  awk -v t="$JWT" 'BEGIN{FS=OFS="="} /^OM_TOKEN=/{$0="OM_TOKEN="t} {print}' .env > .env.tmp && mv .env.tmp .env
else
  printf 'OM_TOKEN=%s\n' "$JWT" >> .env
fi
chmod 600 .env
echo "Wrote OM_TOKEN to .env"
