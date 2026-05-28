#!/usr/bin/env bash
# Render an ingestion YAML template with env vars from .env, then run
# `metadata <subcmd> -c ...` inside a one-shot container started from the
# OpenMetadata ingestion image. The container is joined to the OM network
# so it can reach openmetadata-server:8585 directly.
#
# Usage: scripts/run-ingestion.sh <metadata|usage|autoclass> <evidence-log-path>
set -euo pipefail

KIND="${1:?kind required: metadata|usage|autoclass}"
EVIDENCE="${2:?evidence log path required}"

cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  echo "FATAL: .env not found" >&2
  exit 2
fi
# shellcheck disable=SC1091
set -a; source .env; set +a

case "$KIND" in
  metadata)        TMPL=ingestion/snowflake-metadata.yaml.tmpl;       SUBCMD=ingest ;;
  usage)           TMPL=ingestion/snowflake-usage.yaml.tmpl;          SUBCMD=usage ;;
  autoclass)       TMPL=ingestion/snowflake-autoclass.yaml.tmpl;      SUBCMD=classify ;;
  pii-metadata)    TMPL=ingestion/snowflake-pii-metadata.yaml.tmpl;   SUBCMD=ingest ;;
  pii-autoclass)   TMPL=ingestion/snowflake-pii-autoclass.yaml.tmpl;  SUBCMD=classify ;;
  *) echo "Unknown kind: $KIND" >&2; exit 2 ;;
esac

RENDERED="ingestion/.rendered-$KIND.yaml"
mkdir -p ingestion
envsubst < "$TMPL" > "$RENDERED"

IMAGE="docker.getcollate.io/openmetadata/ingestion:1.12.9"
NETWORK="openmetadata-eval_app_net"

# Build a sed program that scrubs the password, the bot JWT, the username,
# and the account locator out of *everything* before it hits the log file.
SCRUB="
  s|${SNOWFLAKE_PASSWORD}|<sf_password>|g;
  s|${OM_TOKEN}|<om_token>|g;
  s|${SNOWFLAKE_USER}|<sf_user>|g;
  s|${SNOWFLAKE_ACCOUNT}|<sf_account>|g;
  s/(password|jwtToken):.*/\\1: <redacted>/;
"

mkdir -p "$(dirname "$EVIDENCE")"
{
  echo "=== run-ingestion.sh kind=$KIND subcmd=$SUBCMD at $(date -u +%FT%TZ) ==="
  echo "--- rendered YAML (secrets scrubbed) ---"
  sed -E "$SCRUB" "$RENDERED"
  echo "--- end YAML ---"
  echo
  echo "=== metadata $SUBCMD output ==="
  set +e
  docker run --rm \
    --network "$NETWORK" \
    -v "$(pwd)/ingestion:/work" \
    --entrypoint metadata \
    "$IMAGE" \
    "$SUBCMD" -c "/work/$(basename "$RENDERED")" 2>&1
  RC=$?
  set -e
  echo "=== exit_code=$RC ==="
  exit $RC
} | sed -E "$SCRUB" | tee "$EVIDENCE"
