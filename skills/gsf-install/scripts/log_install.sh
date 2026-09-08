#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Append-only install log. Wrap the real command; do not invent flags here.
#
#   log_install.sh [--model NAME] -- command [args...]
#   GSF_INSTALL_MODEL=... GSF_INSTALL_LOG=/tmp/gsf.log log_install.sh -- uv sync

set -euo pipefail

usage() {
	cat <<'EOF'
Usage:
  log_install.sh [--model NAME] -- command [args...]

Appends timestamp, optional model, cwd, the exact command, redacted env,
and exit code to .gsf-install.log at the git repo root (or $GSF_INSTALL_LOG).
EOF
}

MODEL="${GSF_INSTALL_MODEL:-}"
while [[ $# -gt 0 ]]; do
	case "$1" in
	--model)
		MODEL="${2:-}"
		shift 2
		;;
	-h | --help)
		usage
		exit 0
		;;
	--)
		shift
		break
		;;
	*)
		break
		;;
	esac
done

if [[ $# -eq 0 ]]; then
	usage >&2
	exit 2
fi

if [[ -n "${GSF_INSTALL_LOG:-}" ]]; then
	LOG_FILE="$GSF_INSTALL_LOG"
elif ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"; then
	LOG_FILE="$ROOT/.gsf-install.log"
else
	LOG_FILE="$PWD/.gsf-install.log"
fi

redact_value() {
	local key="$1" value="$2"
	local upper
	upper="$(printf '%s' "$key" | tr '[:lower:]' '[:upper:]')"
	case "$upper" in
	*PASSWORD* | *SECRET* | *TOKEN* | *API_KEY* | *CONNECTION_STRINGS* | *ROLE_ID* | *SECRET_ID*)
		if [[ -n "$value" ]]; then
			printf '%s' '***'
		fi
		;;
	*)
		printf '%s' "$value"
		;;
	esac
}

{
	echo "-----"
	echo "timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
	echo "model=${MODEL:-}"
	echo "cwd=$PWD"
	printf 'command='
	printf '%q ' "$@"
	echo
	echo "env:"
	for key in \
		APP_URL \
		AUTH_SECRET \
		POSTGRES_HOST \
		POSTGRES_PORT \
		POSTGRES_USER \
		POSTGRES_PASSWORD \
		POSTGRES_DATABASE \
		CONNECTION_STRINGS \
		DEFAULT_MODELS_ENDPOINT \
		DEFAULT_MODELS_MODEL \
		DEFAULT_MODELS_API_KEY \
		REASONING_MODEL \
		NON_REASONING_MODEL \
		EMBED_ENDPOINT \
		EMBED_MODEL \
		RERANK_MODEL \
		INGESTION_SERVICE_URL \
		PYTHON_API_URL \
		GSF_API_URL \
		GSF_MCP_HOST \
		GSF_MCP_PORT \
		GSF_MCP_PUBLIC_URL \
		VAULT_ADDR \
		VAULT_NAMESPACE \
		VAULT_ROLE_ID \
		VAULT_SECRET_ID \
		GSF_ADMIN_EMAIL \
		GSF_ADMIN_PASSWORD; do
		if [[ -n "${!key+x}" ]]; then
			printf '  %s=%s\n' "$key" "$(redact_value "$key" "${!key}")"
		fi
	done
} >>"$LOG_FILE"

set +e
"$@"
status=$?
set -e

{
	echo "exit_code=$status"
	echo
} >>"$LOG_FILE"

exit "$status"
