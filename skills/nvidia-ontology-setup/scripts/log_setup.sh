#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Append-only setup log. Wrap the real command; do not invent flags here.
#
#   log_setup.sh [--model NAME] -- command [args...]
#   NVIDIA_ONTOLOGY_SETUP_MODEL=... NVIDIA_ONTOLOGY_SETUP_LOG=/tmp/ontology-setup.log log_setup.sh -- uv sync

set -euo pipefail

usage() {
	cat <<'EOF'
Usage:
  log_setup.sh [--model NAME] -- command [args...]

Appends timestamp, optional model, cwd, the redacted command, redacted env,
and exit code to .nvidia-ontology-setup.log at the git repo root (or $NVIDIA_ONTOLOGY_SETUP_LOG).
EOF
}

MODEL="${NVIDIA_ONTOLOGY_SETUP_MODEL:-}"
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

if [[ -n "${NVIDIA_ONTOLOGY_SETUP_LOG:-}" ]]; then
	LOG_FILE="$NVIDIA_ONTOLOGY_SETUP_LOG"
elif ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"; then
	LOG_FILE="$ROOT/.nvidia-ontology-setup.log"
else
	LOG_FILE="$PWD/.nvidia-ontology-setup.log"
fi

# Match env names and Helm camelCase (defaultModelsApiKey, connectionStrings).
is_sensitive_key() {
	local upper
	upper="$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]' | tr -cd 'A-Z')"
	case "$upper" in
	*PASSWORD* | *SECRET* | *TOKEN* | *APIKEY* | *CONNECTIONSTRING* | *ROLEID*)
		return 0
		;;
	esac
	return 1
}

redact_value() {
	local key="$1" value="$2"
	if is_sensitive_key "$key"; then
		if [[ -n "$value" ]]; then
			printf '%s' '***'
		fi
	else
		printf '%s' "$value"
	fi
}

# Helm --set a=1,postgresPassword=secret  (and KEY=VALUE / --flag=VALUE).
redact_csv_assignments() {
	local remaining="$1" out="" piece key
	while [[ -n "$remaining" ]]; do
		if [[ "$remaining" == *,* ]]; then
			piece="${remaining%%,*}"
			remaining="${remaining#*,}"
		else
			piece="$remaining"
			remaining=""
		fi
		if [[ "$piece" == *=* ]]; then
			key="${piece%%=*}"
			if is_sensitive_key "$key"; then
				piece="${key}=***"
			fi
		fi
		if [[ -n "$out" ]]; then
			out="${out},${piece}"
		else
			out="$piece"
		fi
	done
	printf '%s' "$out"
}

redact_assignment() {
	local input="$1"
	if [[ "$input" == --*=* ]]; then
		local flag="${input%%=*}"
		local rest="${input#*=}"
		if is_sensitive_key "${flag#--}"; then
			printf '%s=***' "$flag"
			return
		fi
		printf '%s=%s' "$flag" "$(redact_csv_assignments "$rest")"
		return
	fi
	redact_csv_assignments "$input"
}

redact_logged_command() {
	local -a args=("$@") out=()
	local i n arg skip_next=0
	n=${#args[@]}
	i=0
	while [[ $i -lt $n ]]; do
		if [[ $skip_next -eq 1 ]]; then
			skip_next=0
			i=$((i + 1))
			continue
		fi
		arg="${args[$i]}"
		case "$arg" in
		--set | --set-string | --set-json | --set-file)
			out+=("$arg")
			if [[ $((i + 1)) -lt $n ]]; then
				out+=("$(redact_csv_assignments "${args[$((i + 1))]}")")
				skip_next=1
			fi
			;;
		--password | --token | --secret | --docker-password)
			out+=("$arg")
			if [[ $((i + 1)) -lt $n ]]; then
				out+=("***")
				skip_next=1
			fi
			;;
		--*=*)
			out+=("$(redact_assignment "$arg")")
			;;
		*=*)
			out+=("$(redact_csv_assignments "$arg")")
			;;
		*)
			out+=("$arg")
			;;
		esac
		i=$((i + 1))
	done
	if [[ ${#out[@]} -gt 0 ]]; then
		printf '%q ' "${out[@]}"
	fi
}

{
	echo "-----"
	echo "timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
	echo "model=${MODEL:-}"
	echo "cwd=$PWD"
	printf 'command='
	redact_logged_command "$@"
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
		AUTO_ONTOLOGY_API_URL \
		AUTO_ONTOLOGY_MCP_HOST \
		AUTO_ONTOLOGY_MCP_PORT \
		AUTO_ONTOLOGY_MCP_PUBLIC_URL \
		VAULT_ADDR \
		VAULT_NAMESPACE \
		VAULT_ROLE_ID \
		VAULT_SECRET_ID \
		AUTO_ONTOLOGY_ADMIN_EMAIL \
		AUTO_ONTOLOGY_ADMIN_PASSWORD; do
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
