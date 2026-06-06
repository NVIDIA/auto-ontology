#!/usr/bin/env bash
#
# Publish the canonical GSF Helm chart (this directory) to the Astra GitOps
# deploy repo that ArgoCD watches.
#
# ── Canonical source of truth ────────────────────────────────────────────────
# THIS chart (helm/gsf/ in the GitHub repo) is the single source of truth.
# The GitLab `gsf-deploy` repo is a PUBLISHED COPY consumed by ArgoCD — never
# hand-edit its app templates. To ship a chart change:
#
#     1. edit helm/gsf/  (and commit it to GitHub)
#     2. run ./publish-to-astra.sh
#     3. ArgoCD syncs the deploy repo onto Astra
#
# Fusion injects three platform-glue templates into the deploy repo
# (fusion-serviceaccount.yaml, fusion-secretstore.yaml,
# fusion-externalsecret.yaml). They encode env/DL/Vault specifics and are NOT
# part of this portable chart, so this script PRESERVES them and only manages
# the application templates + Chart.yaml + values.yaml.
#
# Usage:
#   ./publish-to-astra.sh [-e ENV] [-r DEPLOY_REPO_SSH_URL] [-k SSH_KEY] [-n]
#     -e ENV    target environment dir under deployment/ (default: stg)
#     -r URL    deploy repo SSH URL (default: the access-gpu-product-all gsf-deploy)
#     -k KEY    SSH private key for gitlab-master (default: ~/.ssh/id_ed25519_nvidia)
#     -n        dry-run: stage + diff only, do not commit/push
#
set -euo pipefail

ENV="stg"
DEPLOY_REPO="ssh://git@gitlab-master.nvidia.com:12051/ape-repo/astra-projects/access-gpu-product-all/gsf-deploy.git"
SSH_KEY="${GSF_DEPLOY_SSH_KEY:-$HOME/.ssh/id_ed25519_nvidia}"
DRY_RUN=0

while getopts "e:r:k:nh" opt; do
  case "$opt" in
    e) ENV="$OPTARG" ;;
    r) DEPLOY_REPO="$OPTARG" ;;
    k) SSH_KEY="$OPTARG" ;;
    n) DRY_RUN=1 ;;
    h) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown option" >&2; exit 2 ;;
  esac
done

CHART_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$CHART_DIR/Chart.yaml" ] || { echo "error: $CHART_DIR is not a Helm chart" >&2; exit 1; }

# Render-check the chart before publishing anything.
if command -v helm >/dev/null 2>&1; then
  echo ">> helm template (lint render) ..."
  helm template gsf "$CHART_DIR" >/dev/null
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export GIT_SSH_COMMAND="ssh -i $SSH_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no"

echo ">> cloning deploy repo ..."
git clone --quiet "$DEPLOY_REPO" "$WORK/deploy"
DEST="$WORK/deploy/deployment/$ENV"
[ -d "$DEST" ] || { echo "error: $DEST not found in deploy repo" >&2; exit 1; }

echo ">> syncing chart -> deployment/$ENV (preserving fusion-*.yaml) ..."
cp "$CHART_DIR/Chart.yaml"  "$DEST/Chart.yaml"
cp "$CHART_DIR/values.yaml" "$DEST/values.yaml"
mkdir -p "$DEST/templates"
# Drop dest app templates (catches renames/deletions) but keep Fusion glue.
find "$DEST/templates" -type f ! -name 'fusion-*.yaml' -delete
cp "$CHART_DIR/templates/"*.yaml "$DEST/templates/" 2>/dev/null || true
cp "$CHART_DIR/templates/"*.tpl  "$DEST/templates/" 2>/dev/null || true

cd "$WORK/deploy"
git add -A "deployment/$ENV"

if git diff --cached --quiet; then
  echo ">> nothing to publish (deploy repo already up to date)."
  exit 0
fi

echo ">> changes to publish:"
git diff --cached --stat

if [ "$DRY_RUN" -eq 1 ]; then
  echo ">> dry-run: not committing or pushing."
  exit 0
fi

git commit -S -m "Publish GSF chart to $ENV from helm/gsf"
git push origin HEAD:main
echo ">> published. ArgoCD will sync deployment/$ENV."
