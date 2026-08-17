#!/usr/bin/env bash
# Local minikube deploy for the Postgres branch.
#
# Adapted from the copy in the main checkout. Two things had to change, both
# consequences of the migration:
#
#   * `minikube image load neo4j:latest` and `--set neo4jPassword=...` are gone.
#     The chart has no Neo4j template and no neo4j value; Helm silently ignores
#     an unknown --set, so leaving it in would have looked like it was doing
#     something.
#   * REPO points at this worktree, so the images are built from the branch.
#
# Run in stages -- the whole thing takes well over ten minutes, and each stage
# is separately re-runnable:
#
#     ./helm.sh build     # docker build both images
#     ./helm.sh cluster   # recreate minikube, load images
#     ./helm.sh install   # helm install --wait
#     ./helm.sh forward    # port-forward (blocks)
#     ./helm.sh all       # everything except the forward
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_REPO=/Users/yshkolar/Dev/GSF
STAGE="${1:-all}"

# Secrets come from the main checkout's script so they are never duplicated here.
_key() { grep -oE "\-\-set $1=[^ ]+" "$MAIN_REPO/helm.sh" | head -1 | cut -d= -f2- | tr -d "'\" \\\\"; }
NVIDIA_API_KEY="$(_key nvidiaApiKey)"
KUMO_API_KEY="$(_key kumoRfmApiKey)"

build() {
  echo "== build =="
  docker build -t gsf:local "$REPO"
  docker build -t gsf-frontend:local "$REPO/frontend"
}

cluster() {
  echo "== cluster =="
  minikube delete
  minikube start --driver=docker --cpus=4 --memory=6144
  minikube image load gsf:local
  minikube image load gsf-frontend:local
  # Preload third-party images: a fresh cluster pulling these anonymously hits
  # Docker Hub's rate limit, which leaves postgres in ImagePullBackOff and hangs
  # the migrate jobs. No neo4j here any more.
  minikube image load pgvector/pgvector:pg17
  minikube image load dpage/pgadmin4:latest
  minikube image load busybox:1.36
}

install() {
  echo "== install =="
  helm install gsf "$REPO/helm/gsf" \
    --wait --timeout 10m \
    --set backend.image.repository=gsf \
    --set backend.image.tag=local \
    --set ingestion.image.repository=gsf \
    --set ingestion.image.tag=local \
    --set frontend.image.repository=gsf-frontend \
    --set frontend.image.tag=local \
    --set postgresPassword=gsfpassword \
    --set adminEmail=admin@example.com \
    --set adminPassword=admin \
    --set appUrl=http://localhost:3100 \
    --set nvidiaApiKey="$NVIDIA_API_KEY" \
    --set kumoRfmApiKey="$KUMO_API_KEY" \
    --set kumoRfmApiUrl=https://rfm-nim.dev.kumoai.cloud
}

forward() {
  echo "== forward =="
  kubectl rollout status deploy/frontend --timeout=300s
  # 0.0.0.0 so other containers can reach GSF at host.docker.internal:3100.
  kubectl port-forward svc/frontend 3100:3000 --address 0.0.0.0
}

case "$STAGE" in
  build)   build ;;
  cluster) cluster ;;
  install) install ;;
  forward) forward ;;
  all)     build; cluster; install
           echo "== done: run './helm.sh forward' to reach the UI on :3100 ==" ;;
  *) echo "usage: $0 {build|cluster|install|forward|all}"; exit 2 ;;
esac
