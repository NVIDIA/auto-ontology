# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# Auto Ontology FastAPI backend image
# ---------------------------------------------------------------------------
# Build (default — uses the committed nemo-retriever stub; image starts and
# serves /api/health but real chat/datasource calls raise NotImplementedError):
#
#   docker build -t auto-ontology:latest .
#
# Build with the real NeMo-Retriever source — point a BuildKit named context
# at any local clone of the upstream repo (no copy into this repo):
#
#   docker build \
#       --build-context nemo=/path/to/NeMo-Retriever \
#       -t auto-ontology:latest .
#
# Run:
#
#   docker run --rm -p 3001:3001 --env-file .env auto-ontology:latest
#
# ---------------------------------------------------------------------------

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.11.29
ARG BASE_IMG=nvcr.io/nvidia/base/ubuntu
ARG BASE_IMG_TAG=jammy-20250619

############################
# Stage 1: builder
############################
FROM $BASE_IMG:$BASE_IMG_TAG AS builder

ARG UV_VERSION

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/python

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# Install uv without introducing a non-NVIDIA build stage.
RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" \
        -o /tmp/install-uv.sh \
 && env UV_UNMANAGED_INSTALL=/usr/local/bin sh /tmp/install-uv.sh \
 && rm /tmp/install-uv.sh

WORKDIR /app

# Copy lock + project metadata first to maximise layer caching.
COPY pyproject.toml uv.lock .python-version ./
COPY vendor/ vendor/

# Resolve and install the dependency closure into /opt/venv.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project

############################
# Stage 2: runtime
############################
FROM $BASE_IMG:$BASE_IMG_TAG AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    PORT=3001

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        libpq5 \
        tini \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 1000 auto_ontology \
 && useradd  --system --uid 1000 --gid auto_ontology --home /app --shell /usr/sbin/nologin auto_ontology

WORKDIR /app

COPY --from=builder /opt/python /opt/python
COPY --from=builder /opt/venv /opt/venv
COPY --chown=auto_ontology:auto_ontology auto_ontology/ ./auto_ontology/
COPY --chown=auto_ontology:auto_ontology pyproject.toml ./
# Schema migrations for the `public` schema, run by the `migrate` mode below.
COPY --chown=auto_ontology:auto_ontology alembic/ ./alembic/
COPY --chown=auto_ontology:auto_ontology alembic.ini ./

# Dispatcher entrypoint: selects between the FastAPI server, the ingestion
# service, and the one-shot schema migration based on the first arg.
# Defaults to auto_ontology.server.
COPY --chmod=0755 <<'EOF' /usr/local/bin/entrypoint.sh
#!/bin/sh
set -e
mode="${1:-server}"
case "$mode" in
  server)
    exec python -m auto_ontology.server
    ;;
  ingestion_service)
    exec python -m auto_ontology.ingestion_service
    ;;
  migrate)
    # Owns `public` only. Prisma owns `frontend` and migrates separately;
    # the two are independent and may run in either order.
    exec alembic upgrade head
    ;;
  *)
    echo "Unknown mode: $mode" >&2
    echo "Usage: docker run <image> [server|ingestion_service|migrate]" >&2
    exit 2
    ;;
esac
EOF

# License and attribution files (OSRB): see CONTAINER_THIRD_PARTY_NOTICES.md.
COPY LICENSE THIRD_PARTY_NOTICES.md CONTAINER_THIRD_PARTY_NOTICES.md OPEN_SOURCE_NOTICE /licenses/

USER auto_ontology

EXPOSE 3001

# `/api/health/live`, not `/api/health`. The latter is the *readiness* probe:
# it queries Postgres and returns 503 when the database is unreachable. Docker
# has no notion of readiness -- HEALTHCHECK produces one binary container state
# -- so pointing it there makes a database blip mark this container unhealthy,
# and anything gating on `condition: service_healthy` (the frontend, in
# docker-compose.yml) then refuses to start behind a backend that is running
# perfectly well.
#
# Restarting this process cannot fix an unreachable database, which is the whole
# reason the health router splits the two and the Helm chart points its
# livenessProbe at /api/health/live and only its readinessProbe at /api/health.
# Readiness stays available on /api/health for the things that can act on it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{__import__(\"os\").environ.get(\"PORT\",\"3001\")}/api/health/live', timeout=3).status == 200 else 1)"

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["server"]
