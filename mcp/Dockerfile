# ---------------------------------------------------------------------------
# GSF MCP server image
# ---------------------------------------------------------------------------
# Build (from /mcp):  docker build -t gsf-mcp:latest .
# Run:                docker run --rm -p 3003:3003 \
#                         -e GSF_API_URL=https://gsf.example.com \
#                         -e GSF_MCP_PUBLIC_URL=https://mcp.example.com \
#                         gsf-mcp:latest
#
# Separate from the FastAPI image: gsf-mcp talks to GSF over HTTP and must
# not pull the backend's database drivers or model stack.
# ---------------------------------------------------------------------------

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.11.29
ARG BASE_IMG=nvcr.io/nvidia/base/ubuntu
ARG BASE_IMG_TAG=jammy-20250619

############################
# Stage 1: builder
############################
FROM $BASE_IMG:$BASE_IMG_TAG AS builder

ARG PYTHON_VERSION
ARG UV_VERSION

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/python

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
 && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" \
        -o /tmp/install-uv.sh \
 && env UV_UNMANAGED_INSTALL=/usr/local/bin sh /tmp/install-uv.sh \
 && rm /tmp/install-uv.sh

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY gsf_mcp/ ./gsf_mcp/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install "${PYTHON_VERSION}" \
 && uv sync --frozen --no-dev

############################
# Stage 2: runtime
############################
FROM $BASE_IMG:$BASE_IMG_TAG AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    GSF_MCP_HOST=0.0.0.0 \
    GSF_MCP_PORT=3003

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        tini \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 1000 gsf \
 && useradd  --system --uid 1000 --gid gsf --home /app --shell /usr/sbin/nologin gsf

WORKDIR /app

COPY --from=builder /opt/python /opt/python
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=gsf:gsf /app /app

USER gsf

EXPOSE 3003

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import socket,os,sys; p=int(os.environ.get('GSF_MCP_PORT','3003')); s=socket.create_connection(('127.0.0.1',p),3); s.close()"

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["gsf-mcp"]
