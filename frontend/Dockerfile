# ---------------------------------------------------------------------------
# Auto Ontology Next.js frontend image
# ---------------------------------------------------------------------------
# Build (from /frontend):  docker build -t auto-ontology-frontend:latest .
# Run:                     docker run --rm -p 3000:3000 \
#                              -e PYTHON_API_URL=http://auto-ontology:3001 \
#                              auto-ontology-frontend:latest
#
# Uses Next.js standalone output (NEXT_OUTPUT=standalone) for a small image.
# The runtime image only contains the standalone server, static assets and
# public files — node_modules from the build stage are intentionally excluded.
# ---------------------------------------------------------------------------

ARG NODE_MAJOR=22
ARG PNPM_VERSION=10.10.0
ARG BASE_IMG=nvcr.io/nvidia/base/ubuntu
ARG BASE_IMG_TAG=jammy-20250619

############################
# Stage 0: base-node
############################
# Install Node.js on top of the NVIDIA Ubuntu base; shared by every
# downstream stage so the install only happens once.
FROM $BASE_IMG:$BASE_IMG_TAG AS base-node

ARG NODE_MAJOR
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
 && curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/*

############################
# Stage 1: deps
############################
FROM base-node AS deps

ARG PNPM_VERSION
ENV PNPM_HOME=/pnpm \
    PATH=/pnpm:$PATH \
    CI=1

RUN npm install -g --silent corepack@latest \
 && corepack enable \
 && corepack prepare pnpm@${PNPM_VERSION} --activate

WORKDIR /app

# Install only what's needed to resolve the dependency graph deterministically.
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml .npmrc ./
COPY prisma ./prisma

RUN --mount=type=cache,id=pnpm-store,target=/pnpm/store \
    pnpm install --frozen-lockfile

############################
# Stage 2: builder
############################
FROM base-node AS builder

# Backend URL baked into the Next.js rewrite manifest at build time
# (Next.js evaluates rewrites() during `next build`, not at server startup).
# Defaults to the in-cluster Service name used by the helm chart.
ARG PYTHON_API_URL=http://auto-ontology:3001

ARG PNPM_VERSION
ENV PNPM_HOME=/pnpm \
    PATH=/pnpm:$PATH \
    CI=1 \
    NEXT_TELEMETRY_DISABLED=1 \
    NEXT_OUTPUT=standalone \
    PYTHON_API_URL=${PYTHON_API_URL}

RUN npm install -g --silent corepack@latest \
 && corepack enable \
 && corepack prepare pnpm@${PNPM_VERSION} --activate

WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

RUN pnpm build

############################
# Stage 3: runner
############################
FROM base-node AS runner

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

RUN apt-get update \
 && apt-get install -y --no-install-recommends tini ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 1001 nodejs \
 && useradd  --system --uid 1001 --gid nodejs --home /app --shell /usr/sbin/nologin nextjs

WORKDIR /app

# Standalone output places the project under <traceRoot-relative path>/.
# In this build the project is copied to /app, and outputFileTracingRoot
# is the parent dir, so the bundle structure ends up rooted at /app/app/.
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone/ ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./app/.next/static
COPY --from=builder --chown=nextjs:nodejs /app/public ./app/public

# Prisma CLI + schema + config, used by the Helm `frontend-migrate` Job
# to run `prisma db push` against Postgres. Lives in /app/migrate so it
# doesn't perturb the Next.js standalone node_modules at /app/app.
COPY --from=builder --chown=nextjs:nodejs /app/prisma /app/migrate/prisma
COPY --from=builder --chown=nextjs:nodejs /app/prisma.config.ts /app/migrate/prisma.config.ts
RUN cd /app/migrate \
 && echo '{"name":"auto-ontology-migrate","version":"0.0.1","private":true,"dependencies":{"prisma":"7.7.0","dotenv":"17.4.2"}}' > package.json \
 && npm install --omit=dev --no-audit --no-fund --no-package-lock \
 && chown -R nextjs:nodejs /app/migrate

USER nextjs

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD node -e "require('http').get('http://127.0.0.1:'+(process.env.PORT||3000)+'/',(r)=>process.exit(r.statusCode<500?0:1)).on('error',()=>process.exit(1))"

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["node", "app/server.js"]
