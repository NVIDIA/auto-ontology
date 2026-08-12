{{/* Common helpers */}}

{{- define "gsf.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "gsf.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s" .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "gsf.labels" -}}
app.kubernetes.io/name: {{ include "gsf.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{- end -}}

{{- define "gsf.componentLabels" -}}
{{ include "gsf.labels" . }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}

{{- define "gsf.componentSelectorLabels" -}}
app.kubernetes.io/name: {{ include "gsf.name" .root }}
app.kubernetes.io/instance: {{ .root.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}

{{- define "gsf.secretName" -}}
{{ include "gsf.fullname" . }}-secrets
{{- end -}}

{{/*
Image reference for a component image map. Prefers an immutable digest when
`.digest` is set (e.g. GitOps digest-pinning), otherwise falls back to `:tag`.
Usage: {{ include "gsf.image" .Values.backend.image }}
*/}}
{{- define "gsf.image" -}}
{{- if .digest -}}
{{ .repository }}@{{ .digest }}
{{- else -}}
{{ .repository }}:{{ .tag }}
{{- end -}}
{{- end -}}

{{/*
Image pull secrets shared by all GSF workload pods. Empty by default (local /
minikube pull public images); set `imagePullSecrets` to pull private images
(e.g. a private registry credential such as nvcr.io).
*/}}
{{- define "gsf.imagePullSecrets" -}}
{{- with .Values.imagePullSecrets }}
imagePullSecrets:
{{- toYaml . | nindent 0 }}
{{- end }}
{{- end -}}

{{/*
A Service routes only to Ready endpoints, so `nc -z` succeeds only after
each datastore's readiness probe has passed.
*/}}
{{- define "gsf.waitForDeps" -}}
- name: wait-for-postgres
  image: busybox:1.36
  imagePullPolicy: {{ .Values.imagePullPolicy }}
  command:
    - sh
    - -c
    - |
      until nc -z postgres {{ .Values.postgres.service.port }}; do
        echo "waiting for postgres..."
        sleep 2
      done
  {{- include "gsf.initResources" . | nindent 2 }}
{{- end -}}

{{/*
Init container that blocks until the frontend's Prisma schema exists in Postgres,
gating on the `user` table that seed-admin (auth/seed-admin.ts, run from
instrumentation.ts on boot) writes to. Without this the frontend can start and
seed the bootstrap admin before the migrate Job has created the tables, which
fails with TableDoesNotExist and leaves no admin account (login then fails).
Polls with psql from the same image Postgres uses, so the client is guaranteed
present and already pulled. Runs after gsf.waitForDeps, so Postgres is reachable.
*/}}
{{- define "gsf.waitForSchema" -}}
- name: wait-for-schema
  image: "{{ .Values.postgres.image.repository }}:{{ .Values.postgres.image.tag }}"
  imagePullPolicy: {{ .Values.imagePullPolicy }}
  envFrom:
    - secretRef:
        name: {{ include "gsf.secretName" . }}
  command:
    - sh
    - -c
    - |
      until PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" -d "$POSTGRES_DATABASE" -tAc \
        "SELECT 1 FROM information_schema.tables WHERE table_schema='public' AND table_name='user'" \
        2>/dev/null | grep -q 1; do
        echo "waiting for prisma schema (public.user)..."
        sleep 2
      done
  {{- include "gsf.initResources" . | nindent 2 }}
{{- end -}}

{{/*
Init container that blocks until the backend's Alembic migrations have been
applied *to the revision this release ships*. Without it the backend goes Ready
and serves 500s on every catalog call while backend-migrate is still running --
the Job is a normal release resource, so nothing else orders the two.

Gating on a table's existence is not enough: it is satisfied from the first
install onward, so on an upgrade that adds a column the new Pods roll out while
the migration is still running and every query touching that column fails with
UndefinedColumn. This compares `alembic_version.version_num` against the head
revision baked into the image at build time, so it blocks on upgrades too.

Mirrors gsf.waitForSchema, which does the same for Prisma's `public` schema.
Runs after gsf.waitForDeps, so Postgres is reachable.
*/}}
{{- define "gsf.waitForCatalogSchema" -}}
- name: wait-for-catalog-schema
  image: "{{ .Values.postgres.image.repository }}:{{ .Values.postgres.image.tag }}"
  imagePullPolicy: {{ .Values.imagePullPolicy }}
  envFrom:
    - secretRef:
        name: {{ include "gsf.secretName" . }}
  command:
    - sh
    - -c
    - |
      until PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" -d "$POSTGRES_DATABASE" -tAc \
        "SELECT 1 FROM gsf.alembic_version WHERE version_num = '{{ .Values.backend.alembicRevision }}'" \
        2>/dev/null | grep -q 1; do
        echo "waiting for alembic revision {{ .Values.backend.alembicRevision }}..."
        sleep 2
      done
  {{- include "gsf.initResources" . | nindent 2 }}
{{- end -}}

{{/*
Tiny resource bounds for the busybox wait-* init containers. Without an explicit
limit, some namespace LimitRanges inject a large default limits.cpu, which
(because pod quota counts max(initContainers, sum(containers))) can make every
pod claim that much CPU and exhaust a ResourceQuota.
*/}}
{{- define "gsf.initResources" -}}
resources:
  requests:
    cpu: 10m
    memory: 16Mi
  limits:
    cpu: 100m
    memory: 64Mi
{{- end -}}
