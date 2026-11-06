{{/*
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}}
{{/* Common helpers */}}

{{- define "auto-ontology.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "auto-ontology.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s" .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "auto-ontology.labels" -}}
app.kubernetes.io/name: {{ include "auto-ontology.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{- end -}}

{{- define "auto-ontology.componentLabels" -}}
{{ include "auto-ontology.labels" . }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}

{{- define "auto-ontology.componentSelectorLabels" -}}
app.kubernetes.io/name: {{ include "auto-ontology.name" .root }}
app.kubernetes.io/instance: {{ .root.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}

{{- define "auto-ontology.secretName" -}}
{{ include "auto-ontology.fullname" . }}-secrets
{{- end -}}

{{/*
Image reference for a component image map. Prefers an immutable digest when
`.digest` is set (e.g. GitOps digest-pinning), otherwise falls back to `:tag`.
Usage: {{ include "auto-ontology.image" .Values.backend.image }}
*/}}
{{- define "auto-ontology.image" -}}
{{- if .digest -}}
{{ .repository }}@{{ .digest }}
{{- else -}}
{{ .repository }}:{{ .tag }}
{{- end -}}
{{- end -}}

{{/*
Image pull secrets shared by all Auto Ontology workload pods. Empty by default (local /
minikube pull public images); set `imagePullSecrets` to pull private images
(e.g. a private registry credential such as nvcr.io).
*/}}
{{- define "auto-ontology.imagePullSecrets" -}}
{{- with .Values.imagePullSecrets }}
imagePullSecrets:
{{- toYaml . | nindent 0 }}
{{- end }}
{{- end -}}

{{/*
A Service routes only to Ready endpoints, so `nc -z` succeeds only after
each datastore's readiness probe has passed.
*/}}
{{- define "auto-ontology.waitForDeps" -}}
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
  {{- include "auto-ontology.initResources" . | nindent 2 }}
{{- if .Values.elasticsearch.bundled.enabled }}
- name: wait-for-elasticsearch
  image: busybox:1.36
  imagePullPolicy: {{ .Values.imagePullPolicy }}
  command:
    - sh
    - -c
    - |
      until nc -z elasticsearch {{ .Values.elasticsearch.bundled.service.port }}; do
        echo "waiting for elasticsearch..."
        sleep 2
      done
  {{- include "auto-ontology.initResources" . | nindent 2 }}
{{- end }}
{{- end -}}

{{/*
The vector store backend, after checking the values make sense together. Fails
the render rather than deploying a backend that errors on its first write.
*/}}
{{- define "auto-ontology.vectorBackend" -}}
{{- $backend := .Values.vectorStore.backend | default "postgres" | lower -}}
{{- $es := .Values.elasticsearch -}}
{{- if not (has $backend (list "postgres" "elasticsearch")) -}}
{{- fail (printf "vectorStore.backend must be postgres or elasticsearch, not %q." $backend) -}}
{{- end -}}
{{- if not (has (.Values.vectorStore.searchMode | default "semantic" | lower) (list "semantic" "hybrid")) -}}
{{- fail (printf "vectorStore.searchMode must be semantic or hybrid, not %q." .Values.vectorStore.searchMode) -}}
{{- end -}}
{{- if eq $backend "postgres" -}}
{{- if $es.bundled.enabled -}}
{{- fail "elasticsearch.bundled.enabled needs vectorStore.backend=elasticsearch, or nothing would use it." -}}
{{- end -}}
{{- if eq (.Values.vectorStore.searchMode | default "semantic" | lower) "hybrid" -}}
{{- fail "vectorStore.searchMode=hybrid needs vectorStore.backend=elasticsearch." -}}
{{- end -}}
{{- else -}}
{{- if and $es.bundled.enabled $es.url -}}
{{- fail "Set either elasticsearch.url (an existing cluster) or elasticsearch.bundled.enabled, not both." -}}
{{- end -}}
{{- if not (or $es.bundled.enabled $es.url) -}}
{{- fail "vectorStore.backend=elasticsearch needs elasticsearch.url, or elasticsearch.bundled.enabled=true." -}}
{{- end -}}
{{- end -}}
{{- $backend -}}
{{- end -}}

{{/*
ES_URL: the bundled cluster's Service, or the configured one.
*/}}
{{- define "auto-ontology.elasticsearchUrl" -}}
{{- if .Values.elasticsearch.bundled.enabled -}}
{{ printf "http://elasticsearch:%d" (int .Values.elasticsearch.bundled.service.port) }}
{{- else -}}
{{ .Values.elasticsearch.url }}
{{- end -}}
{{- end -}}

{{/*
The Elasticsearch CA certificate, mounted into the backend and ingestion pods
when elasticsearch.caSecret.name is set. Three pieces, one per place in the pod
spec: the env var, the container mount and the pod volume.
*/}}
{{- define "auto-ontology.esCaEnv" -}}
{{- if .Values.elasticsearch.caSecret.name -}}
- name: ES_CA_CERTS
  value: /etc/auto-ontology/elasticsearch/ca.crt
{{- end }}
{{- end -}}

{{- define "auto-ontology.esCaVolumeMount" -}}
{{- if .Values.elasticsearch.caSecret.name -}}
volumeMounts:
  - name: elasticsearch-ca
    mountPath: /etc/auto-ontology/elasticsearch
    readOnly: true
{{- end }}
{{- end -}}

{{- define "auto-ontology.esCaVolume" -}}
{{- if .Values.elasticsearch.caSecret.name -}}
volumes:
  - name: elasticsearch-ca
    secret:
      secretName: {{ .Values.elasticsearch.caSecret.name }}
      items:
        - key: {{ .Values.elasticsearch.caSecret.key | default "ca.crt" }}
          path: ca.crt
{{- end }}
{{- end -}}

{{/*
Init container that blocks until the frontend's Prisma schema exists in Postgres,
gating on the `frontend.user` table that seed-admin (auth/seed-admin.ts, run from
instrumentation.ts on boot) writes to. Without this the frontend can start and
seed the bootstrap admin before the migrate Job has created the tables, which
fails with TableDoesNotExist and leaves no admin account (login then fails).
Polls with psql from the same image Postgres uses, so the client is guaranteed
present and already pulled. Runs after auto-ontology.waitForDeps, so Postgres is reachable.
*/}}
{{- define "auto-ontology.waitForSchema" -}}
- name: wait-for-schema
  image: "{{ .Values.postgres.image.repository }}:{{ .Values.postgres.image.tag }}"
  imagePullPolicy: {{ .Values.imagePullPolicy }}
  envFrom:
    - secretRef:
        name: {{ include "auto-ontology.secretName" . }}
  command:
    - sh
    - -c
    - |
      until PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" -d "$POSTGRES_DATABASE" -tAc \
        "SELECT 1 FROM information_schema.tables WHERE table_schema='frontend' AND table_name='user'" \
        2>/dev/null | grep -q 1; do
        echo "waiting for prisma schema (frontend.user)..."
        sleep 2
      done
  {{- include "auto-ontology.initResources" . | nindent 2 }}
{{- end -}}

{{/*
Init container that blocks until the backend's Alembic migrations have been
applied *to the revision this release ships*. Without it the backend goes Ready
and serves 500s on every catalog call while auto-ontology-migrate is still running --
the Job is a normal release resource, so nothing else orders the two.

Gating on a table's existence is not enough: it is satisfied from the first
install onward, so on an upgrade that adds a column the new Pods roll out while
the migration is still running and every query touching that column fails with
UndefinedColumn. This compares `alembic_version.version_num` against the head
revision baked into the image at build time, so it blocks on upgrades too.

Mirrors auto-ontology.waitForSchema, which does the same for Prisma's `frontend` schema.
Runs after auto-ontology.waitForDeps, so Postgres is reachable.
*/}}
{{- define "auto-ontology.waitForCatalogSchema" -}}
- name: wait-for-catalog-schema
  image: "{{ .Values.postgres.image.repository }}:{{ .Values.postgres.image.tag }}"
  imagePullPolicy: {{ .Values.imagePullPolicy }}
  envFrom:
    - secretRef:
        name: {{ include "auto-ontology.secretName" . }}
  command:
    - sh
    - -c
    - |
      until PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" -d "$POSTGRES_DATABASE" -tAc \
        "SELECT 1 FROM public.alembic_version WHERE version_num = '{{ .Values.backend.alembicRevision }}'" \
        2>/dev/null | grep -q 1; do
        echo "waiting for alembic revision {{ .Values.backend.alembicRevision }}..."
        sleep 2
      done
  {{- include "auto-ontology.initResources" . | nindent 2 }}
{{- end -}}

{{/*
Tiny resource bounds for the busybox wait-* init containers. Without an explicit
limit, some namespace LimitRanges inject a large default limits.cpu, which
(because pod quota counts max(initContainers, sum(containers))) can make every
pod claim that much CPU and exhaust a ResourceQuota.
*/}}
{{- define "auto-ontology.initResources" -}}
resources:
  requests:
    cpu: 10m
    memory: 16Mi
  limits:
    cpu: 100m
    memory: 64Mi
{{- end -}}
