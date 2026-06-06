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

{{/* Image pull secrets shared by all GSF workload pods. */}}
{{- define "gsf.imagePullSecrets" -}}
{{- with .Values.imagePullSecrets }}
imagePullSecrets:
{{- toYaml . | nindent 0 }}
{{- end }}
{{- end -}}

{{/*
Init containers that block until Postgres and Neo4j are reachable.
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
- name: wait-for-neo4j
  image: busybox:1.36
  imagePullPolicy: {{ .Values.imagePullPolicy }}
  command:
    - sh
    - -c
    - |
      until nc -z neo4j {{ .Values.neo4j.service.boltPort }}; do
        echo "waiting for neo4j..."
        sleep 2
      done
  {{- include "gsf.initResources" . | nindent 2 }}
{{- end -}}

{{/*
Tiny resource bounds for the busybox wait-* init containers. Without an
explicit limit, Astra's namespace LimitRange injects a default limits.cpu=10,
which (because pod quota counts max(initContainers, sum(containers))) makes
every pod claim 10 CPU and blows the ResourceQuota after a single pod.
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