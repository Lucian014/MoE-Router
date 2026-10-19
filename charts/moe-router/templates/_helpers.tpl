{{/* ---------------------------------------------------------------- names and labels */}}

{{/* Prefix of every resource: the release name (e.g. "moe" -> moe-postgres, moe-gateway). */}}
{{- define "moe.fullname" -}}
{{- .Release.Name | trunc 40 | trimSuffix "-" -}}
{{- end -}}

{{- define "moe.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/* Selector labels are immutable: only name + instance + component.
     Usage: include "moe.selectorLabels" (dict "root" . "component" "postgres") */}}
{{- define "moe.selectorLabels" -}}
app.kubernetes.io/name: {{ .root.Chart.Name }}
app.kubernetes.io/instance: {{ .root.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}

{{/* ---------------------------------------------------------------- images */}}

{{/* Usage: include "moe.image" (dict "root" . "image" .Values.gateway.image)
     A repository containing "/" is used as is (e.g. docker.io/library/postgres);
     otherwise it is prefixed with global.imageRegistry.
     Tag: image.tag, else global.imageTag, else the chart's appVersion. */}}
{{- define "moe.image" -}}
{{- $tag := .image.tag | default .root.Values.global.imageTag | default .root.Chart.AppVersion -}}
{{- if contains "/" .image.repository -}}
{{ .image.repository }}:{{ $tag }}
{{- else -}}
{{ .root.Values.global.imageRegistry }}/{{ .image.repository }}:{{ $tag }}
{{- end -}}
{{- end -}}

{{/* Usage: include "moe.pullPolicy" (dict "root" . "image" .Values.gateway.image) */}}
{{- define "moe.pullPolicy" -}}
{{- .image.pullPolicy | default .root.Values.global.imagePullPolicy -}}
{{- end -}}

{{/* Env for the platform's services: the shared ConfigMap + the RabbitMQ credentials,
     plus the Postgres credentials with (dict "root" . "postgres" true). */}}
{{- define "moe.serviceEnvFrom" -}}
- configMapRef:
    name: {{ include "moe.fullname" .root }}-config
- secretRef:
    name: {{ include "moe.rabbitmq.secretName" .root }}
{{- if .postgres }}
- secretRef:
    name: {{ include "moe.postgres.secretName" .root }}
{{- end }}
{{- end -}}

{{- define "moe.imagePullSecrets" -}}
{{- with .Values.global.imagePullSecrets }}
imagePullSecrets:
{{- range . }}
  - name: {{ . }}
{{- end }}
{{- end }}
{{- end -}}

{{/* Usage: include "moe.storageClass" (dict "root" . "persistence" .Values.postgresql.persistence) */}}
{{- define "moe.storageClass" -}}
{{- $sc := .persistence.storageClass | default .root.Values.global.storageClass -}}
{{- if $sc }}
storageClassName: {{ $sc | quote }}
{{- end }}
{{- end -}}

{{/* ---------------------------------------------------------------- Postgres */}}

{{- define "moe.postgres.host" -}}
{{- if .Values.postgresql.enabled -}}
{{ include "moe.fullname" . }}-postgres
{{- else -}}
{{ required "postgresql.external.host is required when postgresql.enabled=false" .Values.postgresql.external.host }}
{{- end -}}
{{- end -}}

{{- define "moe.postgres.port" -}}
{{- if .Values.postgresql.enabled -}}5432{{- else -}}{{ .Values.postgresql.external.port }}{{- end -}}
{{- end -}}

{{/* The Secret with POSTGRES_DB / POSTGRES_USER / POSTGRES_PASSWORD (generated or existing). */}}
{{- define "moe.postgres.secretName" -}}
{{- .Values.postgresql.auth.existingSecret | default (printf "%s-postgres" (include "moe.fullname" .)) -}}
{{- end -}}

{{/* ---------------------------------------------------------------- RabbitMQ */}}

{{- define "moe.rabbitmq.host" -}}
{{- if .Values.rabbitmq.enabled -}}
{{ include "moe.fullname" . }}-rabbitmq
{{- else -}}
{{ required "rabbitmq.external.host is required when rabbitmq.enabled=false" .Values.rabbitmq.external.host }}
{{- end -}}
{{- end -}}

{{- define "moe.rabbitmq.port" -}}
{{- if .Values.rabbitmq.enabled -}}5672{{- else -}}{{ .Values.rabbitmq.external.port }}{{- end -}}
{{- end -}}

{{/* The Secret with RABBITMQ_USER / RABBITMQ_PASS (generated or existing). */}}
{{- define "moe.rabbitmq.secretName" -}}
{{- .Values.rabbitmq.auth.existingSecret | default (printf "%s-rabbitmq" (include "moe.fullname" .)) -}}
{{- end -}}

{{/* ---------------------------------------------------------------- models */}}

{{/* A model_id as a Kubernetes name. MUST match shared/utils.py sanitize_k8s_name(), which
     also names the worker's queue (worker.<name>): 'qwen2.5-coder:1.5b' -> 'qwen2-5-coder-1-5b'.
     Usage: include "moe.k8sName" "qwen2.5-coder:1.5b" */}}
{{- define "moe.k8sName" -}}
{{- . | trim | lower | replace ":" "-" | replace "." "-" | replace "/" "-" -}}
{{- end -}}

{{- define "moe.ollama.url" -}}
http://{{ include "moe.fullname" . }}-ollama:11434
{{- end -}}

{{/* Where a catalog model's worker sends its requests.
     Usage: include "moe.inferenceHost" (dict "root" $ "model" .) */}}
{{- define "moe.inferenceHost" -}}
{{- if .model.inference_host -}}
{{ .model.inference_host }}
{{- else if eq .model.engine "ollama" -}}
{{- if not .root.Values.ollama.enabled -}}
{{- fail (printf "catalog model %s uses engine ollama, but ollama.enabled=false: set its inference_host" .model.model_id) -}}
{{- end -}}
{{ include "moe.ollama.url" .root }}
{{- else -}}
{{- fail (printf "catalog model %s (engine %s) needs an inference_host" .model.model_id .model.engine) -}}
{{- end -}}
{{- end -}}

{{/* ---------------------------------------------------------------- secrets */}}

{{/* A password: the explicit value if set; otherwise the one already stored in the Secret
     (helm lookup, so a generated password survives upgrades); otherwise a new random one.
     Changing it here does NOT change it inside a running Postgres/RabbitMQ: they only read it
     when their data volume is first initialised.
     Usage: include "moe.password" (dict "root" . "secret" "<name>" "key" "<key>" "value" .Values...password)
     Note: `helm template` / Argo CD can't look up live Secrets, so they generate a new value on every
     render; use auth.existingSecret there. */}}
{{- define "moe.password" -}}
{{- $existing := lookup "v1" "Secret" .root.Release.Namespace .secret -}}
{{- if .value -}}
{{ .value | b64enc }}
{{- else if and $existing $existing.data (hasKey $existing.data .key) -}}
{{ index $existing.data .key }}
{{- else -}}
{{ randAlphaNum 32 | b64enc }}
{{- end -}}
{{- end -}}
