Phase 0: Decisions (half a day, before any code)

  Write these down, for example in docs/deployment.md. Every later step depends on them.

  1. One namespace per release. Drop platform, gateway and processing. The release namespace is set at install time (helm install -n <anything>). This removes the three copies of each secret and every cross-namespace FQDN.
  2. Image names: ghcr.io/lucian014/moe-gateway, moe-router, moe-worker, moe-tools. moe-tools is your current manage-models image, renamed. It runs migrations, the seed and ollama pull.
  3. Versioning: a git tag vX.Y.Z produces images tagged X.Y.Z and a chart with appVersion: X.Y.Z. Every push to main produces images tagged sha-<short>. Nothing deployable uses :latest.
  4. Visibility: either public GHCR plus public HF checkpoints (check the mmBERT/Laya licence), or private GHCR plus imagePullSecrets. Public is much simpler for an open-source platform.
  5. Branch: git switch -c infra/portable-helm. Commit or stash your current changes first (the staged models.json deletion and the two comment edits).

  ---

  Phase 1: Make the services configurable (code changes)

  The goal is that no service knows anything about the cluster it runs in. Everything comes from environment variables with neutral defaults.

  1.1 Neutral defaults
  - src/database/config.py:4: change the default from postgres.platform.svc.cluster.local to postgres.
  - scripts/build_manifests.py:41-42: this file is deleted in Phase 5, so leave it for now.
  - All services: RABBITMQ_HOST defaults to rabbitmq and POSTGRES_HOST to postgres. Add RABBITMQ_PORT and RABBITMQ_VHOST, and pass them to connect_robust.
  - Credentials have no defaults. Today guest/guest is the fallback. Fail fast if they're missing, as the worker already does for MODEL_NAME.

  1.2 Startup order without hardcoded waits

  In Kubernetes, pods start in any order. RabbitMQClient.connect gives up after 10×5 s and crashes. That's acceptable because Kubernetes restarts the pod, but cleaner is exponential backoff with no hard cap (or a cap from env), plus a clear log line. Do the same for the router's first catalog load: if Postgres isn't up yet, or the tables don't exist yet because migrations haven't run, retry instead of exiting.

  1.3 Health endpoints and probes
  - Gateway: keep /health for liveness. Add /ready, which returns 503 until the RabbitMQ RPC channel is up.
  - Router and worker have no HTTP server. The simplest portable option is that each writes /tmp/ready once it's connected and consuming, and the probe is exec: test -f /tmp/ready. The worker should only write it once the engine answers too (GET {INFERENCE_HOST}/api/tags or /v1/models).
  - Router readiness must not require a non-empty catalog. Otherwise helm install --wait deadlocks with the post-install seed job (Phase 4.6).

  1.4 Graceful shutdown

  Handle SIGTERM in router and worker: stop consuming, let the in-flight message finish or nack it, then close the connection. Right now asyncio.Future() never resolves, so pods are killed after terminationGracePeriodSeconds.

  1.5 Migrations and seed must run headless

  dbctl.sh relies on a port-forward from your laptop. Add a container entrypoint to moe-tools that does the following, using env vars only:
  alembic -c src/database/alembic.ini upgrade head
  python scripts/seed_models.py --from-yaml /config/catalog.yaml   # idempotent upsert
  python scripts/manage_models/manage_models.py --provision         # ollama pull

  - Make the seed idempotent: upsert by model_id, and disable models that aren't in the file instead of deleting them. That way it can run on every helm upgrade.
  - Check that alembic.ini / migrations/env.py read SYNC_DATABASE_URL from env. Check also that the moe-tools requirements include alembic and psycopg2-binary.

  1.6 Dockerfiles
  - Pin dependencies. Today everything is >=, so two builds of the same commit can differ. Generate lock files with pip-compile or uv pip compile per service and install from them.
  - Run as non-root: RUN useradd -u 10001 app and USER 10001. For the router, check that HF_HOME under /app stays readable; it's only read at runtime.
  - Add OCI labels. GHCR uses org.opencontainers.image.source to link the package to your repo, which simplifies its permissions:
  LABEL org.opencontainers.image.source="https://github.com/Lucian014/MoE-Router"
    docker/metadata-action adds this automatically in CI, so it's mainly for local builds.
  - .dockerignore: make sure .venv, .git, src/router/checkpoints, scripts/resources and scripts/laya are excluded. The context is ., so a large .venv slows every build.
  - moe-tools: it copies all of src/ and scripts/. That's fine, but add its requirements lock too.

  1.7 A minimum of tests (needed for the CI gate in Phase 2)

  Add tests/test_routing_rule.py covering select_model and validate_catalog. These are pure functions, so no infrastructure is needed. Add ruff config in pyproject.toml.

  Check: run each image locally with podman run -e ... against a podman run rabbitmq / postgres pair, without any Kubernetes. If it starts with only env vars, Phase 1 is done.

  ---

  Phase 2: CI that builds and pushes to GHCR

  2.1 Repo settings
  - In GitHub, under Settings → Actions → General → Workflow permissions, make sure jobs can request packages: write. The workflow below asks for it explicitly.
  - Add the repository secret HF_TOKEN, a read-only HF token.

  2.2 .github/workflows/ci.yml
  name: ci
  on:
    pull_request:
    push:
      branches: [main]
      tags: ["v*"]

  permissions:
    contents: read
    packages: write

  env:
    REGISTRY: ghcr.io
    OWNER: lucian014          # must be lowercase

  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with: { python-version: "3.11" }
        - run: pip install ruff pytest -r src/router/requirements.lock   # whatever the tests import
        - run: ruff check src scripts
        - run: pytest -q

    images:
      needs: test
      runs-on: ubuntu-latest
      strategy:
        fail-fast: false
        matrix:
          include:
            - { name: moe-gateway, dockerfile: src/gateway/Dockerfile }
            - { name: moe-worker,  dockerfile: src/worker/Dockerfile }
            - { name: moe-router,  dockerfile: src/router/Dockerfile }
            - { name: moe-tools,   dockerfile: scripts/manage_models/Dockerfile }
      steps:
        - uses: actions/checkout@v4
        - uses: docker/setup-buildx-action@v3
        - uses: docker/login-action@v3
          if: github.event_name != 'pull_request'
          with:
            registry: ghcr.io
            username: ${{ github.actor }}
            password: ${{ secrets.GITHUB_TOKEN }}
        - id: meta
          uses: docker/metadata-action@v5
          with:
            images: ${{ env.REGISTRY }}/${{ env.OWNER }}/${{ matrix.name }}
            tags: |
              type=sha,prefix=sha-
              type=semver,pattern={{version}}
              type=ref,event=pr
        - uses: docker/build-push-action@v6
          with:
            context: .
            file: ${{ matrix.dockerfile }}
            push: ${{ github.event_name != 'pull_request' }}
            tags: ${{ steps.meta.outputs.tags }}
            labels: ${{ steps.meta.outputs.labels }}
            secrets: |
              hf_token=${{ secrets.HF_TOKEN }}
            cache-from: type=gha,scope=${{ matrix.name }}
            cache-to: type=gha,mode=max,scope=${{ matrix.name }}
  Notes:
  - secrets: hf_token=... maps onto your existing RUN --mount=type=secret,id=hf_token, so download_checkpoints.py works unchanged.
  - PRs from forks don't get secrets, so the router build fails for them. For your own PRs it works. If you want fork PRs to pass, skip the router in that case with if: matrix.name != 'moe-router' || github.event.pull_request.head.repo.fork == false.
  - Pin DOMAIN_REVISION / COMPLEXITY_REVISION to commit hashes as build args, so one image tag always means the same checkpoints.

  2.3 After the first push to main

  Go to github.com/Lucian014 → Packages and set each package to public, or decide on private (step 0.4). Check that each one is linked to the repo.

  Check: podman pull ghcr.io/lucian014/moe-router:sha-<short> works from a machine that isn't logged in (if public).

  ---

  Phase 3: Chart skeleton

  3.1 Layout
  charts/moe-router/
    Chart.yaml
    values.yaml               # portable defaults: no hostnames, no IPs
    values.schema.json        # added in Phase 7
    templates/
      _helpers.tpl
      gateway/{deployment,service}.yaml
      router/deployment.yaml
      workers/deployments.yaml       # range over models
      ollama/{deployment,service,pvc}.yaml
      postgres/{statefulset,service,secret}.yaml
      rabbitmq/{statefulset,service,service-headless,secret}.yaml
      jobs/migrate-seed.yaml
      config/catalog-configmap.yaml
      ingress.yaml / httproute.yaml
      keda/scaledobjects.yaml        # only if .Values.keda.enabled
      NOTES.txt
    ci/ci-values.yaml               # used by chart-testing in kind
  values/
    values-virtualbox.yaml          # your cluster: nodeSelectors, local registry
    values-openstack.yaml

  3.2 Chart.yaml
  apiVersion: v2
  name: moe-router
  version: 0.1.0        # chart version
  appVersion: "0.1.0"   # default image tag

  3.3 values.yaml (the shape)
  global:
    imageRegistry: ghcr.io/lucian014
    imagePullSecrets: []
    storageClass: ""            # "" = cluster default

  gateway:
    image: { repository: moe-gateway, tag: "", pullPolicy: IfNotPresent }
    replicas: 1
    resources: { requests: { cpu: 100m, memory: 256Mi } }
    nodeSelector: {}
    tolerations: []
    affinity: {}

  router:
    image: { repository: moe-router, tag: "" }
    complexityThreshold: "0.5"
    catalogRefreshSeconds: 30
    resources: { requests: { cpu: "1", memory: 3Gi }, limits: { memory: 4Gi } }

  worker:
    image: { repository: moe-worker, tag: "" }

  tools:
    image: { repository: moe-tools, tag: "" }

  ollama:
    enabled: true
    image: { repository: ollama/ollama, tag: "0.x.y" }   # pin it
    contextLength: 8192          # -> OLLAMA_CONTEXT_LENGTH
    persistence: { size: 50Gi }
    gpu: 0                        # >0 adds nvidia.com/gpu limits

  catalog:                        # replaces src/catalog/models.yaml as the deploy-time source
    models:
      - { id: "gemma3:1b", engine: ollama, category: open-qa, tier: low }
      - { id: "gemma3:4b", engine: ollama, category: open-qa, tier: high, default: true }

  postgresql:
    enabled: true
    image: { repository: postgres, tag: "16" }
    auth: { database: model_registry, username: moe, existingSecret: "" }
    persistence: { size: 5Gi }
    external: { host: "", port: 5432 }   # used when enabled=false

  rabbitmq:
    enabled: true
    image: { repository: rabbitmq, tag: "3.13-management" }
    auth: { username: moe, existingSecret: "" }
    persistence: { size: 2Gi }
    external: { host: "", port: 5672 }

  ingress:   { enabled: false, className: "", host: "", tls: [] }
  httpRoute: { enabled: false, parentRefs: [], hostnames: [] }
  keda:      { enabled: false }

  3.4 _helpers.tpl essentials
  {{- define "moe.fullname" -}}{{ .Release.Name | trunc 40 | trimSuffix "-" }}{{- end }}

  {{- define "moe.image" -}}
  {{- $reg := .root.Values.global.imageRegistry -}}
  {{- $tag := .img.tag | default .root.Chart.AppVersion -}}
  {{- if contains "/" .img.repository }}{{ .img.repository }}:{{ $tag }}
  {{- else }}{{ $reg }}/{{ .img.repository }}:{{ $tag }}{{ end }}
  {{- end }}

  {{- define "moe.rabbitHost" -}}
  {{- if .Values.rabbitmq.enabled }}{{ include "moe.fullname" . }}-rabbitmq{{ else }}{{ required "rabbitmq.external.host" .Values.rabbitmq.external.host }}{{ end }}
  {{- end }}

  {{- define "moe.k8sName" -}}{{ . | lower | replace ":" "-" | replace "." "-" | replace "_" "-" | trunc 50 | trimSuffix "-" }}{{- end }}
  moe.k8sName has to produce exactly the same names as shared.utils.sanitize_k8s_name. Compare the two in a unit test, or have Python generate the queue name from MODEL_NAME the way it does today and use the Helm name only for the Deployment.

  Usage: image: {{ include "moe.image" (dict "root" . "img" .Values.gateway.image) }}

  Also add the standard labels (app.kubernetes.io/name, instance, component, version) to everything. Selectors use only name, instance and component, because selectors are immutable.

  ---

  Phase 4: Port the manifests into templates (one component at a time)

  Order: Postgres → RabbitMQ → gateway → router → Ollama → workers → jobs → exposure. After each component, run helm template and diff it against the old YAML (dyff between old.yaml new.yaml, or kubectl diff -f - against the live cluster).

  4.1 Secrets: generated once, stable across upgrades
  {{- if not .Values.postgresql.auth.existingSecret }}
  {{- $name := printf "%s-postgres" (include "moe.fullname" .) }}
  {{- $old := lookup "v1" "Secret" .Release.Namespace $name }}
  apiVersion: v1
  kind: Secret
  metadata:
    name: {{ $name }}
    annotations: { "helm.sh/resource-policy": keep }
  type: Opaque
  data:
    password: {{ if $old }}{{ index $old.data "password" }}{{ else }}{{ randAlphaNum 24 | b64enc }}{{ end }}
  {{- end }}
  lookup returns nothing under helm template and Argo CD, so it would generate a new password on every render. For those, and for production in general, support existingSecret. Delete k8s/infra/*/…_secret.yaml and admin123 for good.

  4.2 Postgres
  - volumeClaimTemplates with storageClassName taken from global.storageClass (omit the field when it's empty), instead of a separate PVC.
  - PGDATA=/var/lib/postgresql/data/pgdata. Without it, fresh block volumes (Cinder, EBS) fail initdb because of lost+found.
  - Remove the nodeSelector and use {{- with .Values.postgresql.nodeSelector }} instead.
  - Keep your pg_isready probes.

  4.3 RabbitMQ
  - Add a headless Service (clusterIP: None) and make it the StatefulSet's serviceName. Keep the ClusterIP Service for clients.
  - Add probes: rabbitmq-diagnostics -q ping for liveness and rabbitmq-diagnostics -q check_port_connectivity for readiness.

  4.4 Gateway and router
  - Env from values and helpers. Hosts come from moe.rabbitHost / moe.pgHost, secrets via secretKeyRef to the generated Secret.
  - Use imagePullPolicy: IfNotPresent (tags are immutable now).
  - Probes from 1.3, plus resources and securityContext:
  securityContext:
    runAsNonRoot: true
    allowPrivilegeEscalation: false
    capabilities: { drop: [ALL] }
    seccompProfile: { type: RuntimeDefault }

  4.5 Ollama and workers
  - Ollama: PVC instead of hostPath, strategy: { type: Recreate } (an RWO volume can't move during a rolling update), OLLAMA_CONTEXT_LENGTH from values, and resources.limits["nvidia.com/gpu"] when ollama.gpu > 0.
  - Workers: one template that loops over the catalog, replacing build_manifests.py and worker_template.yaml:
  {{- range .Values.catalog.models }}
  {{- if and (ne (.enabled | toString) "false") (eq .engine "ollama") }}
  ---
  apiVersion: apps/v1
  kind: Deployment
  metadata:
    name: {{ include "moe.fullname" $ }}-worker-{{ include "moe.k8sName" .id }}
  spec:
    replicas: {{ .replicas | default 1 }}
    selector: { matchLabels: { app.kubernetes.io/component: worker, moe/model: {{ include "moe.k8sName" .id }} } }
    template:
      ...
        env:
          - { name: MODEL_NAME, value: {{ .id | quote }} }
          - { name: ENGINE_TYPE, value: {{ .engine | quote }} }
          - { name: INFERENCE_HOST, value: "http://{{ include "moe.fullname" $ }}-ollama:11434" }
  {{- end }}{{- end }}
    Inside range, use $ for the root context. This is the most common mistake here.

  4.6 Migration, seed and provisioning Job
  - One Job running the moe-tools entrypoint from 1.5, with the catalog mounted from a ConfigMap rendered from .Values.catalog:
  annotations:
    "helm.sh/hook": post-install,post-upgrade
    "helm.sh/hook-weight": "0"
    "helm.sh/hook-delete-policy": before-hook-creation,hook-succeeded
  - Use post-install, not pre-install: on a first install Postgres doesn't exist yet when pre-install hooks run. The router retries until the tables exist (1.2), so the order sorts itself out.
  - Set backoffLimit: 6 and activeDeadlineSeconds; ollama pull can take minutes.
  - Afterwards, delete scripts/manage_models/*.yaml.

  4.7 Exposure

  - ingress.yaml (generic, ingressClassName from values) and httproute.yaml (Gateway API, parentRefs from values), both disabled by default. Gateway Service stays ClusterIP. On OpenStack you expose the ingress controller with an Octavia LB, not the gateway itself.
  - Add a NOTES.txt that prints the port-forward command, or the URL when ingress is enabled.

  4.8 KEDA (optional)

  Wrap the ScaledObjects in {{- if .Values.keda.enabled }} and also guard with .Capabilities.APIVersions.Has "keda.sh/v1alpha1". That way the chart still installs on clusters without KEDA.

  Check after Phase 4:
  helm lint charts/moe-router
  helm template t charts/moe-router | kubeconform -strict -summary -ignore-missing-schemas
  helm template t charts/moe-router -f values/values-virtualbox.yaml | grep -nE '192\.168|k8s-(infra|route|compute)|cluster\.local|:latest'
  The last command should only match lines from values-virtualbox.yaml. Without -f it should match nothing.

  ---

  Phase 5: Switch your dev loop (Skaffold) to the chart

  5.1 Remove the IP from skaffold.yaml. Use short artifact names and set the registry per machine:
  skaffold config set default-repo 192.168.56.102:30500
  skaffold config set --global insecure-registries 192.168.56.102:30500

  5.2 skaffold.yaml
  apiVersion: skaffold/v4beta7
  kind: Config
  metadata: { name: moe-router }
  build:
    artifacts:
      - image: moe-gateway
        context: .
        custom:
          buildCommand: podman build -t $IMAGE -f src/gateway/Dockerfile . && podman push --tls-verify=false $IMAGE
          dependencies: { paths: [src/gateway/**, src/shared/**] }
      - image: moe-router
        context: .
        custom:
          buildCommand: podman build --secret id=hf_token,env=HF_TOKEN -t $IMAGE -f src/router/Dockerfile . && podman push --tls-verify=false $IMAGE
          dependencies:
            paths: [src/router/**, src/shared/**, src/database/**]
            ignore: [src/router/checkpoints/**, "**/__pycache__/**"]
      - image: moe-worker
        ...
      - image: moe-tools
        ...
  deploy:
    helm:
      releases:
        - name: moe
          chartPath: charts/moe-router
          namespace: moe
          createNamespace: true
          valuesFiles: [values/values-virtualbox.yaml]
          setValueTemplates:
            global.imageRegistry: ""        # full refs come from skaffold below
            gateway.image.repository: "{{.IMAGE_REPO_moe_gateway}}"
            gateway.image.tag: "{{.IMAGE_TAG_moe_gateway}}"
            router.image.repository: "{{.IMAGE_REPO_moe_router}}"
            router.image.tag: "{{.IMAGE_TAG_moe_router}}"
            worker.image.repository: "{{.IMAGE_REPO_moe_worker}}"
            worker.image.tag: "{{.IMAGE_TAG_moe_worker}}"
            tools.image.repository: "{{.IMAGE_REPO_moe_tools}}"
            tools.image.tag: "{{.IMAGE_TAG_moe_tools}}"
  Skaffold replaces - with _ in those template keys. If a key doesn't resolve, run skaffold render and check what it produced. The moe.image helper already passes full references (those containing /) through unchanged.

  5.3 values/values-virtualbox.yaml holds everything specific to your machines:
  gateway:   { nodeSelector: { kubernetes.io/hostname: k8s-route } }
  router:    { nodeSelector: { kubernetes.io/hostname: k8s-route } }
  postgresql: { nodeSelector: { kubernetes.io/hostname: k8s-infra } }
  rabbitmq:   { nodeSelector: { kubernetes.io/hostname: k8s-infra } }
  ollama:    { nodeSelector: { kubernetes.io/hostname: k8s-compute }, resources: { limits: { memory: 24Gi } } }
  Better: label your nodes with a role (kubectl label node k8s-compute moe/role=compute) and select on moe/role, so the same pattern works on OpenStack.

  5.4 Cut over
  1. Back up the registry DB: ./dbctl.sh exec pg_dump > backup.sql.
  2. Deploy the chart into the new moe namespace next to the old stack. Restore the dump or let the seed fill it, and test /generate through a port-forward.
  3. Delete the old namespaces (platform, gateway, processing) and the in-cluster registry only once the new namespace works. Keep the registry if you still push dev images to it. It then becomes dev infrastructure that the chart doesn't ship.
  4. Delete k8s/, scripts/build_manifests.py, scripts/templates/, scripts/manage_models/*.yaml. Point dbctl.sh at the new namespace and secret names (or read them from env).

  Check: skaffold dev rebuilds and redeploys on a code change, like before.

  ---

  Phase 6: Test the chart in CI and publish it

  6.1 charts/moe-router/ci/ci-values.yaml makes the chart small enough for a GitHub runner:
  ollama: { persistence: { size: 5Gi }, contextLength: 2048 }
  catalog: { models: [ { id: "qwen2.5:0.5b", engine: ollama, category: open-qa, tier: low, default: true } ] }
  router: { resources: { requests: { cpu: 500m, memory: 2Gi } } }

  6.2 Add a chart job to ci.yml
    chart:
      needs: images
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
          with: { fetch-depth: 0 }
        - uses: azure/setup-helm@v4
        - uses: helm/chart-testing-action@v2
        - run: ct lint --charts charts/moe-router --validate-maintainers=false
        - run: |
            helm template t charts/moe-router -f charts/moe-router/ci/ci-values.yaml \
              | docker run -i --rm ghcr.io/yannh/kubeconform:latest -strict -summary -ignore-missing-schemas
        - uses: helm/kind-action@v1
        - run: |
            ct install --charts charts/moe-router \
              --helm-extra-set-args "--set=global.imageRegistry=ghcr.io/lucian014 --set=gateway.image.tag=sha-${GITHUB_SHA::7} --set=router.image.tag=sha-${GITHUB_SHA::7} --set=worker.image.tag=sha-${GITHUB_SHA::7} --set=tools.image.tag=sha-${GITHUB_SHA::7}" \
              --helm-extra-args "--timeout 15m"
  - The kubeconform image is pinned to latest here only for brevity. Pin a version, like everything else.
  - On PRs the images aren't pushed. Either build them with load: true in the same job and run kind load docker-image, or run ct install only on main. Start with main only.
  - Add a helm test pod (templates/tests/smoke.yaml) that POSTs a prompt to the gateway and expects a 200. ct install runs it automatically. This is your end-to-end smoke test.

  6.3 Publish the chart to GHCR as an OCI artifact (on tags only)
    release-chart:
      if: startsWith(github.ref, 'refs/tags/v')
      needs: chart
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: azure/setup-helm@v4
        - run: |
            V=${GITHUB_REF_NAME#v}
            helm package charts/moe-router --version "$V" --app-version "$V"
            echo "${{ secrets.GITHUB_TOKEN }}" | helm registry login ghcr.io -u "${{ github.actor }}" --password-stdin
            helm push moe-router-$V.tgz oci://ghcr.io/lucian014/charts
  Anyone can then install with:
  helm install moe oci://ghcr.io/lucian014/charts/moe-router --version 0.1.0 -n moe --create-namespace

  6.4 Releasing: git tag v0.1.0 && git push --tags produces images tagged 0.1.0 and chart 0.1.0 with appVersion 0.1.0. Chart and images always match.

  ---

  Phase 7: Prove portability and harden

  1. values.schema.json: types and required fields for the values above. helm lint and helm install then reject a bad values file before anything is deployed.
  2. Install the same chart version in three places, with only the values file changing:
     - kind with defaults (CI already does this)
     - VirtualBox with values-virtualbox.yaml
     - OpenStack with values-openstack.yaml (Cinder storageClass, ollama.gpu: 1 plus a GPU nodeSelector/toleration, ingress.className, postgresql.auth.existingSecret)

     Record this in the thesis as your portability evidence.
  3. Hardening, in templates gated by values:
     - NetworkPolicies: default deny, plus gateway→rabbitmq, router→rabbitmq/postgres, worker→rabbitmq/ollama, tools→postgres/ollama, ingress→gateway, and DNS egress.
     - A PodDisruptionBudget for the gateway when replicas > 1.
     - A pg_dump CronJob.(Kubernetes version, a default StorageClass, an ingress controller or Gateway API, and optionally KEDA and the GPU Operator).

---

Phase 8 (optional): Continuous deployment to your cluster

GitHub-hosted runners can't reach 192.168.56.x. Two clean options:
- Pull-based (recommended): install Argo CD or Flux in the cluster and point it at oci://ghcr.io/lucian014/charts/moe-router (follow tags) or at charts/ in the repo. Nothing needs to be exposed to the internet. Because of the lookup caveat in 4.1, use existingSecret there.
- Push-based: a self-hosted runner on your host machine runs helm upgrade --install after release-chart.

---

Order and rough effort

┌─────────────────────────────┬──────────┬──────────────────────────────────┐
│            Phase            │  Effort  │             Merge as             │
├─────────────────────────────┼──────────┼──────────────────────────────────┤
│ 0 Decisions                 │ 0.5 day  │ —                                │
├─────────────────────────────┼──────────┼──────────────────────────────────┤
│ 1 Configurable services     │ 2–3 days │ PR 1                             │
├─────────────────────────────┼──────────┼──────────────────────────────────┤
│ 2 GHCR CI                   │ 1 day    │ PR 2                             │
├─────────────────────────────┼──────────┼──────────────────────────────────┤
│ 3–4 Chart                   │ 4–6 days │ PRs 3–4 (infra first, then apps) │
├─────────────────────────────┼──────────┼──────────────────────────────────┤
│ 5 Skaffold cut-over         │ 1 day    │ PR 5 (deletes k8s/)              │
├─────────────────────────────┼──────────┼──────────────────────────────────┤
│ 6 Chart CI and publish      │ 1–2 days │ PR 6                             │
├─────────────────────────────┼──────────┼──────────────────────────────────┤
│ 7 Portability and hardening │ 2–3 days │ PR 7                             │
├─────────────────────────────┼──────────┼──────────────────────────────────┤
│ 8 CD                        │ 1 day    │ optional                         │
└─────────────────────────────┴──────────┴──────────────────────────────────┘

That's roughly 2.5–3 weeks of focused work. It fits the Dec–Jan slot from the review if you start Phase 1 now, since Phase 1 also helps the OpenAI-API work.

If you get stuck on a specific template or the Skaffold value keys, send me the file and I'll review it.