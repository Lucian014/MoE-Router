#!/usr/bin/env bash
#
# smoke-test.sh -- check that an installed moe-router release actually works.
#
#   charts/moe-router/ci/smoke-test.sh <namespace> [release]          # infra, gateway, setup Job, workers
#   FULL=1 charts/moe-router/ci/smoke-test.sh <namespace> [release]   # + router, and one real prompt
#
# FULL needs a router that can download the private classifiers (a real HF token).
# Used by CI (.github/workflows/ci.yml, job `chart-install`) on a fresh kind cluster; works on any
# cluster. Prints diagnostics and exits non-zero on the first failure.
set -euo pipefail

NS="${1:?usage: smoke-test.sh <namespace> [release]}"
REL="${2:-moe}"
FULL="${FULL:-0}"
TIMEOUT="${TIMEOUT:-600s}"
PORT="${PORT:-18000}"

step() { echo; echo "=== $*"; }

diagnostics() {
    echo; echo "=================== FAILED: diagnostics ==================="
    kubectl get pods,jobs,pvc -n "$NS" -o wide || true
    kubectl get events -n "$NS" --sort-by=.lastTimestamp | tail -30 || true
    for pod in $(kubectl get pods -n "$NS" -o name 2>/dev/null); do
        echo "--- logs $pod"
        kubectl logs -n "$NS" "$pod" --all-containers --tail=40 2>&1 || true
    done
}
trap diagnostics ERR

REV=$(helm status "$REL" -n "$NS" -o json | python3 -c "import json,sys; print(json.load(sys.stdin)['version'])")

step "Postgres and RabbitMQ"
kubectl rollout status -n "$NS" statefulset/"$REL"-postgres --timeout="$TIMEOUT"
kubectl rollout status -n "$NS" statefulset/"$REL"-rabbitmq --timeout="$TIMEOUT"

step "setup Job $REL-setup-$REV (migrations, catalog seed, model pulls)"
kubectl wait -n "$NS" --for=condition=complete job/"$REL"-setup-"$REV" --timeout="$TIMEOUT"

step "gateway"
kubectl rollout status -n "$NS" deployment/"$REL"-gateway --timeout="$TIMEOUT"

step "workers (ready = their model is pulled and they consume)"
WORKERS=$(kubectl get deploy -n "$NS" -l app.kubernetes.io/instance="$REL",app.kubernetes.io/component=worker -o name)
[ -n "$WORKERS" ] || { echo "no worker Deployments found"; false; }
for w in $WORKERS; do kubectl rollout status -n "$NS" "$w" --timeout="$TIMEOUT"; done

if [ "$FULL" = "1" ]; then
    step "router (classifier download + catalog)"
    kubectl rollout status -n "$NS" deployment/"$REL"-router --timeout="$TIMEOUT"
fi

step "gateway over HTTP"
kubectl port-forward -n "$NS" svc/"$REL"-gateway "$PORT":8000 >/dev/null 2>&1 &
PF=$!
trap 'kill $PF 2>/dev/null || true' EXIT
for _ in $(seq 1 30); do curl -sf "localhost:$PORT/health" >/dev/null && break; sleep 1; done
curl -sf "localhost:$PORT/ready"; echo

if [ "$FULL" = "1" ]; then
    step "one prompt end to end: gateway -> router -> worker"
    RESPONSE=$(curl -sf --max-time 300 -X POST "localhost:$PORT/generate" \
        -H 'content-type: application/json' -d '{"prompt": "What is the capital of France? Answer in one word."}')
    echo "$RESPONSE" | python3 -c "
import json, sys
r = json.load(sys.stdin)
assert r.get('response', '').strip(), f'empty response: {r}'
ro = r['routing']
print(f\"routed {ro['task_type']}/{ro['complexity']} -> {r['model']} in {r['latency_ms']/1000:.1f}s: {r['response'].strip()[:80]!r}\")"
fi

echo; echo "=== smoke test passed (FULL=$FULL)"
