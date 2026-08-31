#!/usr/bin/env bash
#
# dbctl.sh — control tool for the in-cluster Postgres: run Alembic
# migrations, drop into psql, or run any other command that needs a DB
# connection, without manually opening a port-forward and exporting env
# vars every time.
#
# Usage:
#   ./dbctl.sh upgrade head            (apply all pending migrations)
#   ./dbctl.sh downgrade -1            (roll back one migration)
#   ./dbctl.sh current                 (show current migration)
#   ./dbctl.sh revision --autogenerate -m "add users table"
#
#   ./dbctl.sh exec <command...>       (run any command with the tunnel +
#                                        credentials already set up, e.g.:)
#   ./dbctl.sh exec python scripts/seed_models.py --from-yaml src/catalog/models.yaml
#   ./dbctl.sh exec python scripts/test_crud.py
#   ./dbctl.sh exec psql
#
# Anything you pass after the script name is forwarded as-is to `alembic`
# — UNLESS the first argument is `exec`, in which case everything after
# `exec` is run directly as its own command instead (tunnel + credentials
# are still set up the same way either way).
#
# Credentials (POSTGRES_DB/USER/PASSWORD) are pulled live from the
# `postgres-secret` k8s Secret — the same one the pod itself consumes
# (see k8s/infra/postgresql/postgresql-secret.yaml) — instead of being
# duplicated here. This is the single source of truth; if the secret
# rotates, this script picks it up automatically with no edits needed.
#
# Only HOST/PORT are overridden here, because from outside the cluster
# you reach Postgres through the port-forward tunnel on localhost, not
# through the in-cluster DNS name.

set -euo pipefail

# -- config: adjust here if your namespace/service/secret values change ----
NAMESPACE="platform"
SERVICE="postgres"
SECRET="postgres-secret"
LOCAL_PORT="5432"
REMOTE_PORT="5432"
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" != "exec" ]]; then
    cd "${SCRIPT_DIR}"
fi

if [[ $# -eq 0 ]]; then
    echo "Usage: $(basename "$0") <alembic-args...>" >&2
    echo "       $(basename "$0") exec <command...>" >&2
    exit 1
fi

echo "[*] Reading credentials from secret/${SECRET} in namespace ${NAMESPACE}..."
export POSTGRES_DB="$(kubectl get secret -n "${NAMESPACE}" "${SECRET}" -o jsonpath='{.data.POSTGRES_DB}' | base64 -d)"
export POSTGRES_USER="$(kubectl get secret -n "${NAMESPACE}" "${SECRET}" -o jsonpath='{.data.POSTGRES_USER}' | base64 -d)"
export POSTGRES_PASSWORD="$(kubectl get secret -n "${NAMESPACE}" "${SECRET}" -o jsonpath='{.data.POSTGRES_PASSWORD}' | base64 -d)"

if [[ -z "${POSTGRES_USER}" || -z "${POSTGRES_PASSWORD}" || -z "${POSTGRES_DB}" ]]; then
    echo "[!] Failed to read one or more keys from secret/${SECRET}. Check the" >&2
    echo "    secret exists and has POSTGRES_DB/POSTGRES_USER/POSTGRES_PASSWORD keys." >&2
    exit 1
fi

export POSTGRES_HOST="localhost"
export POSTGRES_PORT="${LOCAL_PORT}"

export PGHOST="${POSTGRES_HOST}"
export PGPORT="${POSTGRES_PORT}"
export PGDATABASE="${POSTGRES_DB}"
export PGUSER="${POSTGRES_USER}"
export PGPASSWORD="${POSTGRES_PASSWORD}"

PF_PID=""

cleanup() {
    if [[ -n "${PF_PID}" ]] && kill -0 "${PF_PID}" 2>/dev/null; then
        echo "[*] Closing port-forward (pid ${PF_PID})..."
        kill "${PF_PID}" 2>/dev/null || true
        wait "${PF_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[*] Starting port-forward to svc/${SERVICE} in namespace ${NAMESPACE}..."
kubectl port-forward -n "${NAMESPACE}" "svc/${SERVICE}" "${LOCAL_PORT}:${REMOTE_PORT}" \
    > /tmp/pf-postgres.log 2>&1 &
PF_PID=$!

echo "[*] Waiting for tunnel to be ready..."
for _ in $(seq 1 20); do
    if (echo > "/dev/tcp/localhost/${LOCAL_PORT}") 2>/dev/null; then
        echo "[OK] Tunnel is up."
        break
    fi
    sleep 0.5
done

if ! (echo > "/dev/tcp/localhost/${LOCAL_PORT}") 2>/dev/null; then
    echo "[!] Tunnel never came up. See /tmp/pf-postgres.log for details."
    exit 1
fi

if [[ "$1" == "exec" ]]; then
    shift
    if [[ $# -eq 0 ]]; then
        echo "Usage: $(basename "$0") exec <command...>" >&2
        exit 1
    fi
    echo "[*] Running: $*"
    "$@"
else
    echo "[*] Running: alembic $*"
    alembic "$@"
fi

echo "[OK] Done."