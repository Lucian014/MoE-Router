"""Wait until TCP services accept connections, e.g. before the setup Job's steps.

    python scripts/manage_models/wait_for.py moe-postgres:5432 moe-ollama:11434

Waiting here, instead of failing and letting Kubernetes restart the pod, keeps the
Job's backoffLimit for real failures: restart backoff grows to minutes, and every
restart while Postgres or Ollama is still starting counts towards the limit.
WAIT_TIMEOUT_SECONDS (default 900) caps the wait; exits 1 when it runs out.
"""
import os
import socket
import sys
import time

TIMEOUT = float(os.getenv("WAIT_TIMEOUT_SECONDS", "900"))


def wait(target: str, deadline: float) -> bool:
    host, _, port = target.rpartition(":")
    attempt = 0
    while True:
        attempt += 1
        try:
            with socket.create_connection((host, int(port)), timeout=5):
                print(f"[OK] {target} is up")
                return True
        except OSError as e:
            if time.monotonic() >= deadline:
                print(f"[ERROR] {target} still down after {TIMEOUT:.0f}s: {e}", file=sys.stderr)
                return False
            if attempt == 1 or attempt % 10 == 0:
                print(f"[*] waiting for {target}: {e}")
            time.sleep(3)


def main() -> None:
    targets = sys.argv[1:]
    if not targets:
        sys.exit("usage: wait_for.py host:port [host:port ...]")
    deadline = time.monotonic() + TIMEOUT
    sys.exit(0 if all(wait(t, deadline) for t in targets) else 1)


if __name__ == "__main__":
    main()
