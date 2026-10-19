"""
shared/lifecycle.py -- start-up retries, readiness and graceful shutdown for the router and worker.

    stop = install_stop_signals()                       # SIGTERM / SIGINT set this event
    await retry(load_catalog, "loading the catalog", stop_event=stop)
    ...start consuming...
    set_ready(path, True)                               # readinessProbe: test -f <path>
    await stop.wait()
    set_ready(path, False)
    await mq_client.drain(grace_seconds)                # finish in-flight messages
"""
import asyncio
import signal
from pathlib import Path


class Stopped(Exception):
    """A shutdown was requested while retrying."""


async def retry(fn, what: str, *, max_attempts: int = 0, base_delay: float = 1.0, max_delay: float = 30.0,
                stop_event: asyncio.Event | None = None):
    """Await `fn()` until it succeeds, with exponential backoff (1, 2, 4, ... `max_delay` s).

    max_attempts=0 retries forever. Raises the last error when the attempts run out, and
    Stopped if `stop_event` is set while waiting between attempts."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return await fn()
        except Exception as e:
            if max_attempts and attempt >= max_attempts:
                raise
            delay = min(base_delay * 2 ** (attempt - 1), max_delay)
            of = f"/{max_attempts}" if max_attempts else ""
            print(f"[!] {what} failed (attempt {attempt}{of}): {e}. Retrying in {delay:g}s...")
            if stop_event is None:
                await asyncio.sleep(delay)
                continue
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=delay)
            except asyncio.TimeoutError:
                continue
            raise Stopped(f"{what}: shutdown requested") from None


def install_stop_signals() -> asyncio.Event:
    """Return an event that SIGTERM (Kubernetes stopping the pod) and SIGINT (Ctrl+C) set."""
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    return stop


def set_ready(path: str, ready: bool) -> None:
    """Create / remove the readiness file checked by the pod's exec readinessProbe."""
    p = Path(path)
    if ready:
        p.touch()
    else:
        p.unlink(missing_ok=True)
