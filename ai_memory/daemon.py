"""
Dream daemon: long-running process that fires the consolidation cycle on
schedule, on idle, or under pressure.

This is a separate process from the MCP server (which is short-lived and
spawned by Claude Desktop / Cursor / etc. via stdio). The daemon watches
the same on-disk database and fires `dream()` when any trigger condition
is met.

Triggers (all checked once per polling tick):
    - SCHEDULED:  Once per day at config.schedule_time_local (HH:MM string).
    - PRESSURE:   When more than `pressure_trigger_turns` turns have been
                  written since the last completed dream pass.
    - IDLE:       When no new turns have arrived for `idle_trigger_minutes`
                  AND there is at least one un-consolidated episode.

A pid-file at `<home>/daemon.pid` prevents double-runs. SIGINT / Ctrl+C
shuts down cleanly. On Windows, the daemon is intended to be installed as
a service via `examples/install-service.ps1` (using NSSM under the hood).

Logs are written to `<home>/logs/daemon-YYYY-MM-DD.jsonl` — one JSON object
per significant event so we can grep/tail them.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ai_memory.config import Config
from ai_memory.core.service import MemoryService
from ai_memory.process_lock import process_alive
from ai_memory.timestamps import now_iso, unix_to_iso

logger = logging.getLogger(__name__)

# How often we wake up to evaluate triggers.
_POLL_INTERVAL_SECONDS = 60

# Circuit breaker: if this many consecutive passes make no forward progress
# *and* fail at least one episode (or raise), stop re-firing the same doomed
# work and back off for the cooldown. Prevents the runaway 63s retry loop that
# logged 50k+ failed passes when a single episode poison-pilled every attempt.
_CIRCUIT_BREAKER_THRESHOLD = 3
_CIRCUIT_BREAKER_COOLDOWN_SECONDS = 3600  # 1 hour


@dataclass
class DaemonRunState:
    """Mutable state held while the daemon loop is running."""

    last_scheduled_run_date: str | None = None  # 'YYYY-MM-DD' in local time
    last_pressure_run_at: int = 0
    last_idle_run_at: int = 0
    consecutive_failures: int = 0  # passes that failed/made no progress in a row
    circuit_open_until: int = 0    # unix time; while now < this, skip all passes


class _StopRequested(Exception):
    """Raised inside the loop when SIGINT/SIGTERM is received."""


def run_watch(config: Config) -> int:
    """Run the dream daemon until interrupted. Returns an exit code."""
    pid_path = config.home / "daemon.pid"
    if not _claim_pid(pid_path):
        logger.error("Another daemon is already running (pid file at %s).", pid_path)
        return 2

    stop_event = threading.Event()
    _install_signal_handlers(stop_event)

    log_path_today = lambda: (
        config.logs_dir
        / f"daemon-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
    )
    config.logs_dir.mkdir(parents=True, exist_ok=True)

    service = MemoryService.build(config)
    service.start()
    state = DaemonRunState()

    _emit(log_path_today(), {"event": "daemon_start", "home": str(config.home)})
    logger.info("Dream daemon started; polling every %ds", _POLL_INTERVAL_SECONDS)

    try:
        while not stop_event.is_set():
            try:
                _tick(service=service, state=state, log_path=log_path_today())
            except Exception as exc:  # one tick failing must not kill the daemon
                logger.exception("Tick failed: %s", exc)
                _emit(log_path_today(), {"event": "tick_error", "error": str(exc)})
            stop_event.wait(_POLL_INTERVAL_SECONDS)
        return 0
    finally:
        service.stop()
        try:
            pid_path.unlink(missing_ok=True)
        except OSError:
            pass
        _emit(log_path_today(), {"event": "daemon_stop"})
        logger.info("Dream daemon stopped.")


# --- Tick logic -------------------------------------------------------------


def _tick(*, service: MemoryService, state: DaemonRunState, log_path: Path) -> None:
    """One polling tick: evaluate every trigger and fire at most one dream pass."""
    now_local = datetime.now()  # naive local time — fine for HH:MM comparison
    now_unix = int(time.time())

    # 0) Circuit breaker open? Skip all work until the cooldown expires.
    if now_unix < state.circuit_open_until:
        if now_unix % 600 < _POLL_INTERVAL_SECONDS:
            _emit(
                log_path,
                {
                    "event": "circuit_open_heartbeat",
                    "until": unix_to_iso(state.circuit_open_until),
                    "consecutive_failures": state.consecutive_failures,
                },
            )
        return

    # 1) Scheduled trigger
    schedule_str = service.config.dream.schedule_cron
    scheduled_hhmm = _parse_hhmm(schedule_str)
    today_iso = now_local.strftime("%Y-%m-%d")
    if (
        scheduled_hhmm is not None
        and state.last_scheduled_run_date != today_iso
        and (now_local.hour, now_local.minute) >= scheduled_hhmm
    ):
        _run_pass(service, state, "scheduled", log_path)
        state.last_scheduled_run_date = today_iso
        return

    # 2) Pressure trigger
    last_dream_completed = service.store.last_dream_completed_at() or "1970-01-01T00:00:00Z"
    new_turns = service.store.count_turns_since(last_dream_completed)
    if new_turns >= service.config.dream.pressure_trigger_turns:
        _run_pass(service, state, "pressure", log_path, extra={"new_turns": new_turns})
        state.last_pressure_run_at = now_unix
        return

    # 3) Idle trigger
    idle_threshold_seconds = service.config.dream.idle_trigger_minutes * 60
    no_recent_turns = service.store.count_turns_since(
        unix_to_iso(now_unix - idle_threshold_seconds)
    ) == 0
    has_unconsolidated = any(
        ep.consolidated_at is None
        for ep in service.store.episodes_since(last_dream_completed)
    )
    enough_gap_since_last_idle = (
        now_unix - state.last_idle_run_at >= idle_threshold_seconds
    )
    if no_recent_turns and has_unconsolidated and enough_gap_since_last_idle:
        _run_pass(service, state, "idle", log_path)
        state.last_idle_run_at = now_unix
        return

    # No triggers fired — emit a heartbeat once per ten minutes for ops visibility.
    if now_unix % 600 < _POLL_INTERVAL_SECONDS:
        _emit(
            log_path,
            {
                "event": "heartbeat",
                "new_turns_since_last_dream": new_turns,
                "last_dream_completed_at": last_dream_completed,  # already ISO string
            },
        )


def _run_pass(
    service: MemoryService,
    state: DaemonRunState,
    trigger: str,
    log_path: Path,
    extra: dict | None = None,
) -> None:
    """Fire one dream pass, journal the outcome, and update the circuit breaker."""
    started = int(time.time())
    _emit(log_path, {"event": "dream_start", "trigger": trigger, **(extra or {})})
    try:
        report = service.dream(trigger=trigger)
    except Exception as exc:
        # dream() isolates per-episode failures internally, so reaching here
        # means something broke around the pass itself. Count it and let the
        # breaker decide whether to back off.
        state.consecutive_failures += 1
        logger.exception("Dream pass (%s) raised: %s", trigger, exc)
        _emit(
            log_path,
            {
                "event": "dream_error",
                "trigger": trigger,
                "error": str(exc)[:500],
                "consecutive_failures": state.consecutive_failures,
            },
        )
        _maybe_open_circuit(state, log_path)
        return

    _emit(
        log_path,
        {
            "event": "dream_complete",
            "trigger": trigger,
            "log_id": report.log_id,
            "episodes_processed": report.episodes_processed,
            "episodes_failed": report.episodes_failed,
            "notes_added": report.notes_added,
            "notes_invalidated": report.notes_invalidated,
            "notes_promoted_to_profile": report.notes_promoted_to_profile,
            "notes_pruned": report.notes_pruned,
            "duration_seconds": int(time.time()) - started,
        },
    )

    # A "stuck" pass = no episode consolidated AND at least one failed. Repeated
    # stuck passes trip the breaker; any forward progress resets the counter.
    if report.episodes_processed == 0 and report.episodes_failed > 0:
        state.consecutive_failures += 1
    else:
        state.consecutive_failures = 0
    _maybe_open_circuit(state, log_path)


def _maybe_open_circuit(state: DaemonRunState, log_path: Path) -> None:
    """Open the circuit breaker once consecutive failures hit the threshold."""
    if state.consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
        state.circuit_open_until = int(time.time()) + _CIRCUIT_BREAKER_COOLDOWN_SECONDS
        logger.error(
            "Circuit breaker OPEN after %d consecutive failed/stuck dream passes; "
            "backing off until %s.",
            state.consecutive_failures,
            unix_to_iso(state.circuit_open_until),
        )
        _emit(
            log_path,
            {
                "event": "circuit_open",
                "consecutive_failures": state.consecutive_failures,
                "until": unix_to_iso(state.circuit_open_until),
            },
        )
        state.consecutive_failures = 0


# --- Helpers ----------------------------------------------------------------


def _emit(log_path: Path, payload: dict) -> None:
    """Append one JSON line to the daemon log."""
    payload = {"ts": now_iso(), **payload}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _claim_pid(pid_path: Path, expected_name: str | None = None) -> bool:
    """Atomically claim a pidfile. Returns False if a live process already holds it.

    `expected_name`, when given, is matched (case-insensitive, substring)
    against the holder's executable path to survive PID recycling. It defaults
    to None because the daemon actually runs as base python (the ai-memory.exe /
    venv launcher spawns it as a child), so a name check of "ai-memory" would
    never match the live holder and would defeat the lock — see process_lock.py.
    """
    if pid_path.exists():
        try:
            other_pid = int(pid_path.read_text().strip())
            if process_alive(other_pid, expected_name=expected_name):
                return False
        except (ValueError, OSError):
            pass  # stale or unreadable -> we'll overwrite
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    return True


def _install_signal_handlers(stop_event: threading.Event) -> None:
    def _handler(signum: int, _frame: object) -> None:
        logger.info("Received signal %s; shutting down.", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handler)
    # SIGBREAK is Windows-specific; ignore on POSIX.
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _handler)  # type: ignore[attr-defined]


def _parse_hhmm(schedule: str) -> tuple[int, int] | None:
    """Parse "HH:MM" or a simple cron "M H * * *" string. Returns (hour, minute) or None."""
    schedule = schedule.strip()
    if ":" in schedule and " " not in schedule:
        try:
            hh, mm = schedule.split(":", 1)
            return (int(hh), int(mm))
        except ValueError:
            return None
    parts = schedule.split()
    if len(parts) >= 2:
        try:
            minute = int(parts[0])
            hour = int(parts[1])
            return (hour, minute)
        except ValueError:
            return None
    return None
