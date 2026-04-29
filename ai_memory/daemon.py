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

logger = logging.getLogger(__name__)

# How often we wake up to evaluate triggers.
_POLL_INTERVAL_SECONDS = 60


@dataclass
class DaemonRunState:
    """Mutable state held while the daemon loop is running."""

    last_scheduled_run_date: str | None = None  # 'YYYY-MM-DD' in local time
    last_pressure_run_at: int = 0
    last_idle_run_at: int = 0


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

    # 1) Scheduled trigger
    schedule_str = service.config.dream.schedule_cron
    scheduled_hhmm = _parse_hhmm(schedule_str)
    today_iso = now_local.strftime("%Y-%m-%d")
    if (
        scheduled_hhmm is not None
        and state.last_scheduled_run_date != today_iso
        and (now_local.hour, now_local.minute) >= scheduled_hhmm
    ):
        _run_pass(service, "scheduled", log_path)
        state.last_scheduled_run_date = today_iso
        return

    # 2) Pressure trigger
    last_dream_completed = service.store.last_dream_completed_at() or 0
    new_turns = service.store.count_turns_since(last_dream_completed)
    if new_turns >= service.config.dream.pressure_trigger_turns:
        _run_pass(service, "pressure", log_path, extra={"new_turns": new_turns})
        state.last_pressure_run_at = now_unix
        return

    # 3) Idle trigger
    idle_threshold_seconds = service.config.dream.idle_trigger_minutes * 60
    no_recent_turns = service.store.count_turns_since(now_unix - idle_threshold_seconds) == 0
    has_unconsolidated = any(
        ep.consolidated_at is None
        for ep in service.store.episodes_since(last_dream_completed)
    )
    enough_gap_since_last_idle = (
        now_unix - state.last_idle_run_at >= idle_threshold_seconds
    )
    if no_recent_turns and has_unconsolidated and enough_gap_since_last_idle:
        _run_pass(service, "idle", log_path)
        state.last_idle_run_at = now_unix
        return

    # No triggers fired — emit a heartbeat once per ten minutes for ops visibility.
    if now_unix % 600 < _POLL_INTERVAL_SECONDS:
        _emit(
            log_path,
            {
                "event": "heartbeat",
                "new_turns_since_last_dream": new_turns,
                "last_dream_completed_at": last_dream_completed,
            },
        )


def _run_pass(
    service: MemoryService, trigger: str, log_path: Path, extra: dict | None = None
) -> None:
    """Fire one dream pass and journal the outcome."""
    started = int(time.time())
    _emit(log_path, {"event": "dream_start", "trigger": trigger, **(extra or {})})
    report = service.dream(trigger=trigger)
    _emit(
        log_path,
        {
            "event": "dream_complete",
            "trigger": trigger,
            "log_id": report.log_id,
            "episodes_processed": report.episodes_processed,
            "notes_added": report.notes_added,
            "notes_invalidated": report.notes_invalidated,
            "notes_promoted_to_profile": report.notes_promoted_to_profile,
            "notes_pruned": report.notes_pruned,
            "duration_seconds": int(time.time()) - started,
        },
    )


# --- Helpers ----------------------------------------------------------------


def _emit(log_path: Path, payload: dict) -> None:
    """Append one JSON line to the daemon log."""
    payload = {"ts": int(time.time()), **payload}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _claim_pid(pid_path: Path, expected_name: str = "ai-memory") -> bool:
    """Atomically claim a pidfile. Returns False if a live ai-memory process already holds it.

    `expected_name` is matched (case-insensitive, substring) against the
    process executable name so that a recycled PID belonging to an unrelated
    process (e.g. pwsh) doesn't block startup.
    """
    if pid_path.exists():
        try:
            other_pid = int(pid_path.read_text().strip())
            if _process_alive(other_pid, expected_name=expected_name):
                return False
        except (ValueError, OSError):
            pass  # stale or unreadable -> we'll overwrite
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    return True


def _process_alive(pid: int, expected_name: str | None = None) -> bool:
    """Cross-platform 'is this pid running' check.

    If `expected_name` is given (case-insensitive substring), the function
    also verifies that the process executable contains that name.  This
    guards against PID recycling: a recycled PID held by an unrelated
    process (e.g. pwsh) looks alive but isn't *our* process.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            import ctypes.wintypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if not handle:
                return False
            try:
                if expected_name is not None:
                    # QueryFullProcessImageNameW to get the exe path
                    buf = ctypes.create_unicode_buffer(1024)
                    size = ctypes.wintypes.DWORD(1024)
                    ok = ctypes.windll.kernel32.QueryFullProcessImageNameW(
                        handle, 0, buf, ctypes.byref(size)
                    )
                    if ok:
                        exe_path = buf.value.lower()
                        if expected_name.lower() not in exe_path:
                            return False  # PID recycled by a different program
                return True
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            if expected_name is not None:
                # Best-effort name check via /proc on Linux/macOS
                try:
                    exe = os.readlink(f"/proc/{pid}/exe")
                    if expected_name.lower() not in exe.lower():
                        return False
                except OSError:
                    pass  # /proc not available or unreadable — trust the signal
            return True
        except OSError:
            return False


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
