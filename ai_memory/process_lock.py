"""
Cross-process single-flight lock backed by a pidfile.

A small, dependency-free primitive shared by two callers:

* the dream **daemon** — at most one daemon process at a time (`daemon.pid`), and
* the dream **pass** itself — at most one consolidation pass at a time
  (`dream.pid`), across every entry point that can start one: the daemon, the
  CLI (`ai-memory dream`), the MCP server (`memory_dream`), and the Claude Code
  PreCompact hook (which fires `ai-memory dream --trigger idle` fire-and-forget).

The pass-level lock exists because `dream()` selects *all* pending episodes up
front with no per-row claim; two passes running at once therefore double-process
the same episodes — wasted LLM spend on a throttled account, and near-duplicate
notes. The daemon's own pid-file never covered the hook/CLI path, so those could
pile up. See ADR-0013.

Lives in its own module (not `daemon.py`) so the service layer can claim a lock
without importing the daemon — that would be a circular import — and to keep the
concern discrete for the planned C# port.

The pidfile stores the owning process id. A claim succeeds unless the file
already names a *live* process. An optional ``expected_name`` check (off by
default) additionally requires the live holder's exe path to contain a given
substring, to guard against PID recycling — where the OS hands a dead holder's
pid to an unrelated program (e.g. ``pwsh``) that would otherwise look alive and
block forever.

**Why ``expected_name`` defaults to ``None``:** on Windows the ``ai-memory.exe``
console-script and the venv ``python.exe`` are *launcher stubs* that spawn the
real interpreter (base ``python.exe``) as a child. The running image whose pid
ends up in the pidfile is therefore base python — whose path contains no
``ai-memory`` substring — so a name check of ``"ai-memory"`` would treat the
live holder as recycled and silently defeat the lock. Leaving the check off
keeps the lock correct everywhere; the recycle edge case (stale file + reused
pid) is rare and self-heals once the unrelated process exits.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path


def process_alive(pid: int, expected_name: str | None = None) -> bool:
    """Cross-platform 'is this pid running' check.

    If ``expected_name`` is given (case-insensitive substring), also verify the
    process executable path contains that name, so a recycled PID held by an
    unrelated process isn't mistaken for a live holder.
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


class PidLock:
    """Best-effort single-flight lock backed by a pidfile.

    Usage::

        lock = PidLock(path)
        if not lock.acquire():
            return  # another live holder — skip
        try:
            ...  # critical section
        finally:
            lock.release()

    or as a context manager (inspect ``.acquired`` inside the block)::

        with PidLock(path) as lock:
            if not lock.acquired:
                return
            ...

    ``acquire()`` returns ``False`` (leaving any existing file untouched) when a
    live process already holds the lock, and ``True`` after writing our pid. A
    stale pidfile (holder dead, or — with ``expected_name`` — recycled to an
    unrelated process) is self-healing: the next claimant overwrites it. This is
    not crash-proof against ``kill -9`` leaving the file behind, but the liveness
    check means a leftover file never blocks forever.
    """

    def __init__(self, path: Path | str, expected_name: str | None = None) -> None:
        self.path = Path(path)
        self.expected_name = expected_name
        self.acquired = False

    def acquire(self) -> bool:
        if self.path.exists():
            try:
                other_pid = int(self.path.read_text(encoding="utf-8").strip())
                if process_alive(other_pid, expected_name=self.expected_name):
                    return False
            except (ValueError, OSError):
                pass  # stale / unreadable -> we'll overwrite
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(str(os.getpid()), encoding="utf-8")
        except OSError:
            return False
        self.acquired = True
        return True

    def release(self) -> None:
        if self.acquired:
            with contextlib.suppress(OSError):
                self.path.unlink(missing_ok=True)
            self.acquired = False

    def __enter__(self) -> PidLock:
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.release()
        return False
