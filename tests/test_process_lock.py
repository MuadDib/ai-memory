"""Unit tests for the single-flight pidfile lock (ADR-0013).

These pin the mechanism that stops concurrent dream passes from piling up —
the gap where the daemon's pid-file never covered the CLI / PreCompact-hook
path. Deterministic: they use ``expected_name=None`` so the liveness decision
depends only on whether the pid is alive, not on the test runner's exe path.
"""
from __future__ import annotations

import os
from pathlib import Path

from ai_memory.process_lock import PidLock, process_alive

_DEAD_PID = 0x7FFFFFFF  # far above any real pid — never running


def test_process_alive_self_and_dead() -> None:
    assert process_alive(os.getpid()) is True
    assert process_alive(_DEAD_PID) is False
    assert process_alive(0) is False
    assert process_alive(-1) is False


def test_acquire_blocks_second_then_releases(tmp_path: Path) -> None:
    p = tmp_path / "dream.pid"

    first = PidLock(p, expected_name=None)
    assert first.acquire() is True
    assert p.exists()
    assert p.read_text().strip() == str(os.getpid())

    # A second claimant sees a live holder (our own pid) and is refused.
    second = PidLock(p, expected_name=None)
    assert second.acquire() is False
    assert second.acquired is False

    # Releasing frees the lock for the next claimant.
    first.release()
    assert not p.exists()

    third = PidLock(p, expected_name=None)
    assert third.acquire() is True
    third.release()


def test_stale_pidfile_is_reclaimed(tmp_path: Path) -> None:
    p = tmp_path / "dream.pid"
    p.write_text(str(_DEAD_PID), encoding="utf-8")

    lock = PidLock(p, expected_name=None)
    assert lock.acquire() is True  # dead holder -> we take over
    assert p.read_text().strip() == str(os.getpid())
    lock.release()


def test_garbage_pidfile_is_overwritten(tmp_path: Path) -> None:
    p = tmp_path / "dream.pid"
    p.write_text("not-a-pid", encoding="utf-8")

    lock = PidLock(p, expected_name=None)
    assert lock.acquire() is True  # unparseable -> overwrite
    lock.release()


def test_expected_name_mismatch_treated_as_recycled(tmp_path: Path) -> None:
    """A live pid whose exe path lacks expected_name is assumed recycled."""
    p = tmp_path / "dream.pid"
    p.write_text(str(os.getpid()), encoding="utf-8")

    lock = PidLock(p, expected_name="no-such-binary-name-zzz")
    assert lock.acquire() is True  # name mismatch -> not *our* process -> reclaim
    lock.release()


def test_context_manager_acquires_and_releases(tmp_path: Path) -> None:
    p = tmp_path / "dream.pid"
    with PidLock(p, expected_name=None) as outer:
        assert outer.acquired is True
        with PidLock(p, expected_name=None) as inner:
            assert inner.acquired is False
    assert not p.exists()  # outer released on exit
