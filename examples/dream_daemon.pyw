"""
Console-less launcher for the dream daemon (Task Scheduler via pythonw.exe).

pythonw has no console, so sys.stdout / sys.stderr are None and anything that
prints or logs to them is lost or crashes. Point both at a log file under the
data home, then hand off to the normal CLI: `ai-memory --home <home> dream --watch`.

Usage: pythonw.exe dream_daemon.pyw <home>
"""
import sys
from pathlib import Path

home = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\ai-mem\data")
(home / "logs").mkdir(parents=True, exist_ok=True)
# Held open for the life of the process, so no context manager.
sys.stdout = sys.stderr = open(home / "logs" / "daemon-stdio.log", "a", encoding="utf-8", buffering=1)  # noqa: SIM115

from ai_memory.cli import main  # noqa: E402  (after the stdio redirect, on purpose)

main(["--home", str(home), "dream", "--watch"], prog_name="ai-memory")
