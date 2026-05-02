# ai-memory Stop hook
# Fires after every Claude Code response. Runs import-cowork incrementally so
# Tier 3 (raw JSONL) stays current throughout the session.
#
# Deliberately writes nothing to stdout — Claude Code feeds hook stdout back
# to the model as a system message, and we don't want import noise there.
# All output goes to the ai-memory log file instead.

$ErrorActionPreference = "SilentlyContinue"

$aiMemory = "C:\ai-mem\ai-memory\.venv\Scripts\ai-memory.exe"
$logFile  = "$env:LOCALAPPDATA\ai-memory\logs\hook-stop.log"

try {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$ts] stop: import-cowork" | Add-Content -Path $logFile -Encoding UTF8
    & $aiMemory import-cowork 2>&1 | Add-Content -Path $logFile -Encoding UTF8
} catch {
    # Never surface errors to Claude Code — memory capture is best-effort
}
