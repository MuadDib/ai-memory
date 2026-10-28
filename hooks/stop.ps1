# ai-memory Stop hook
# Fires after every Claude Code response. Runs import-cowork incrementally so
# Tier 3 (raw JSONL) stays current throughout the session.
#
# Deliberately writes nothing to stdout — Claude Code feeds hook stdout back
# to the model as a system message, and we don't want import noise there.
# All output goes to the ai-memory log file instead.

# Data lives outside AppData: the Claude desktop app is an MSIX package and
# redirects new AppData folders into its private LocalCache.
$env:AI_MEMORY_HOME = "C:\ai-mem\data"
$ErrorActionPreference = "SilentlyContinue"

$aiMemory = "C:\ai-mem\ai-memory\.venv\Scripts\ai-memory.exe"
$logFile  = "$env:AI_MEMORY_HOME\logs\hook-stop.log"

try {
    $claudeRoot = "$env:USERPROFILE\.claude\projects"
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$ts] stop: import-cowork --root $claudeRoot" | Add-Content -Path $logFile -Encoding UTF8
    # SilentlyContinue drops native stderr before 2>&1 can capture it, which
    # hid every traceback — use Continue for the call so failures reach the log.
    $ErrorActionPreference = "Continue"
    & $aiMemory import-cowork --root $claudeRoot 2>&1 | ForEach-Object { "$_" } | Add-Content -Path $logFile -Encoding UTF8
    "[$ts] stop: import-cowork exit=$LASTEXITCODE" | Add-Content -Path $logFile -Encoding UTF8
} catch {
    # Never surface errors to Claude Code — memory capture is best-effort
}
