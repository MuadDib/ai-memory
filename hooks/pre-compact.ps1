# ai-memory PreCompact hook
# Fires when Claude Code is about to compact the conversation context.
# A compaction event means the session has grown large — exactly when
# consolidation is most valuable.
#
# 1. import-cowork (sync) — flush all turns to Tier 3 before the transcript
#    gets summarised by compaction. Do this first so dream has a complete
#    picture even if it starts while compaction is still in progress.
#
# 2. dream (async, detached) — spawn as a fully independent process so this
#    hook returns immediately and does not block compaction. Output goes to
#    a dedicated log file.

$ErrorActionPreference = "SilentlyContinue"

$aiMemory  = "C:\ai-mem\ai-memory\.venv\Scripts\ai-memory.exe"
$logFile   = "$env:LOCALAPPDATA\ai-memory\logs\hook-precompact.log"
$dreamLog  = "$env:LOCALAPPDATA\ai-memory\logs\dream-async.log"

try {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$ts] pre-compact: import-cowork" | Add-Content -Path $logFile -Encoding UTF8
    & $aiMemory import-cowork 2>&1 | Add-Content -Path $logFile -Encoding UTF8

    # Spawn dream as a fully detached process — no parent/child relationship,
    # returns before dream finishes, does not block compaction.
    "[$ts] pre-compact: spawning dream async (log: $dreamLog)" |
        Add-Content -Path $logFile -Encoding UTF8

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName        = $aiMemory
    $psi.Arguments       = "dream --trigger idle"
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow  = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true

    $proc = [System.Diagnostics.Process]::Start($psi)
    # Don't wait — fire and forget. Dream runs independently.
    $proc | Out-Null
} catch {
    # Never surface errors to Claude Code
}
