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

# Data lives outside AppData: the Claude desktop app is an MSIX package and
# redirects new AppData folders into its private LocalCache.
$env:AI_MEMORY_HOME = "C:\ai-mem\data"
$ErrorActionPreference = "SilentlyContinue"

$aiMemory  = "C:\ai-mem\ai-memory\.venv\Scripts\ai-memory.exe"
$logFile   = "$env:AI_MEMORY_HOME\logs\hook-precompact.log"
$dreamLog  = "$env:AI_MEMORY_HOME\logs\dream-async.log"

try {
    $claudeRoot = "$env:USERPROFILE\.claude\projects"
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$ts] pre-compact: import-cowork --root $claudeRoot" | Add-Content -Path $logFile -Encoding UTF8
    # SilentlyContinue drops native stderr before 2>&1 can capture it, which
    # hid every traceback — use Continue for the call so failures reach the log.
    $ErrorActionPreference = "Continue"
    & $aiMemory import-cowork --root $claudeRoot 2>&1 | ForEach-Object { "$_" } | Add-Content -Path $logFile -Encoding UTF8
    "[$ts] pre-compact: import-cowork exit=$LASTEXITCODE" | Add-Content -Path $logFile -Encoding UTF8
    $ErrorActionPreference = "SilentlyContinue"

    # Spawn dream as a fully detached process — no parent/child relationship,
    # returns before dream finishes, does not block compaction.
    "[$ts] pre-compact: spawning dream async (log: $dreamLog)" |
        Add-Content -Path $logFile -Encoding UTF8

    # Redirect via cmd.exe to the log file: redirecting to .NET pipes that
    # nobody reads loses the output and can stall or kill dream once the
    # pipe fills or this hook exits.
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName        = "cmd.exe"
    $psi.Arguments       = "/c `"`"$aiMemory`" dream --trigger idle >> `"$dreamLog`" 2>&1`""
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow  = $true

    $proc = [System.Diagnostics.Process]::Start($psi)
    # Don't wait — fire and forget. Dream runs independently.
    $proc | Out-Null
} catch {
    # Never surface errors to Claude Code
}
