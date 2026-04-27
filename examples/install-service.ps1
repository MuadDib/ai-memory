# install-service.ps1 — install the dream daemon as a Windows service.
#
# Wraps `ai-memory dream --watch` as a Windows service using NSSM
# (https://nssm.cc), set to Automatic (Delayed Start) so it doesn't slow
# the initial boot. The daemon is single-user and reads the same
# %LOCALAPPDATA%\ai-memory\ files the MCP server uses.
#
# Run from an *elevated* PowerShell (right-click -> Run as Administrator).
#
# Prerequisites:
#   - Real Python 3.11+ on PATH (see setup.ps1)
#   - The repo's venv exists at <repo>\.venv\ (created by setup.ps1)
#   - NSSM installed (`winget install NSSM.NSSM` or download from nssm.cc)
#
# What it does:
#   1. Locates nssm.exe (PATH or default winget install location).
#   2. Installs a service called `AiMemoryDream` that runs the venv's
#      ai-memory.exe with `dream --watch`.
#   3. Sets startup type to Automatic-Delayed-Start.
#   4. Configures stdout/stderr capture, restart-on-failure, and the
#      ANTHROPIC_API_KEY / OPENAI_API_KEY env vars from the current
#      session (so launch the elevated shell with the keys exported).
#   5. Starts the service.
#
# To uninstall, run install-service.ps1 -Uninstall.

param(
    [switch]$Uninstall,
    [string]$ServiceName = "AiMemoryDream"
)

$ErrorActionPreference = "Stop"

# --- 0. Elevation ----------------------------------------------------------

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "This script must be run from an elevated PowerShell. Right-click -> Run as Administrator."
    exit 1
}

# --- 1. Locate NSSM --------------------------------------------------------

function Find-Nssm {
    $cmd = Get-Command nssm -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Path }
    foreach ($candidate in @(
        "$env:ProgramData\chocolatey\bin\nssm.exe",
        "$env:ProgramFiles\nssm\nssm.exe",
        "$env:ProgramFiles(x86)\nssm\nssm.exe",
        "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\NSSM.NSSM_*\nssm.exe"
    )) {
        $hits = Get-ChildItem -Path $candidate -ErrorAction SilentlyContinue
        if ($hits) { return $hits[0].FullName }
    }
    return $null
}

$nssm = Find-Nssm
if ($null -eq $nssm) {
    Write-Error @"
NSSM not found.

Install one of:
  winget install NSSM.NSSM        (recommended)
  choco install nssm
  https://nssm.cc/download         (manual)

Then re-run this script.
"@
    exit 1
}
Write-Host "NSSM: $nssm" -ForegroundColor Green

# --- 2. Uninstall path -----------------------------------------------------

if ($Uninstall) {
    try { & $nssm stop $ServiceName 2>$null | Out-Null } catch {}
    $global:LASTEXITCODE = 0
    try { & $nssm remove $ServiceName confirm 2>$null | Out-Null } catch {}
    $global:LASTEXITCODE = 0
    Write-Host "Service $ServiceName removed." -ForegroundColor Green
    exit 0
}

# --- 3. Resolve the venv's ai-memory.exe -----------------------------------

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoRoot = Split-Path -Parent $scriptRoot
$venvExe = Join-Path $repoRoot ".venv\Scripts\ai-memory.exe"

if (-not (Test-Path $venvExe)) {
    Write-Error @"
Could not find the ai-memory entry point at:
  $venvExe

Run setup.ps1 first to create the venv and install the package.
"@
    exit 1
}
Write-Host "ai-memory exe: $venvExe" -ForegroundColor Green

# --- 4. Resolve the home directory the daemon should use -------------------

$memHome = $env:AI_MEMORY_HOME
if ([string]::IsNullOrWhiteSpace($memHome)) {
    $memHome = Join-Path $env:LOCALAPPDATA "ai-memory"
}
New-Item -ItemType Directory -Force -Path $memHome | Out-Null
Write-Host "Daemon home: $memHome" -ForegroundColor Green

# --- 5. Pull API keys from the current shell -------------------------------

$openaiKey = $env:OPENAI_API_KEY
$anthropicKey = $env:ANTHROPIC_API_KEY
if ([string]::IsNullOrWhiteSpace($anthropicKey)) {
    Write-Warning "ANTHROPIC_API_KEY is not set in this session. The dream daemon needs it; service will start but dream calls will fail until the key is configured. Re-run after setting it, or edit the service env vars later via 'nssm edit AiMemoryDream'."
}

# --- 6. (Re)install the service --------------------------------------------

# Remove any prior install so we always end up with a clean config.
# These are best-effort -- ignore failures when the service doesn't exist yet.
try { & $nssm stop $ServiceName 2>$null | Out-Null } catch {}
$global:LASTEXITCODE = 0
try { & $nssm remove $ServiceName confirm 2>$null | Out-Null } catch {}
$global:LASTEXITCODE = 0

& $nssm install $ServiceName $venvExe "dream" "--watch" | Out-Null
& $nssm set $ServiceName AppDirectory $repoRoot | Out-Null
& $nssm set $ServiceName Description "ai-memory dream daemon: scheduled / idle / pressure consolidation" | Out-Null
& $nssm set $ServiceName Start SERVICE_DELAYED_AUTO_START | Out-Null
& $nssm set $ServiceName ObjectName "LocalSystem" | Out-Null

# Logs
$logDir = Join-Path $memHome "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
& $nssm set $ServiceName AppStdout (Join-Path $logDir "service-stdout.log") | Out-Null
& $nssm set $ServiceName AppStderr (Join-Path $logDir "service-stderr.log") | Out-Null
& $nssm set $ServiceName AppRotateFiles 1 | Out-Null
& $nssm set $ServiceName AppRotateBytes 10485760 | Out-Null  # 10 MiB

# Restart on crash
& $nssm set $ServiceName AppExit Default Restart | Out-Null
& $nssm set $ServiceName AppRestartDelay 5000 | Out-Null

# Environment
$envLines = @(
    "AI_MEMORY_HOME=$memHome"
)
if ($openaiKey)    { $envLines += "OPENAI_API_KEY=$openaiKey" }
if ($anthropicKey) { $envLines += "ANTHROPIC_API_KEY=$anthropicKey" }
& $nssm set $ServiceName AppEnvironmentExtra ($envLines -join "`r`n") | Out-Null

# --- 7. Start --------------------------------------------------------------

& $nssm start $ServiceName | Out-Null
Start-Sleep -Seconds 2
& $nssm status $ServiceName

Write-Host ""
Write-Host "Service '$ServiceName' installed (Automatic - Delayed Start)." -ForegroundColor Green
Write-Host "Logs: $logDir"
Write-Host ""
Write-Host "Useful commands:"
Write-Host "  Stop      : nssm stop $ServiceName"
Write-Host "  Start     : nssm start $ServiceName"
Write-Host "  Edit env  : nssm edit $ServiceName"
Write-Host "  Uninstall : .\install-service.ps1 -Uninstall"
