# install-dream-task.ps1 — run the dream daemon as a per-user logon task.
#
# Alternative to install-service.ps1 that needs no admin rights and no NSSM.
# Registers a Task Scheduler task that starts `ai-memory dream --watch` at
# logon, as the current user, with no console window (pythonw.exe +
# dream_daemon.pyw). Because it runs as you, OPENAI_API_KEY is read from your
# User environment (see ai_memory/env.py) — no key is copied into the task.
#
# Trade-off vs the service: the daemon only runs while you are logged in.
#
#   .\install-dream-task.ps1              install (or reinstall) and start now
#   .\install-dream-task.ps1 -Uninstall   stop and remove

param(
    [switch]$Uninstall,
    [string]$TaskName = "AiMemoryDream",
    [string]$MemHome = $(if ($env:AI_MEMORY_HOME) { $env:AI_MEMORY_HOME } else { "C:\ai-mem\data" })
)

$ErrorActionPreference = "Stop"

# Always start from a clean registration.
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}
if ($Uninstall) {
    Write-Host "Task '$TaskName' removed." -ForegroundColor Green
    exit 0
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonw  = Join-Path $repoRoot ".venv\Scripts\pythonw.exe"
$launcher = Join-Path $PSScriptRoot "dream_daemon.pyw"
foreach ($p in $pythonw, $launcher) {
    if (-not (Test-Path $p)) { throw "Missing: $p" }
}

$user      = "$env:USERDOMAIN\$env:USERNAME"
$action    = New-ScheduledTaskAction -Execute $pythonw -Argument "`"$launcher`" `"$MemHome`"" -WorkingDirectory $repoRoot
$trigger   = New-ScheduledTaskTrigger -AtLogOn -User $user
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
# ExecutionTimeLimit 0 = unlimited; the default (72h) would kill the daemon.
$settings  = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings `
    -Description "ai-memory dream daemon: scheduled / idle / pressure consolidation (home: $MemHome)" | Out-Null

Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 5
$info = Get-ScheduledTaskInfo -TaskName $TaskName
"Task '{0}': state={1} lastResult=0x{2:X}" -f $TaskName, (Get-ScheduledTask -TaskName $TaskName).State, $info.LastTaskResult
"Logs: $MemHome\logs\daemon-*.jsonl, $MemHome\logs\daemon-stdio.log"
