$home_ = if ($env:AI_MEMORY_HOME) { $env:AI_MEMORY_HOME } else { "C:\ai-mem\data" }
# Online backup through the SQLite API: a plain Copy-Item of memory.db misses
# everything still in memory.db-wal. See backup_db.py.
& "$PSScriptRoot\..\.venv\Scripts\python.exe" "$PSScriptRoot\backup_db.py" $home_
exit $LASTEXITCODE
