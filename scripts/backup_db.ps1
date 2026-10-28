$home_ = if ($env:AI_MEMORY_HOME) { $env:AI_MEMORY_HOME } else { "C:\ai-mem\data" }
$src = "$home_\memory.db"
$ts  = Get-Date -Format "yyyyMMdd-HHmmss"
$dst = "$home_\memory.db.bak-$ts"
Copy-Item $src $dst
"{0}  ({1:N1} MB)" -f $dst, ((Get-Item $dst).Length / 1MB)
