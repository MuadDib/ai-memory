$src = "C:\Users\Igor\AppData\Local\ai-memory\memory.db"
$ts  = Get-Date -Format "yyyyMMdd-HHmmss"
$dst = "C:\Users\Igor\AppData\Local\ai-memory\memory.db.bak-$ts"
Copy-Item $src $dst
"{0}  ({1:N1} MB)" -f $dst, ((Get-Item $dst).Length / 1MB)
