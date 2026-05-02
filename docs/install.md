# Installing ai-memory on a new machine

Instructions for Claude Code — paste into a fresh session to install and fully wire ai-memory.
Follow every step in order; stop and report if anything fails.

---

## Prerequisites — check first

```powershell
# Confirm Python 3.11 or 3.12 is available
py -3.11 --version   # or py -3.12 --version
git --version
```

If Python is missing, download from python.org (3.11 or 3.12). Git must also be installed.

---

## Step 1 — Clone to a short path

> **Windows MAX_PATH:** Deep paths break Python venvs. Always clone to `C:\ai-mem\ai-memory`.

**Windows:**
```powershell
git clone https://github.com/MuadDib/ai-memory.git C:\ai-mem\ai-memory
cd C:\ai-mem\ai-memory
```

**Linux/macOS:**
```bash
git clone https://github.com/MuadDib/ai-memory.git ~/ai-memory
cd ~/ai-memory
```

---

## Step 2 — Create venv and install

**Windows:**
```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser   # one-time; needed for venv activation
py -3.11 -m venv .venv                                # use py -3.12 if that's what you have
.\.venv\Scripts\Activate.ps1
pip install -e .
```

**Linux/macOS:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Step 3 — Set API keys

ai-memory needs an OpenAI API key for embeddings (`text-embedding-3-small`) and LLM dream cycles
(`gpt-4o-mini`). An Anthropic key is optional (only needed if you switch `llm.provider` to
`anthropic` in config.yaml).

**Windows — persist for all future sessions:**
```powershell
[System.Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "sk-YOUR-KEY-HERE", "User")
```

**Linux/macOS — add to `~/.bashrc` or `~/.zshrc`:**
```bash
export OPENAI_API_KEY="sk-YOUR-KEY-HERE"
```

Then reload: `source ~/.bashrc`

---

## Step 4 — Verify install

```powershell
# Windows (venv active)
ai-memory stats
```
```bash
# Linux/macOS (venv active)
ai-memory stats
```

Expected output: a table showing zeroes (empty corpus). Any other output means something went
wrong — stop and report.

---

## Step 5 — Wire the MCP server into Claude Code

Create or edit `~/.claude/mcp.json` (global, so it works in every project).

**Windows** — replace `OPENAI_API_KEY_HERE` with your actual key:
```json
{
  "mcpServers": {
    "ai-memory": {
      "command": "C:\\ai-mem\\ai-memory\\.venv\\Scripts\\ai-memory.exe",
      "args": ["serve"],
      "env": {
        "OPENAI_API_KEY": "OPENAI_API_KEY_HERE"
      }
    }
  }
}
```

**Linux/macOS:**
```json
{
  "mcpServers": {
    "ai-memory": {
      "command": "/home/YOUR_USERNAME/ai-memory/.venv/bin/ai-memory",
      "args": ["serve"],
      "env": {
        "OPENAI_API_KEY": "OPENAI_API_KEY_HERE"
      }
    }
  }
}
```

---

## Step 6 — Wire Stop and PreCompact hooks into Claude Code

These hooks keep ai-memory in sync after every response and before every context compaction.

Edit `~/.claude/settings.json`. Add the `hooks` block — merge with any existing content,
don't replace the whole file.

**Windows:**
```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "C:\\ai-mem\\ai-memory\\hooks\\stop.ps1",
            "shell": "powershell"
          }
        ]
      }
    ],
    "PreCompact": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "C:\\ai-mem\\ai-memory\\hooks\\pre-compact.ps1",
            "shell": "powershell"
          }
        ]
      }
    ]
  }
}
```

**Linux/macOS:**
```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/home/YOUR_USERNAME/ai-memory/hooks/stop.sh"
          }
        ]
      }
    ],
    "PreCompact": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/home/YOUR_USERNAME/ai-memory/hooks/pre-compact.sh"
          }
        ]
      }
    ]
  }
}
```

> **Windows critical:** The `"shell": "powershell"` field is required. Using `powershell.exe`
> as the command directly does not work — Claude Code ignores the hook silently with no error.

---

## Step 6b — Create shell hooks (Linux/macOS only)

The repo ships `.ps1` hooks for Windows. Create the bash equivalents:

**`hooks/stop.sh`:**
```bash
#!/usr/bin/env bash
AI_MEMORY="$HOME/ai-memory/.venv/bin/ai-memory"
LOG="$HOME/.local/share/ai-memory/logs/hook-stop.log"
mkdir -p "$(dirname "$LOG")"
TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$TS] stop: import-cowork --root $HOME/.claude/projects" >> "$LOG"
"$AI_MEMORY" import-cowork --root "$HOME/.claude/projects" >> "$LOG" 2>&1
```

**`hooks/pre-compact.sh`:**
```bash
#!/usr/bin/env bash
AI_MEMORY="$HOME/ai-memory/.venv/bin/ai-memory"
LOG="$HOME/.local/share/ai-memory/logs/hook-precompact.log"
mkdir -p "$(dirname "$LOG")"
TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$TS] pre-compact: import-cowork --root $HOME/.claude/projects" >> "$LOG"
"$AI_MEMORY" import-cowork --root "$HOME/.claude/projects" >> "$LOG" 2>&1
"$AI_MEMORY" dream --trigger idle >> "$HOME/.local/share/ai-memory/logs/dream-async.log" 2>&1 &
disown
```

```bash
chmod +x hooks/stop.sh hooks/pre-compact.sh
```

---

## Step 7 — Create the logs directory

**Windows:**
```powershell
New-Item -ItemType Directory -Force "$env:LOCALAPPDATA\ai-memory\logs"
```

**Linux/macOS:**
```bash
mkdir -p ~/.local/share/ai-memory/logs
```

---

## Step 8 — Restart Claude Code

Restart Claude Code now so it picks up:
- The MCP server from `mcp.json`
- The Stop and PreCompact hooks from `settings.json`

After restart, verify the MCP server is active: you should see `memory_recall`, `memory_remember`,
etc. in the available tools.

---

## Step 9 — Import existing Claude Code sessions (optional)

If this machine already has Claude Code session history you want in memory:

```powershell
# Windows
ai-memory import-cowork
```
```bash
# Linux/macOS
ai-memory import-cowork
```

This walks `~/.claude/projects/` and imports all past sessions incrementally. Safe to re-run.

---

## Step 10 — Run a first dream cycle (optional, recommended after import)

```powershell
ai-memory dream --trigger manual
```

This consolidates imported sessions into searchable notes. Takes a few minutes for large corpora;
sessions with 100+ turns automatically use `gpt-4o` for better quality.

---

## Verify everything is working

```powershell
ai-memory stats          # should show non-zero turns/episodes after import+dream
ai-memory recall "test"  # should return results if you have notes
```

After a Claude Code restart, send a short message and check the hook log:

```powershell
# Windows
Get-Content "$env:LOCALAPPDATA\ai-memory\logs\hook-stop.log" -Tail 5
```
```bash
# Linux/macOS
tail -5 ~/.local/share/ai-memory/logs/hook-stop.log
```

You should see a timestamped `stop: import-cowork` entry — that confirms the hook is firing
automatically after every response.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ai-memory stats` fails with import error | Run `pip install -e .` again inside the venv |
| MCP tools not visible in Claude Code | Check `~/.claude/mcp.json` syntax; restart Claude Code |
| Hook not firing on Windows | Ensure `"shell": "powershell"` is present; restart Claude Code after editing `settings.json` |
| `sessions seen=0` on import-cowork | Pass `--root "$env:USERPROFILE\.claude\projects"` explicitly |
| `dream` fails with API error | Check `OPENAI_API_KEY` is set in the env block in `mcp.json` |
