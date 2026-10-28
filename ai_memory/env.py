"""
Environment-variable lookup with a Windows user-scope fallback.

MCP clients (Claude Desktop included) launch stdio servers with a filtered
environment — only an allowlist like PATH/USERPROFILE/APPDATA is inherited —
so a key set as a User environment variable never reaches `ai-memory serve`.
Rather than force the key into a plaintext `env` block in the client config,
fall back to reading the User-scope variable from the registry.

C# port: `Environment.GetEnvironmentVariable(name)` falling back to
`Environment.GetEnvironmentVariable(name, EnvironmentVariableTarget.User)`.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Callable


def _read_user_env_windows(name: str) -> str | None:
    if sys.platform != "win32":
        return None
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, value_type = winreg.QueryValueEx(key, name)
    except OSError:
        return None
    if value_type == winreg.REG_EXPAND_SZ:
        value = os.path.expandvars(value)
    return value if isinstance(value, str) else None


def get_env(
    name: str,
    *,
    user_lookup: Callable[[str], str | None] = _read_user_env_windows,
) -> str | None:
    """Return `name` from the process env, else from the user-scope env (Windows)."""
    return os.environ.get(name) or user_lookup(name) or None
