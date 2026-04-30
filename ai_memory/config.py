"""
Configuration loading and validation.

Layered resolution (highest precedence first):
    1. Environment variables (AI_MEMORY_*)
    2. config.yaml in the home directory
    3. Built-in defaults

The home directory is `%LOCALAPPDATA%\\ai-memory\\` on Windows by default
(via platformdirs). Override with the `AI_MEMORY_HOME` environment variable.

`environment` is a mandatory parameter — every flow gets `dev | local | prod`
in its config explicitly. Defaulting to "local" rather than guessing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from platformdirs import user_data_path

Environment = Literal["dev", "local", "prod"]


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for the embedding provider."""

    provider: Literal["openai", "voyage", "local"] = "openai"
    model: str = "text-embedding-3-small"
    dimensions: int = 1536  # text-embedding-3-small native; can truncate via Matryoshka


@dataclass(frozen=True)
class LlmConfig:
    """Configuration for the LLM used by consolidation / dreaming."""

    provider: Literal["anthropic", "openai"] = "anthropic"
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 4096


@dataclass(frozen=True)
class DreamConfig:
    """Configuration for the dream-cycle consolidation worker."""

    schedule_cron: str = "0 3 * * *"  # daily at 03:00 local
    idle_trigger_minutes: int = 20
    pressure_trigger_turns: int = 200
    promotion_min_endorsing_notes: int = 3
    promotion_min_episodes: int = 2
    decay_half_life_days: int = 90
    prune_threshold: float = 0.05


@dataclass(frozen=True)
class RecallConfig:
    """Configuration for the recall pipeline."""

    rrf_k: int = 60
    default_k: int = 8
    deep_k: int = 20
    vector_candidates: int = 50
    bm25_candidates: int = 50
    recency_half_life_days: int = 90

    # Relevance floor for vector neighbours. With text-embedding-3-small,
    # observed L2 distances on this corpus run roughly:
    #   ~0.6  = strong match (same topic, same phrasing)
    #   ~0.9  = related (same topic, different angle)
    #   ~1.1  = borderline (drifting off-topic)
    #   >1.2  = effectively unrelated noise
    # We drop hits above this threshold *before* RRF so the agent doesn't
    # see confidently-ranked nonsense. BM25 hits are not filtered — token
    # matches are a deterministic relevance signal.
    # Tune via `recall.vector_distance_floor` in config.yaml. The recall
    # pipeline logs candidate distance stats so you can see the live
    # distribution before tightening.
    vector_distance_floor: float = 1.1
    final_score_floor: float = 0.0  # post-RRF/recency cutoff (0 = keep everything fused)


@dataclass(frozen=True)
class Config:
    """Top-level service configuration."""

    environment: Environment = "local"
    home: Path = field(default_factory=lambda: _default_home())
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    dream: DreamConfig = field(default_factory=DreamConfig)
    recall: RecallConfig = field(default_factory=RecallConfig)
    log_level: str = "INFO"

    @property
    def database_path(self) -> Path:
        """Absolute path to the SQLite database file."""
        return self.home / "memory.db"

    @property
    def profile_md_path(self) -> Path:
        """Absolute path to the human-readable profile mirror."""
        return self.home / "profile.md"

    @property
    def raw_dir(self) -> Path:
        """Root directory for raw transcript JSONL files."""
        return self.home / "raw"

    @property
    def exports_dir(self) -> Path:
        """Directory holding bootstrap memory dumps."""
        return self.home / "exports"

    @property
    def logs_dir(self) -> Path:
        """Operational log directory."""
        return self.home / "logs"

    @property
    def config_yaml_path(self) -> Path:
        """Optional yaml-on-disk override file."""
        return self.home / "config.yaml"


def _default_home() -> Path:
    """Resolve the default ai-memory home directory.

    On Windows: %LOCALAPPDATA%\\ai-memory\\
    On macOS:   ~/Library/Application Support/ai-memory/
    On Linux:   $XDG_DATA_HOME/ai-memory/  (fallback ~/.local/share/ai-memory/)
    """
    return user_data_path("ai-memory", appauthor=False, ensure_exists=False, roaming=False)


def load_config(home_override: Path | str | None = None) -> Config:
    """Load configuration from yaml + environment variables.

    The home directory is resolved as:
        1. `home_override` argument if provided
        2. `AI_MEMORY_HOME` env var if set
        3. The platform-default app data directory

    All other settings come from `<home>/config.yaml` if present, with
    `AI_MEMORY_*` environment variables overriding individual keys.
    """
    # Resolve home
    if home_override is not None:
        home = Path(home_override).expanduser().resolve()
    elif env_home := os.environ.get("AI_MEMORY_HOME"):
        home = Path(env_home).expanduser().resolve()
    else:
        home = _default_home()

    # Read yaml if present
    yaml_data: dict = {}
    yaml_path = home / "config.yaml"
    if yaml_path.exists():
        with yaml_path.open("r", encoding="utf-8") as fh:
            yaml_data = yaml.safe_load(fh) or {}

    # Environment overrides (only the ones we care about for now)
    environment: Environment = (
        os.environ.get("AI_MEMORY_ENV") or yaml_data.get("environment") or "local"
    )  # type: ignore[assignment]
    log_level = os.environ.get("AI_MEMORY_LOG_LEVEL") or yaml_data.get("log_level") or "INFO"

    embeddings = EmbeddingConfig(**(yaml_data.get("embeddings") or {}))
    llm = LlmConfig(**(yaml_data.get("llm") or {}))
    dream = DreamConfig(**(yaml_data.get("dream") or {}))
    recall = RecallConfig(**(yaml_data.get("recall") or {}))

    return Config(
        environment=environment,
        home=home,
        embeddings=embeddings,
        llm=llm,
        dream=dream,
        recall=recall,
        log_level=log_level,
    )


def ensure_home_layout(config: Config) -> None:
    """Create all required subdirectories under the home dir."""
    for directory in (config.home, config.raw_dir, config.exports_dir, config.logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
