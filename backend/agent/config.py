"""Central configuration — env, models, mock detection.

Public repo: NEVER hardcode or log keys. All secrets come from .env / env vars.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Load .env from repo root if python-dotenv is available (optional dep).
try:
    from dotenv import load_dotenv

    _here = Path(__file__).resolve()
    for parent in _here.parents:
        candidate = parent / ".env"
        if candidate.exists():
            load_dotenv(candidate)
            break
except ImportError:  # pragma: no cover
    pass

# Crusoe Managed Inference — OpenAI-compatible (source: docs/CRUSOE.md)
DEFAULT_BASE_URL = "https://api.inference.crusoecloud.com/v1/"

# Exact free-tier model strings for the RAISE hackathon (docs/CRUSOE.md §3).
MODEL_FAST_DEFAULT = "deepseek-ai/Deepseek-V4-Flash"          # ~5 s loop, tier-2 classify, skeptic
MODEL_REASONING_DEFAULT = "nvidia/NVIDIA-Nemotron-3-Ultra-550B"  # advisories, advocate, jury, boss
MODEL_OMNI_DEFAULT = "nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B"  # image/audio/video if needed


def thinking_off_extra_body(model: str) -> dict:
    """Reasoning models emit <think> tokens that break JSON parsing.

    Per docs/CRUSOE.md: Nemotron family uses enable_thinking, Kimi/DeepSeek use
    thinking, Gemma needs nothing.
    """
    m = model.lower()
    if "nemotron" in m:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    if "kimi" in m or "deepseek" in m:
        return {"chat_template_kwargs": {"thinking": False}}
    return {}


@dataclass
class Settings:
    api_key: str = field(default_factory=lambda: os.environ.get("CRUSOE_API_KEY", ""))
    base_url: str = field(
        default_factory=lambda: os.environ.get("CRUSOE_BASE_URL", DEFAULT_BASE_URL)
    )
    model_fast: str = field(
        default_factory=lambda: os.environ.get("CRUSOE_MODEL_FAST", MODEL_FAST_DEFAULT)
    )
    model_reasoning: str = field(
        default_factory=lambda: os.environ.get(
            "CRUSOE_MODEL_REASONING", MODEL_REASONING_DEFAULT
        )
    )
    model_omni: str = field(
        default_factory=lambda: os.environ.get("CRUSOE_MODEL_OMNI", MODEL_OMNI_DEFAULT)
    )
    hmac_secret: str = field(
        default_factory=lambda: os.environ.get("PINN_HMAC_SECRET", "")
    )
    backend_port: int = field(
        default_factory=lambda: int(os.environ.get("BACKEND_PORT", "8000"))
    )
    db_path: str = field(
        default_factory=lambda: os.environ.get(
            "AGENT_DB_PATH", str(Path(__file__).resolve().parent / "praetor_state.db")
        )
    )
    # Jury gate: advisories scoring below this overall are flagged, not shown as-is.
    jury_pass_threshold: float = field(
        default_factory=lambda: float(os.environ.get("JURY_PASS_THRESHOLD", "3.0"))
    )

    @property
    def mock_mode(self) -> bool:
        """MOCK_LLM=1 forces mock; otherwise mock iff no API key present."""
        forced = os.environ.get("MOCK_LLM", "").strip()
        if forced == "1":
            return True
        if forced == "0":
            return False
        return not self.api_key


settings = Settings()
