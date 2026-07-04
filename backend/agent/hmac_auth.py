"""Origin authentication of PINN payloads (architecture doc §3).

HMAC-SHA256 over a canonical JSON serialization. Verified in the edge agent
before ANY processing. Standard library only.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

from .config import settings

# Demo fallback so the repo runs out of the box. NOT safe for real deployments —
# provision PINN_HMAC_SECRET in .env.
_DEMO_SECRET = "praetor-demo-secret-do-not-use-in-prod"


def _secret() -> bytes:
    return (settings.hmac_secret or _DEMO_SECRET).encode("utf-8")


def canonical(payload: dict[str, Any]) -> bytes:
    """Stable serialization: sorted keys, no whitespace drift."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_payload(payload: dict[str, Any]) -> str:
    return hmac.new(_secret(), canonical(payload), hashlib.sha256).hexdigest()


def verify_payload(payload: dict[str, Any], signature: str) -> bool:
    expected = sign_payload(payload)
    return hmac.compare_digest(expected, signature)
