"""Signed telemetry payloads for the information layer (teammate's part).

DRAFT SCHEMA — the exact schema must be locked with the information-layer
owner before finalizing; minimum agreed fields per the team brief:

    {machine_id, head_outputs: {...}, timestamp, hmac_signature}

We add a `provenance` block so downstream consumers can never mistake
synthetic/simulated stand-in data for measured telemetry (hard project rule).

stdlib only (hmac + hashlib + json) — matches the information layer's
stdlib-only constraint.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone

SCHEMA_VERSION = "0.1-draft"

# Provenance labels per head — keep in sync with pinn/data/__init__.py.
HEAD_PROVENANCE = {
    "vibration": "REAL (CWRU / IMS public measurements)",
    "thermal_power": "SIMULATED (AI4I 2020, documented generative rules)",
    "degradation_rul": "SIMULATED (NASA C-MAPSS FD001)",
    "pressure": "SYNTHETIC (generated in-repo, no public dataset exists)",
}


def _canonical(payload: dict) -> bytes:
    """Deterministic serialization: sorted keys, no whitespace drift."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_payload(machine_id: str, head_outputs: dict, key: bytes,
                  timestamp: str | None = None) -> dict:
    """Assemble and sign one telemetry message (HMAC-SHA256 over the body)."""
    body = {
        "schema_version": SCHEMA_VERSION,
        "machine_id": machine_id,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "head_outputs": head_outputs,
        "provenance": {k: HEAD_PROVENANCE[k] for k in head_outputs if k in HEAD_PROVENANCE},
    }
    signature = hmac.new(key, _canonical(body), hashlib.sha256).hexdigest()
    return {**body, "hmac_signature": signature}


def verify_payload(payload: dict, key: bytes) -> bool:
    body = {k: v for k, v in payload.items() if k != "hmac_signature"}
    expected = hmac.new(key, _canonical(body), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, payload.get("hmac_signature", ""))
