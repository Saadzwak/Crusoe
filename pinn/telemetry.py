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


# --------------------------------------------------------------------------- #
# PRAETOR-compatible emission — matches the information layer's LANDED contract
# (origin/feat/llm-information-layer: backend/agent/schemas.py + hmac_auth.py).
# Their canonicalization is byte-identical to ours (sorted keys, no whitespace);
# the difference is shape: signature travels BESIDE the payload
# (SignedReading{payload, signature}), not inside it.
# --------------------------------------------------------------------------- #

import os

# Teammate's demo fallback (backend/agent/hmac_auth.py). Real deployments set
# PINN_HMAC_SECRET in the environment on both sides.
_PRAETOR_DEMO_SECRET = "praetor-demo-secret-do-not-use-in-prod"


def _praetor_secret() -> bytes:
    # `or` (not a get() default): .env.example ships `PINN_HMAC_SECRET=` which
    # dotenv exports as an EMPTY STRING — get()'s default would then sign with
    # b"" while the teammate's `settings.hmac_secret or _DEMO_SECRET` falls
    # back to the demo secret, silently breaking every signature. Caught by
    # scripts/test_cross_layer_contract.py (empty-env case).
    return (os.environ.get("PINN_HMAC_SECRET") or _PRAETOR_DEMO_SECRET).encode("utf-8")


def to_praetor_signed_reading(
    machine_id: str,
    department: str,
    epoch: int,
    signals: dict[str, float],
    health_index: float,
    rul_cycles: float | None,
    residual: float,
    failure_mode_probs: dict[str, float],
    note: str = "",
    timestamp: float | None = None,
) -> dict:
    """Emit one SignedReading exactly as backend/agent expects.

    Field semantics pinned by their schemas.py:
    - health_index: 1.0 = HEALTHY. Our IMS lifetime-position proxy and the
      fatigue head's damage D run the other way (1.0 = at failure) — callers
      pass `1 - damage`, never the raw proxy. Inversion bugs here would
      silently flip every triage decision downstream.
    - residual: physics-consistency residual, higher = worse — we emit the
      relevant head's physics-loss residual at inference.
    - signals: scalar sensor vocabulary for the Curing press (RC-07) per
      their telemetry_source.py: mould_temp_C, coil_power_kW,
      vibration_rms_mm_s, pressure_bar.
    - note: keep neutral wording for healthy readings (their mock triage
      keys on substrings like "drift"/"critical").
    """
    import time as _time

    payload = {
        "machine_id": machine_id,
        "department": department,
        "epoch": int(epoch),
        "timestamp": float(timestamp if timestamp is not None else _time.time()),
        "signals": {k: float(v) for k, v in signals.items()},
        "pinn": {
            "health_index": max(0.0, min(1.0, float(health_index))),
            "rul_cycles": None if rul_cycles is None else float(rul_cycles),
            "residual": float(residual),
            "failure_mode_probs": {k: float(v) for k, v in failure_mode_probs.items()},
        },
        "note": note,
    }
    signature = hmac.new(_praetor_secret(), _canonical(payload),
                         hashlib.sha256).hexdigest()
    return {"payload": payload, "signature": signature}
