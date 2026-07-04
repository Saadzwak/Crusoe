"""REAL-CONDITIONS cross-layer contract test — both codebases, same tree.

Until integration, the PINN side was verified against a re-implementation of
the information layer's HMAC logic. This test imports the TEAMMATE'S ACTUAL
MODULES (backend.agent.hmac_auth, backend.agent.schemas) and checks that a
SignedReading emitted by pinn.telemetry:

  1. verifies with backend.agent.hmac_auth.verify_payload (byte-level HMAC),
  2. parses into backend.agent.schemas.SignedReading + PinnReading + PinnState
     (pydantic validation — field names, types, bounds),
  3. fails verification when tampered,
  4. stays verifiable when PINN_HMAC_SECRET is set to an EMPTY string on both
     sides (the .env.example default — regression test for the get()-default
     bug found at integration),
  5. round-trips a reading built from the REAL v2 model metrics if present.

Run from repo root:  python scripts/test_cross_layer_contract.py
Exit code 0 = contract holds in real conditions.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    # --- the teammate's REAL modules (not re-implementations) ---------------
    from backend.agent.hmac_auth import sign_payload, verify_payload
    from backend.agent.schemas import PinnReading, PinnState, SignedReading

    from pinn.telemetry import to_praetor_signed_reading

    print("== cross-layer contract, both codebases in the same tree ==")

    # Reading built from the real v2 metrics when available (demo-faithful).
    metrics_path = ROOT / "runs" / "v2" / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        damage = m.get("fatigue", {}).get("final_window_damage_D", 0.12)
        rul = m.get("rul", {}).get("pred_range", [100.0])[0]
        vib_acc = m.get("vibration", {}).get("val_accuracy_4class", 0.95)
        src = "runs/v2/metrics.json (real v2 outputs)"
    else:
        damage, rul, vib_acc = 0.12, 100.0, 0.95
        src = "fallback constants (metrics.json absent)"
    print(f"  reading source: {src}")

    sr = to_praetor_signed_reading(
        machine_id="RC-07", department="Curing", epoch=1,
        signals={"mould_temp_C": 195.0, "coil_power_kW": 71.0,
                 "vibration_rms_mm_s": 2.1, "pressure_bar": 17.1},
        health_index=1.0 - damage, rul_cycles=rul, residual=0.0,
        failure_mode_probs={"bearing_fault": round(1.0 - vib_acc, 4)},
        note="integration contract test reading")

    # 1. their HMAC verifies our signature
    check("their verify_payload accepts our signature",
          verify_payload(sr["payload"], sr["signature"]))
    # ...and our signature equals what their signer would produce
    check("byte-identical signature vs their sign_payload",
          sign_payload(sr["payload"]) == sr["signature"])

    # 2. their pydantic contract parses our payload
    try:
        parsed = SignedReading(**sr)
        reading = PinnReading(**parsed.payload)
        ok = isinstance(reading.pinn, PinnState) and 0.0 <= reading.pinn.health_index <= 1.0
        check("SignedReading + PinnReading + PinnState pydantic parse", ok,
              f"health_index={reading.pinn.health_index:.3f}, "
              f"rul_cycles={reading.pinn.rul_cycles}")
    except Exception as e:  # noqa: BLE001
        check("SignedReading + PinnReading + PinnState pydantic parse", False, repr(e))

    # 3. tampering breaks it
    tampered = json.loads(json.dumps(sr))
    tampered["payload"]["signals"]["pressure_bar"] = 3.0
    check("tampered payload is rejected",
          not verify_payload(tampered["payload"], tampered["signature"]))

    # 4. empty-string PINN_HMAC_SECRET (the .env.example default) on both sides
    old = os.environ.get("PINN_HMAC_SECRET")
    os.environ["PINN_HMAC_SECRET"] = ""
    try:
        sr_empty = to_praetor_signed_reading(
            machine_id="RC-07", department="Curing", epoch=2,
            signals={"pressure_bar": 17.0}, health_index=0.9, rul_cycles=120.0,
            residual=0.0, failure_mode_probs={}, note="empty-env regression")
        check("empty PINN_HMAC_SECRET: both sides on demo fallback",
              verify_payload(sr_empty["payload"], sr_empty["signature"]))
    finally:
        if old is None:
            os.environ.pop("PINN_HMAC_SECRET", None)
        else:
            os.environ["PINN_HMAC_SECRET"] = old

    # 5. shared non-empty secret on both sides
    os.environ["PINN_HMAC_SECRET"] = "integration-test-secret"
    try:
        sr_named = to_praetor_signed_reading(
            machine_id="RC-07", department="Curing", epoch=3,
            signals={"pressure_bar": 17.0}, health_index=0.9, rul_cycles=120.0,
            residual=0.0, failure_mode_probs={}, note="named-secret case")
        # their side reads the secret at import via settings; re-import fresh
        import importlib

        import backend.agent.config as cfg
        import backend.agent.hmac_auth as ha
        importlib.reload(cfg)
        importlib.reload(ha)
        check("shared named secret verifies through their reloaded settings",
              ha.verify_payload(sr_named["payload"], sr_named["signature"]))
    finally:
        os.environ.pop("PINN_HMAC_SECRET", None)

    print(f"\n{'CONTRACT HOLDS' if not FAILURES else 'CONTRACT BROKEN'} "
          f"({5 - len(FAILURES)}/5 groups pass)")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
