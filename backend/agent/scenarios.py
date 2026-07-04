"""Synthetic RC-07 scenario catalogue built on the VERIFIED limits. # D4-scenarios-v1

Ten scenarios (S1-S10) exercising every architecture component; each entry is
  dict(id, title, readings=[PinnReading kwargs...], expected={component: outcome},
       expected_risk=<label used in the CSV>)
All readings are RC-07 / Curing unless stated. Values are chosen AGAINST
`backend.agent.limits` (the single source of truth):
  temp normal 180-210 (steam 190-210) | pressure steam 16-19 bar OR >20 bar
  (19-20 gap indeterminate) | cycle 10-15 min normal, <=30 extended |
  vibration ISO zones A<=1.12, B<=2.8, C<=7.1, D>7.1 mm/s.

Mock-note discipline (NOTES-B.md section 2 / telemetry_source docstring):
healthy notes carry NO trigger words; genuinely anomalous notes deliberately
carry one ("rising"/"anomal"/"failure"/"imminent"), because the offline
MockClient tier-2 classifies by keyword. Scenario notes follow that rule so
the expected outcomes hold in mock mode AND read sensibly live.

`export_csv(path)` writes the deterministic synthetic dataset
(pinn/data/synthetic_rc07_scenarios.csv) for PINN training/eval.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .limits import vib_zone

_DATA_DIR = Path(__file__).resolve().parents[2] / "pinn" / "data"
DEFAULT_CSV = _DATA_DIR / "synthetic_rc07_scenarios.csv"

S10_OVERRIDE_REASON = "vibration sensor recalibrated last week, reading suspect"
S9_QUESTION = "Why the alarm on curing press RC-07? cite data"


def _r(epoch: int, temp: float, press: float, vib: float, cyc: float,
       health: float, rul, residual: float, note: str,
       modes: dict[str, float] | None = None) -> dict[str, Any]:
    """One PinnReading kwargs dict (RC-07/Curing)."""
    return dict(
        machine_id="RC-07", department="Curing", epoch=epoch,
        signals={"mould_temp_C": temp, "pressure_bar": press,
                 "vibration_rms_mm_s": vib, "cycle_min": cyc},
        pinn=dict(health_index=health, rul_cycles=rul, residual=residual,
                  failure_mode_probs=modes or {}),
        note=note,
    )


def _s8_readings() -> list[dict[str, Any]]:
    """10 epochs, vibration ramp 0.9 -> 3.2 (zone A into low C), health
    0.95 -> 0.80, benign notes (no mock trigger words)."""
    out = []
    for i in range(10):
        f = i / 9.0
        out.append(_r(
            epoch=10 + i,
            temp=round(196.0 + 1.0 * f, 2),
            press=round(17.5 - 0.2 * f, 2),
            vib=round(0.9 + (3.2 - 0.9) * f, 2),
            cyc=12.5,
            health=round(0.95 - 0.15 * f, 4),
            rul=None,
            residual=0.03,
            note=("Routine cure cycles on press RC-07; vibration inside the "
                  "acceptable long-term band, mould temperature steady."),
        ))
    return out


SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "S1", "title": "nominal-steam-cure",
        "readings": [_r(1, 196.0, 17.5, 0.9, 12.5, 0.97, 150.0, 0.02,
                        "Nominal steam cure on press RC-07: all curing "
                        "signals inside the verified envelope.")],
        "expected": {"tier1": "CLEAR at tier 1", "advisory": "none",
                     "spec": "run_diagnostic: NO out-of-limit signal "
                             "(196 degC is healthy steam-direct)"},
        "expected_risk": "CLEAR",
    },
    {
        "id": "S2", "title": "zoneB-longterm",
        "readings": [_r(2, 197.0, 17.2, 2.5, 13.0, 0.93, 140.0, 0.05,
                        "Steady cure cycles on press RC-07; vibration in the "
                        "acceptable long-term band.")],
        "expected": {"tier1": "CLEAR or WATCH max (zone B acceptable "
                              "long-term)", "advisory": "none"},
        "expected_risk": "<=WATCH",
    },
    {
        "id": "S3", "title": "zoneC-surveillance",
        "readings": [_r(3, 205.0, 17.8, 5.5, 12.8, 0.55, 60.0, 0.28,
                        "Vibration rising on press RC-07 mould shoulder "
                        "bearing over recent cycles; keeping the press "
                        "under close watch.")],
        "expected": {"tier2": "reached (zone C flag)", "risk": ">=HIGH",
                     "advisory": "yes, WITH debate AND jury attached"},
        "expected_risk": ">=HIGH",
    },
    {
        "id": "S4", "title": "zoneD-overheat-critical",
        "readings": [_r(4, 214.0, 15.2, 8.2, 18.0, 0.12, 6.0, 0.9,
                        "Danger on press RC-07: mould overheating and "
                        "bearing failure imminent — stop-level condition.",
                        modes={"bearing_wearout": 0.62,
                               "thermal_runaway": 0.30})],
        "expected": {"risk": "CRITICAL",
                     "tier1": "reasons name the limits (214 above 210 "
                              "ceiling, zone D, pressure below band)",
                     "advisory": "yes + debate + jury; jury.passed True; "
                                 "message cites digits"},
        "expected_risk": "CRITICAL",
    },
    {
        "id": "S5", "title": "pressure-loss-undercure",
        "readings": [_r(5, 195.0, 14.5, 1.0, 12.8, 0.6, 40.0, 0.18,
                        "Pressure dropping on press RC-07 steam circuit: "
                        "cure pressure below the steam-direct band, "
                        "undercure risk rising cycle over cycle.",
                        modes={"pressure_loss": 0.55})],
        "expected": {"risk": ">=HIGH", "advisory": "yes",
                     "causal": "causal/justification mentions pressure"},
        "expected_risk": ">=HIGH",
    },
    {
        "id": "S6", "title": "cycle-overrun",
        "readings": [_r(6, 198.0, 17.0, 1.05, 31.0, 0.8, 90.0, 0.1,
                        "Cycle-time anomaly on press RC-07: cure cycle "
                        "stretched to 31 minutes, beyond the 30-minute "
                        "extended ceiling.")],
        "expected": {"tier1": "flags cycle overrun (31 > 30 extended max)",
                     "risk": "WATCH or HIGH, NOT CLEAR",
                     "advisory": "only if HIGH"},
        "expected_risk": "WATCH|HIGH",
    },
    {
        "id": "S7", "title": "tampered-hmac",
        # Valid S4-style payload; the HARNESS corrupts the signature.
        "readings": [_r(7, 213.5, 15.3, 8.1, 18.2, 0.13, 7.0, 0.88,
                        "Danger on press RC-07: mould overheating and "
                        "bearing failure imminent — stop-level condition.",
                        modes={"bearing_wearout": 0.6})],
        "expected": {"hmac": "hmac_verified False, rejected before triage",
                     "events": "hmac_rejected logged", "advisory": "none"},
        "expected_risk": "REJECTED",
    },
    {
        "id": "S8", "title": "slow-drift",
        "readings": _s8_readings(),
        "expected": {"ticks": "each tick <=WATCH, NO advisory",
                     "drift-tool": "analyze_drift reports drifting/verdict "
                                   "change on vibration"},
        "expected_risk": "<=WATCH",
    },
    {
        "id": "S9", "title": "operator-diagnosis",
        "readings": [],  # question-only scenario (uses data stored by S1-S8)
        "expected": {"operator-agent": ">=2 real digits from stored data, "
                                       "'Data Provenance' section, >=2 tools "
                                       "in provenance"},
        "expected_risk": "n/a",
    },
    {
        "id": "S10", "title": "override-feedback",
        # S3-like reading replayed for the fresh debate after the override.
        "readings": [_r(20, 205.0, 17.8, 5.6, 12.8, 0.54, 58.0, 0.29,
                        "Vibration rising again on press RC-07 mould "
                        "shoulder bearing; repeat of the earlier pattern.")],
        "expected": {"override-loop": "store.get_overrides shows the "
                                      "override; fresh debate skeptic "
                                      "context contains the pushback text"},
        "expected_risk": ">=HIGH",
    },
]


def export_csv(path: str | Path = DEFAULT_CSV) -> int:
    """Write the deterministic synthetic RC-07 dataset. Returns row count
    (data rows, header excluded)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["scenario_id", "epoch", "machine_id", "mould_temp_C",
            "pressure_bar", "vibration_rms_mm_s", "cycle_min", "health_index",
            "rul_cycles", "residual", "vib_zone", "expected_risk", "note"]
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for sc in SCENARIOS:
            for rd in sc["readings"]:
                sig, pinn = rd["signals"], rd["pinn"]
                rul = pinn["rul_cycles"]
                w.writerow([
                    sc["id"], rd["epoch"], rd["machine_id"],
                    sig["mould_temp_C"], sig["pressure_bar"],
                    sig["vibration_rms_mm_s"], sig["cycle_min"],
                    pinn["health_index"],
                    "" if rul is None else rul,
                    pinn["residual"],
                    vib_zone(float(sig["vibration_rms_mm_s"])),
                    sc["expected_risk"], rd["note"],
                ])
                n += 1
    return n


__all__ = ["SCENARIOS", "export_csv", "DEFAULT_CSV",
           "S10_OVERRIDE_REASON", "S9_QUESTION"]
