"""Tier 1 (pure-python thresholds) + Tier 2 (fast-model classify) + causal matrix.

Tier-1 CLEAR envelope — every condition must hold, otherwise escalate to
Tier 2. Chosen so ~80% of healthy replay ticks exit here without any LLM call:

  health_index >= 0.70      (1.0 = healthy; 0.70 keeps a comfortable margin)
  residual     <= 0.20      (physics-consistency residual; above it, the PINN
                             and the machine disagree — worth a model look)
  rul_cycles   None or >=50 (under ~50 cycles we want an LLM opinion)
  failure_mode_probs all < 0.30   (AI4I modes TWF/HDF/PWF/OSF/RNF, bearings)
  |*_delta / *_z / *_sigma signals| < 3.0   (classic 3-sigma envelope)
  note has no alarm keyword (fail/alarm/imminent/critical/overheat/trip/leak)

Hard floors quoted in the reason when crossed (strongly suspicious):
  health_index < 0.45, residual > 0.50, rul < 30, any mode prob >= 0.50.

Tier 2 — one-word CLEAR/WATCH/HIGH/CRITICAL from the fast model
(DeepSeek V4 Flash, hint="tier2_classify"). Unparseable answer -> WATCH
(conservative: keeps the machine on the radar without waking Tier 3).
"""
from __future__ import annotations

import re
import time
from typing import Optional

from .crusoe_client import LLMClient
from .prompts import TIER2_SYSTEM, reading_digest, tier2_user
from .schemas import PinnReading, RiskLabel, TriageResult

# Thresholds (documented above).
CLEAR_HEALTH_FLOOR = 0.70
HARD_HEALTH_FLOOR = 0.45
CLEAR_RESIDUAL_CEIL = 0.20
HARD_RESIDUAL_CEIL = 0.50
CLEAR_RUL_FLOOR = 50.0
HARD_RUL_FLOOR = 30.0
MODE_PROB_WATCH = 0.30
MODE_PROB_HARD = 0.50
SIGMA_LIMIT = 3.0
ALARM_KEYWORDS = ("fail", "alarm", "imminent", "critical", "overheat", "trip", "leak")
_DELTA_SUFFIXES = ("_delta", "_z", "_sigma")

# Causal matrix — anomaly tag -> probable cause / downstream effect / remedy.
# Tags align with AI4I 2020 failure modes + bearing degradation, phrased for
# the Roanne chain (Mixing Banbury MX-02, Calender CL-03, Curing press RC-07).
CAUSAL_MATRIX: dict[str, dict[str, str]] = {
    "HDF": {
        "probable_cause": "Heat-dissipation failure: cooling circuit fouled or "
                          "process/ambient temperature gap collapsed.",
        "downstream_effect": "Mould/roll temperature drifts out of spec; cured "
                             "tyres off-spec, green tyres scrapped at the curing bottleneck.",
        "remedy": "Inspect press cooling circuit and fans, verify thermocouples, "
                  "plan a heat-exchanger clean in the next maintenance window.",
    },
    "PWF": {
        "probable_cause": "Power failure band: torque x speed outside the "
                          "3.5-9 kW operating window (drive or supply issue).",
        "downstream_effect": "Torque dips stall the Banbury mixer or calender "
                             "drive; upstream compound starves the line.",
        "remedy": "Check drive electrics, motor current draw and supply quality "
                  "before the next batch; keep a spare drive on standby.",
    },
    "OSF": {
        "probable_cause": "Overstrain: load x tool-wear product beyond the "
                          "design limit (over-dense compound or jammed feed).",
        "downstream_effect": "Mechanical overstress risks a snapped shaft or "
                             "roll — sudden stop and a nip-point hazard at the calender.",
        "remedy": "Reduce line load, verify feed density, schedule a mechanical "
                  "inspection of the loaded assembly.",
    },
    "TWF": {
        "probable_cause": "Tool wear at/near end of life (wear minutes beyond "
                          "the replacement band).",
        "downstream_effect": "Dimension drift and surface defects; rising "
                             "reject rate at inspection.",
        "remedy": "Swap the worn tool at the next planned changeover; do not "
                  "run to failure.",
    },
    "BEARING": {
        "probable_cause": "Bearing degradation: vibration RMS rising against "
                          "baseline (spall or lubrication starvation).",
        "downstream_effect": "Vibration propagates to the press frame; seizure "
                             "would force an unplanned curing stop (the costliest on the line).",
        "remedy": "Trend vibration under dense polling, prepare a bearing swap "
                  "kit, plan replacement inside the next maintenance window.",
    },
    "RNF": {
        "probable_cause": "No dominant physical driver: possible sensor fault "
                          "or random excursion.",
        "downstream_effect": "Low direct impact, but masks real drift if ignored.",
        "remedy": "Re-read the sensor, cross-check against neighbouring "
                  "signals, keep the machine on watch.",
    },
}
CAUSAL_MATRIX["GENERIC"] = CAUSAL_MATRIX["RNF"]

_KEYWORD_TAGS = (
    (("heat", "temp", "therm", "cool"), "HDF"),
    (("power", "current", "torque", "volt"), "PWF"),
    (("strain", "overload", "stress", "jam"), "OSF"),
    (("wear", "tool"), "TWF"),
    (("vib", "bearing", "rms"), "BEARING"),
)


def pick_causal(reading: PinnReading) -> dict[str, str]:
    """Dominant failure-mode prob wins; else note/signal keywords; else generic."""
    probs = reading.pinn.failure_mode_probs or {}
    if probs:
        tag = max(probs, key=probs.get).upper()
        entry = CAUSAL_MATRIX.get(tag) or CAUSAL_MATRIX.get(
            "BEARING" if "bear" in tag.lower() else "GENERIC"
        )
        return {"tag": tag, **entry}
    haystack = (reading.note + " " + " ".join(reading.signals)).lower()
    for keys, tag in _KEYWORD_TAGS:
        if any(k in haystack for k in keys):
            return {"tag": tag, **CAUSAL_MATRIX[tag]}
    return {"tag": "GENERIC", **CAUSAL_MATRIX["GENERIC"]}


def _tier1_flags(reading: PinnReading) -> list[str]:
    """Empty list == CLEAR. Wording avoids mock trigger words where possible."""
    p, flags = reading.pinn, []
    if p.health_index < HARD_HEALTH_FLOOR:
        flags.append(f"health {p.health_index * 100:.0f}% under hard floor {HARD_HEALTH_FLOOR * 100:.0f}%")
    elif p.health_index < CLEAR_HEALTH_FLOOR:
        flags.append(f"health {p.health_index * 100:.0f}% under envelope {CLEAR_HEALTH_FLOOR * 100:.0f}%")
    if p.residual > HARD_RESIDUAL_CEIL:
        flags.append(f"residual {p.residual:.2f} over hard ceiling {HARD_RESIDUAL_CEIL}")
    elif p.residual > CLEAR_RESIDUAL_CEIL:
        flags.append(f"residual {p.residual:.2f} over envelope {CLEAR_RESIDUAL_CEIL}")
    if p.rul_cycles is not None:
        if p.rul_cycles < HARD_RUL_FLOOR:
            flags.append(f"RUL {p.rul_cycles:.0f} cycles under hard floor {HARD_RUL_FLOOR:.0f}")
        elif p.rul_cycles < CLEAR_RUL_FLOOR:
            flags.append(f"RUL {p.rul_cycles:.0f} cycles under envelope {CLEAR_RUL_FLOOR:.0f}")
    for mode, prob in (p.failure_mode_probs or {}).items():
        if prob >= MODE_PROB_WATCH:
            band = "hard" if prob >= MODE_PROB_HARD else "watch"
            flags.append(f"mode {mode} p={prob:.2f} ({band} band)")
    for name, value in reading.signals.items():
        if name.lower().endswith(_DELTA_SUFFIXES) and abs(value) >= SIGMA_LIMIT:
            flags.append(f"signal {name} at {abs(value):.1f} sigma (limit {SIGMA_LIMIT:.0f})")
    hits = [k for k in ALARM_KEYWORDS if k in reading.note.lower()]
    if hits:
        flags.append(f"note keyword: {', '.join(hits)}")
    return flags


_LABELS = ("CRITICAL", "HIGH", "WATCH", "CLEAR")


def _parse_label(text: str) -> RiskLabel:
    words = re.sub(r"[^A-Z]", " ", (text or "").upper()).split()
    for lbl in _LABELS:  # exact word first, most severe first
        if lbl in words:
            return RiskLabel[lbl]
    for lbl in _LABELS:  # then substring (models love prose)
        if lbl in (text or "").upper():
            return RiskLabel[lbl]
    return RiskLabel.WATCH  # conservative default


async def run_triage(client: LLMClient, reading: PinnReading) -> TriageResult:
    """Tier 1 pure python; Tier 2 fast-model one-word classify when flagged."""
    t0 = time.perf_counter()
    flags = _tier1_flags(reading)
    tier1_ms = (time.perf_counter() - t0) * 1000.0
    if not flags:
        return TriageResult(tier_reached=1, risk=RiskLabel.CLEAR,
                            reason="within envelope",
                            latency_ms={"tier1": round(tier1_ms, 3)})
    reason = "; ".join(flags)
    causal = pick_causal(reading)
    t1 = time.perf_counter()
    raw = await client.complete(
        role="fast",
        messages=[{"role": "system", "content": TIER2_SYSTEM},
                  {"role": "user", "content": tier2_user(reading_digest(reading), reason)}],
        max_tokens=8, temperature=0.0, hint="tier2_classify",
    )
    tier2_ms = (time.perf_counter() - t1) * 1000.0
    risk = _parse_label(raw)
    return TriageResult(
        tier_reached=2, risk=risk,
        reason=f"tier1: {reason} | tier2: {raw.strip()[:60]}",
        causal_context=causal,
        latency_ms={"tier1": round(tier1_ms, 3), "tier2": round(tier2_ms, 1)},
    )
