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

D4-triage-v1 — VERIFIED physical limits on Curing (single source of truth =
backend.agent.limits, provenance strings included there). For department
"Curing", tier 1 additionally checks the raw signals:
  mould_temp_C        vs temp_status(): 180-210 normal (190-210 steam-direct);
                      outside 180-210 = hard flag; 180-190 = soft watch.
  pressure_bar        vs pressure_status(): 16-19 bar steam band normal AND
                      >20 bar normal (two source formulations); <16 = hard
                      flag; the 19-20 gap = "indeterminate — two source
                      formulations" soft WATCH, never an alarm.
  vibration_rms_mm_s  vs vib_zone() (ISO 10816-3/20816-3 Group 2, general
                      reference): zones A/B pass, C = hard flag "ISO zone C —
                      surveillance", D = hard flag "ISO zone D — danger".
  cycle_min           vs cycle_status(): 10-15 normal; 15-30 extended
                      (tyre-size dependent) = soft watch; >30 overrun and
                      <10 short = hard flags.
Reasons always name the limit AND the measured value ("mould_temp 214.0°C
above 210 normal ceiling"). Soft flags alone -> tier 1 returns WATCH without
waking tier 2 (watch, not alarm); any hard flag -> tier 2 classify.

Tier 2 — one-word CLEAR/WATCH/HIGH/CRITICAL from the fast model
(DeepSeek V4 Flash, hint="tier2_classify"). Unparseable answer -> WATCH
(conservative: keeps the machine on the radar without waking Tier 3).
Flag wording stays free of the offline mock's keyword triggers so mock
classification reacts to DATA (digest + note), never to our own labels.
"""
from __future__ import annotations

import re
import time
from typing import Optional

from .crusoe_client import LLMClient
from .limits import (CURING_LIMITS, cycle_status, pressure_status,
                     temp_status, vib_zone)
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
    # D4-triage-v1: curing pressure loss (undercure) — verified steam band
    # 16-19 bar / >20 bar (two source formulations, see limits.PROVENANCE).
    "PRESSURE": {
        "probable_cause": "Curing pressure loss: steam/bladder pressure under "
                          "the 16-19 bar steam-direct band (bladder, valve or "
                          "steam-supply defect).",
        "downstream_effect": "Undercured tyres at the bottleneck press — every "
                             "cycle run below the band risks scrap at inspection.",
        "remedy": "Check steam supply, bladder and pressure valve; verify the "
                  "pressure transducer against a reference gauge before the "
                  "next cycle.",
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
    (("undercure",), "PRESSURE"),  # D4-triage-v1 (note-driven only)
)


def pick_causal(reading: PinnReading) -> dict[str, str]:
    """Dominant failure-mode prob wins; else note/signal keywords; else generic."""
    probs = reading.pinn.failure_mode_probs or {}
    if probs:
        tag = max(probs, key=probs.get).upper()
        entry = CAUSAL_MATRIX.get(tag)
        if entry is None and ("press" in tag.lower() or "undercure" in tag.lower()):
            entry = CAUSAL_MATRIX["PRESSURE"]  # D4-triage-v1
        if entry is None:
            entry = CAUSAL_MATRIX.get(
                "BEARING" if "bear" in tag.lower() else "GENERIC")
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


# --------------------------------------------------------------- D4-triage-v1
def _sig(signals: dict, wanted: str) -> Optional[float]:
    """Case-insensitive signal lookup, tolerant of float-able values."""
    for k, v in signals.items():
        if k.lower() == wanted:
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def _curing_signal_flags(reading: PinnReading) -> tuple[list[str], list[str]]:
    """(hard_flags, soft_flags) vs the VERIFIED curing limits (limits.py).

    Hard flags escalate to tier 2; soft flags alone yield a tier-1 WATCH
    (indeterminate/extended bands are watch material, never alarms).
    Every reason names the limit AND the measured value. Wording is audited
    against the offline mock's tier-2 keyword triggers.
    """
    hard: list[str] = []
    soft: list[str] = []
    if reading.department.strip().lower() != "curing":
        return hard, soft
    t_lo, t_hi = CURING_LIMITS["temp_normal"]
    s_lo, _s_hi = CURING_LIMITS["temp_steam"]
    p_lo, p_hi = CURING_LIMITS["pressure_steam_bar"]
    c_lo, c_hi = CURING_LIMITS["cycle_min_range"]
    c_ext = CURING_LIMITS["cycle_max_extended"]
    zb = CURING_LIMITS["vib_zones_mm_s"]["B"]
    zc = CURING_LIMITS["vib_zones_mm_s"]["C"]

    t = _sig(reading.signals, "mould_temp_c")
    if t is not None:
        st = temp_status(t)
        if st == "above_normal_ceiling":
            hard.append(f"mould_temp {t:.1f}°C above {t_hi:.0f} normal ceiling")
        elif st == "below_normal_floor":
            hard.append(f"mould_temp {t:.1f}°C below {t_lo:.0f} normal floor")
        elif st == "normal_below_steam_band":
            soft.append(f"mould_temp {t:.1f}°C inside the {t_lo:.0f}-{t_hi:.0f} "
                        f"general band but under the {s_lo:.0f} steam-direct floor")

    p = _sig(reading.signals, "pressure_bar")
    if p is not None:
        sp = pressure_status(p)
        if sp == "below_bands":
            hard.append(f"pressure {p:.2f} bar below the {p_lo:.0f}-{p_hi:.0f} bar "
                        "steam-direct band (under both source formulations)")
        elif sp == "indeterminate_two_formulations":
            soft.append(f"pressure {p:.2f} bar in the {p_hi:.0f}-"
                        f"{CURING_LIMITS['pressure_normal_min_bar']:.0f} bar gap — "
                        "indeterminate — two source formulations")

    v = _sig(reading.signals, "vibration_rms_mm_s")
    if v is not None:
        zone = vib_zone(v)
        if zone == "C":
            hard.append(f"vibration_rms {v:.2f} mm/s — ISO zone C — surveillance "
                        f"(over {zb} mm/s)")
        elif zone == "D":
            hard.append(f"vibration_rms {v:.2f} mm/s — ISO zone D — danger "
                        f"(over {zc} mm/s)")

    cy = _sig(reading.signals, "cycle_min")
    if cy is not None:
        sc = cycle_status(cy)
        if sc == "overrun":
            hard.append(f"cycle {cy:.1f} min beyond the {c_ext:.0f} min extended "
                        f"ceiling ({c_lo:.0f}-{c_hi:.0f} normal)")
        elif sc == "short":
            hard.append(f"cycle {cy:.1f} min under the {c_lo:.0f}-{c_hi:.0f} min "
                        "normal band (undercure risk)")
        elif sc == "extended":
            soft.append(f"cycle {cy:.1f} min beyond the {c_lo:.0f}-{c_hi:.0f} min "
                        f"normal band (tyre-size dependent up to {c_ext:.0f})")
    return hard, soft


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
    """Tier 1 pure python; Tier 2 fast-model one-word classify when flagged.

    D4-triage-v1: Curing signals are checked against the verified limits IN
    ADDITION to the generic health/residual/RUL/mode/sigma/note checks. Soft
    flags alone (indeterminate pressure gap, extended-but-allowed cycle,
    below-steam-band temperature) return WATCH from tier 1 directly.
    """
    t0 = time.perf_counter()
    curing_hard, curing_soft = _curing_signal_flags(reading)
    hard_flags = curing_hard + _tier1_flags(reading)
    tier1_ms = (time.perf_counter() - t0) * 1000.0
    if not hard_flags:
        if curing_soft:
            return TriageResult(
                tier_reached=1, risk=RiskLabel.WATCH,
                reason="watch: " + "; ".join(curing_soft),
                causal_context=pick_causal(reading),
                latency_ms={"tier1": round(tier1_ms, 3)},
            )
        return TriageResult(tier_reached=1, risk=RiskLabel.CLEAR,
                            reason="within envelope",
                            latency_ms={"tier1": round(tier1_ms, 3)})
    reason = "; ".join(hard_flags + curing_soft)
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
