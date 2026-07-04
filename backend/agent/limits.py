"""Verified operating limits — SINGLE SOURCE OF TRUTH (Builder D). # D4-limits-v1

Every module that judges a signal against a limit (triage tier-1,
operator_tools machine specs, telemetry scaling, scenario generation) must
read the numbers from HERE, never hardcode its own copy. That is how the old
invented-limit bugs happened (e.g. a 195 degC "trip" that false-alarmed a
perfectly healthy 196 degC steam cure).

Provenance discipline: `PROVENANCE` maps each parameter to its EXACT source
string as verified by Eric, including
  - the ISO 10816-3 / 20816-3 caveat for the vibration zones (a GENERAL
    industrial reference, NOT measured on this specific press), and
  - the two-pressure-formulations note (">20 bar" vs "1.6-1.9 MPa = 16-19
    bar") which are kept verbatim, never merged into one rounded value.
The 19-20 bar gap between the two pressure formulations is therefore
"indeterminate" — flagged WATCH, never treated as an alarm.

CL-03 / MX-02 limits (AI4I-2020-derived stand-ins) also live here so specs
have ONE home (moved from operator_tools.MACHINE_SPECS literals).
"""
from __future__ import annotations

from typing import Any

# ==================================================================== RC-07
# Tyre curing press (RC-07 class), steam-direct.
CURING_LIMITS: dict[str, Any] = {
    # Temperature, degC: normal 180-210; steam-direct specifically 190-210.
    "temp_normal": (180.0, 210.0),
    "temp_steam": (190.0, 210.0),
    # Pressure, bar: TWO source formulations kept side by side —
    # ">20 bar normal" AND "1.6-1.9 MPa (steam-direct) = 16-19 bar".
    "pressure_normal_min_bar": 20.0,
    "pressure_steam_bar": (16.0, 19.0),
    # Cycle time, minutes: 10-15 normal, up to 30 depending on tyre size/type.
    "cycle_min_range": (10.0, 15.0),
    "cycle_max_extended": 30.0,
    # Vibration RMS, mm/s — ISO zone ceilings. Zone D = anything above C.
    "vib_zones_mm_s": {"A": 1.12, "B": 2.8, "C": 7.1},
}

# =============================================================== CL-03/MX-02
# AI4I-2020-derived stand-in limits (dataset logic, not plant-verified).
CL03_LIMITS: dict[str, Any] = {
    "roll_temp_C": (30.0, 45.0),
    "roll_temp_trip_C": 50.0,
    "thermal_margin_K": (8.0, 15.0),          # floor 8 K per AI4I HDF logic
    "thermal_margin_floor_K": 8.0,
    "thermal_margin_alert_low_K": 9.0,
    "hdf_rpm_ceiling": 1380.0,                # HDF band: margin <8 K AND rpm <~1380
    "drive_speed_rpm": (1200.0, 2900.0),
    "drive_torque_Nm": (10.0, 70.0),
    "drive_torque_trip_Nm": 76.0,
    "tool_wear_min": (0.0, 220.0),
    "tool_wear_trip_min": 250.0,
}

MX02_LIMITS: dict[str, Any] = {
    "chamber_temp_C": (30.0, 45.0),
    "chamber_temp_trip_C": 55.0,
    "thermal_margin_K": (8.0, 15.0),
    "drive_speed_rpm": (1200.0, 2900.0),
    "drive_torque_Nm": (10.0, 70.0),
    "drive_torque_trip_Nm": 76.0,
    "tool_wear_min": (0.0, 220.0),
    "tool_wear_trip_min": 250.0,
    "pwf_power_window_kW": (3.5, 9.0),        # torque x speed outside -> PWF band
    "osf_wear_torque_limit_minNm": 11000.0,   # wear x torque beyond -> OSF band
}

# ================================================================ provenance
# EXACT source strings — cite these verbatim, do not paraphrase the numbers.
PROVENANCE: dict[str, str] = {
    "mould_temp_C": (
        'Normal 180-210 degC; steam-direct specifically 190-210 degC. '
        'Source: "presse de cuisson pneu, vapeur directe".'
    ),
    "pressure_bar": (
        'TWO source formulations, both kept verbatim: (1) ">20 bar" normal; '
        '(2) "1.6-1.9 MPa" in steam-direct = 16-19 bar. Do NOT merge into a '
        'single rounded value; the 19-20 bar gap between the two formulations '
        'is treated as indeterminate (WATCH, not alarm). '
        'Source: "presse de cuisson pneu, vapeur directe".'
    ),
    "cycle_min": (
        'Cycle time 10-15 min normal; up to 30 min depending on tyre '
        'size/type. Source: "presse de cuisson pneu, vapeur directe".'
    ),
    "vibration_rms_mm_s": (
        'Vibration RMS zones (mm/s): Zone A <=1.12 excellent; Zone B <=2.8 '
        'acceptable long-term; Zone C <=7.1 surveillance required; Zone D '
        '>7.1 danger. Source: ISO 10816-3 / ISO 20816-3, Group 2 (driven '
        'machines 15-300 kW, rigid foundation) — GENERAL industrial '
        'reference, not measured on this specific press; treat as a starting '
        'reference.'
    ),
    "cl03_thermal_margin_K": (
        'AI4I 2020 HDF logic: process/air thermal margin under 8 K with nip '
        'drive under ~1380 rpm = heat-dissipation failure band. '
        'Dataset-derived stand-in limit, not plant-verified.'
    ),
    "mx02_torque_wear": (
        'AI4I 2020 PWF/OSF logic: torque x speed outside the 3.5-9 kW power '
        'window = power-failure band; wear x torque beyond ~11,000 min.Nm = '
        'overstrain band. Dataset-derived stand-in limits, not plant-verified.'
    ),
}


# ============================================================ status helpers
def vib_zone(v: float) -> str:
    """ISO 10816-3/20816-3 Group 2 zone for a vibration RMS value (mm/s)."""
    z = CURING_LIMITS["vib_zones_mm_s"]
    if v <= z["A"]:
        return "A"
    if v <= z["B"]:
        return "B"
    if v <= z["C"]:
        return "C"
    return "D"


def temp_status(t: float) -> str:
    """Mould temperature vs the verified 180-210 (190-210 steam-direct) bands.

    "normal"                  — inside the steam-direct band 190-210 degC
    "normal_below_steam_band" — inside the general 180-210 band but under the
                                190 degC steam-direct floor (soft watch)
    "below_normal_floor"      — under 180 degC (abnormal)
    "above_normal_ceiling"    — over 210 degC (abnormal)
    """
    lo, hi = CURING_LIMITS["temp_normal"]
    steam_lo, _ = CURING_LIMITS["temp_steam"]
    if t < lo:
        return "below_normal_floor"
    if t > hi:
        return "above_normal_ceiling"
    if t < steam_lo:
        return "normal_below_steam_band"
    return "normal"


def pressure_status(p: float) -> str:
    """Pressure vs BOTH source formulations (16-19 bar steam AND >20 bar).

    "normal_steam_band"             — 16-19 bar (steam-direct formulation)
    "normal_high_formulation"       — >=20 bar (">20 bar normal" formulation)
    "indeterminate_two_formulations"— 19-20 bar gap between the two source
                                      formulations: WATCH, not alarm
    "below_bands"                   — under 16 bar (below both formulations)
    """
    steam_lo, steam_hi = CURING_LIMITS["pressure_steam_bar"]
    if p < steam_lo:
        return "below_bands"
    if p <= steam_hi:
        return "normal_steam_band"
    if p < CURING_LIMITS["pressure_normal_min_bar"]:
        return "indeterminate_two_formulations"
    return "normal_high_formulation"


def cycle_status(m: float) -> str:
    """Cycle time vs 10-15 min normal / <=30 min extended (tyre-dependent).

    "normal"   — 10-15 min
    "extended" — 15-30 min (acceptable depending on tyre size/type; watch)
    "overrun"  — over 30 min (beyond the extended ceiling)
    "short"    — under 10 min (undercure risk)
    """
    lo, hi = CURING_LIMITS["cycle_min_range"]
    if m < lo:
        return "short"
    if m <= hi:
        return "normal"
    if m <= CURING_LIMITS["cycle_max_extended"]:
        return "extended"
    return "overrun"


__all__ = [
    "CURING_LIMITS", "CL03_LIMITS", "MX02_LIMITS", "PROVENANCE",
    "vib_zone", "temp_status", "pressure_status", "cycle_status",
]
