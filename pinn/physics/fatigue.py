"""Bearing fatigue physics — fatigue-anticipation head ("replace before it fails").

Confidence categories as in cure_kinetics.py: (a) textbook law, (b) cited
source, (c) plausible-but-unconfirmed, awaiting validation.

Chain of reasoning
------------------
 load + geometry --(Hertz, a)--> contact stress sigma_H
 sigma_H + S-N   --(Basquin, a/b)--> cycles-to-failure N_f
 elapsed cycles  --(Miner, a)--> damage D in [0, 1], replace before D -> 1

1. Hertzian line contact [(a) — classical contact mechanics, e.g. Harris,
   "Rolling Bearing Analysis"]:
       sigma_H = sqrt(P_lin * E_star / (pi * R_eff))
   with P_lin the max roller load per unit length, E_star the contact modulus
   (steel/steel: E=210 GPa, nu=0.3 -> E_star = 115.4 GPa, (a)), R_eff the
   equivalent radius at the inner-race contact (worst case).
   Max roller load via the Stribeck-type distribution [(b) Harris]:
       P_max = 4.6 * F_r / (i * Z * cos(alpha))   (zero-clearance roller value;
   ~5.0 with nominal clearance — we carry 4.6 and say so).

2. Basquin S-N law [(a) in form]:  sigma_a = sigma_f' * (2 N_f)^b
   - b for steels: reported -0.05..-0.12 [(b) general fatigue literature,
     e.g. Dowling, "Mechanical Behavior of Materials"].
   - A torsion study on AISI 52100 (HRC 58-62) found NO observable fatigue
     limit and a shear-stress-life EXPONENT of 10.34 [(b), specific study].
     That figure is from a DIFFERENT test methodology and parameterization
     (tau^k * N = C form, torsional loading) — it must NOT be averaged or
     interchanged with b above. We note (without equating) that 1/10.34 =
     0.097 happens to fall inside the |b| range, which is consonance, not
     validation.
   - Material: the IMS readme does not state the bearing steel; AISI 52100 is
     the industry-standard assumption for rolling bearings [(c), flagged].

3. Palmgren-Miner cumulative damage [(a) as the standard engineering rule]:
       D = sum_i n_i / N_f(sigma_i),  failure expected near D ~= 1
   Under the IMS rig's CONSTANT load and speed this reduces to D(t) = t/T_f,
   which is what supervises the fatigue head; Miner is what lets the same
   head extrapolate under variable future load. AI4I's OSF rule
   (tool_wear x torque > threshold) is conceptually a crude Miner analog —
   used as a documented cross-check only, never as fatigue training data.

Calibration against IMS (real failures) instead of importing constants
----------------------------------------------------------------------
IMS ran every test at ONE load/speed point (6000 lbs, 2000 rpm) — a single
stress level CANNOT identify both sigma_f' and b (two unknowns, one abscissa).
Honest procedure, stated as such:
  - count stress cycles to failure N_f from the real test durations
    (cycles = elapsed_seconds * BPFO Hz: each roller passage over the failing
    outer-race point is one Hertzian stress cycle; for the inner-race failure
    of test 1 use BPFI),
  - fix b on the literature grid, solve sigma_f' = sigma_H / (2 N_f)^b,
  - report the (b, sigma_f') family; literature values serve as a
    plausibility check, not ground truth.
Scatter across the three tests is expected (fatigue life is Weibull-
distributed; the L10/median distinction matters in real bearing rating).

Load-share caveat [(c), flagged]: the readme says the 6000 lbs radial load is
applied onto bearings 2 and 3; the exact share seen by each of the four
bearings (including the outboard ones that actually failed in tests 1 and 3)
depends on the rig's statics. We compute stress for a nominal
F_r = 6000 lbs / 2 per loaded bearing and carry this as an explicit
uncertainty absorbed by the sigma_f' calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pinn.physics.bearing import IMS_REXNORD_ZA2115, fault_frequencies

IN_TO_M = 0.0254
LBF_TO_N = 4.448_221_6

# --- material & load constants (tagged) -------------------------------------
E_STEEL_PA = 210e9            # (a) structural/bearing steel Young's modulus
NU_STEEL = 0.3                # (a)
E_STAR_PA = E_STEEL_PA / (2.0 * (1.0 - NU_STEEL**2))   # contact modulus, (a)
STRIBECK_ROLLER = 4.6         # (b) Harris, zero clearance (5.0 with clearance)
IMS_RADIAL_LOAD_LBF = 6000.0  # (b) IMS readme
IMS_LOAD_SHARE = 0.5          # (c) 6000 lbs applied onto bearings 2 & 3 -> /2
IMS_SHAFT_RPM = 2000.0        # (b) IMS readme
ROLLER_LENGTH_IN = 0.35       # (c) NOT in the readme — typical for d=0.331"
                              #     rollers of this class; flagged assumption
BASQUIN_B_GRID = (-0.05, -0.0713, -0.09, -0.12)  # (b) literature range for steels
B_52100_TORSION_EXPONENT = 10.34  # (b) torsion study, DIFFERENT parameterization
                                  #     — documented, NOT used in Basquin math


@dataclass(frozen=True)
class HertzResult:
    p_max_roller_N: float
    p_lin_N_per_m: float
    r_eff_m: float
    sigma_h_pa: float


def ims_hertz_stress(radial_load_lbf: float = IMS_RADIAL_LOAD_LBF * IMS_LOAD_SHARE,
                     roller_length_in: float = ROLLER_LENGTH_IN) -> HertzResult:
    """Max Hertzian contact pressure at the inner-race contact of the ZA-2115."""
    g = IMS_REXNORD_ZA2115
    alpha = math.radians(g.contact_angle_deg)
    f_r = radial_load_lbf * LBF_TO_N

    # Stribeck max roller load; i = 2 rows share the radial load.
    p_max = STRIBECK_ROLLER * f_r / (2 * g.n_elements * math.cos(alpha))

    r_roller = 0.5 * g.d_element * IN_TO_M
    # Inner raceway contact radius: (pitch - d*cos(alpha)) / 2.
    r_inner = 0.5 * (g.d_pitch - g.d_element * math.cos(alpha)) * IN_TO_M
    r_eff = (r_roller * r_inner) / (r_roller + r_inner)

    p_lin = p_max / (roller_length_in * IN_TO_M)
    sigma_h = math.sqrt(p_lin * E_STAR_PA / (math.pi * r_eff))
    return HertzResult(p_max, p_lin, r_eff, sigma_h)


def stress_cycles_per_second(failure_type: str = "OR",
                             shaft_rpm: float = IMS_SHAFT_RPM) -> float:
    """Stress cycles/s at the failing raceway point = roller-passage frequency."""
    f = fault_frequencies(IMS_REXNORD_ZA2115, shaft_rpm / 60.0)
    return f["BPFO"] if failure_type == "OR" else f["BPFI"]


def basquin_nf(sigma_a_pa: float, sigma_f_prime_pa: float, b: float) -> float:
    """Cycles to failure from Basquin: N_f = 0.5 * (sigma_a / sigma_f')^(1/b)."""
    return 0.5 * (sigma_a_pa / sigma_f_prime_pa) ** (1.0 / b)


def calibrate_sigma_f_prime(nf_observed_cycles: float, sigma_h_pa: float,
                            b: float) -> float:
    """Solve Basquin for sigma_f' at one observed (sigma, N_f) point."""
    return sigma_h_pa / (2.0 * nf_observed_cycles) ** b


def miner_damage(elapsed_cycles, nf_cycles) -> np.ndarray:
    """Constant-amplitude Miner damage fraction, clipped to [0, 1.2] for report."""
    return np.clip(np.asarray(elapsed_cycles, dtype=float) / nf_cycles, 0.0, 1.2)


def hours_to_replacement(damage_now: float, nf_cycles: float,
                         cps: float, safety_margin: float = 0.9) -> float:
    """Hours until D reaches `safety_margin` at the current cycling rate."""
    remaining = max(0.0, safety_margin - damage_now) * nf_cycles
    return remaining / (cps * 3600.0)
