"""Rubber cure (vulcanization) kinetics — the chemistry behind the pressure head.

CONFIDENCE CATEGORIES used throughout this repo's physics docs:
  (a) well-established physical law — textbook, not in question
  (b) from a specific cited source (paper / patent / standard) — cited inline
  (c) plausible-but-unconfirmed choice — awaiting validation, never presented
      as settled fact

Equations
---------
1. Heat conduction with reaction source [(a) — Fourier's law]:
       rho * c_p * dT/dt = div(k grad T) + Q_reaction
   Curing is COUPLED: the reaction releases exothermic heat (Q), it is not a
   one-way heat-in problem. We do not solve the full PDE in v1 (no spatial
   mesh); it is stated because it is *why* mold temperature and internal cure
   state differ — the internal cure state is the unmeasurable quantity the
   pressure/cure head estimates.

2. Cure rate constant [(a) Arrhenius form; (b) value range]:
       k(T) = A * exp(-(Ea/R) / T)
   Ea/R ~ 10,000-14,000 K for typical vulcanization reactions per patent
   US4022555 — a NON-peer-reviewed, indicative order of magnitude, cited as
   such (category (b) for provenance, treat the numeric range as indicative).

3. Cure kinetics [(b) model form — Kamal & Sourour 1976 (Polym. Eng. Sci.),
   the standard autocatalytic cure model in the rubber/thermoset literature,
   e.g. Rafei, Ghoreishy & Naderi, Comput. Mater. Sci. 2009 for rubber]:
       dalpha/dt = (k1 + k2 * alpha^m) * (1 - alpha)^n
   with k1, k2 Arrhenius in T. The exponents m, n and the ratio k2/k1 are
   COMPOUND-SPECIFIC, fitted per formulation from rheometer data in the
   literature; our defaults below are category (c) placeholders chosen inside
   commonly reported ranges (m ~ 0.5-2, n ~ 1-2) and must be re-fitted on ODR
   data (ASTM D2084 / ISO 3417) if a real compound is ever characterized.

4. Practical degree of cure [(b) — industrial practice per ASTM D2084 /
   ISO 3417]: alpha ~= t_cure / t99, with t99 the time to 99% of maximum
   torque on an oscillating-disc rheometer.

Calibration instead of invention
--------------------------------
The Arrhenius pre-factor A is NOT taken from literature (none exists for our
unspecified stand-in compound). It is SOLVED so that t99 at 195 degC equals
12 minutes — the midpoint of the VERIFIED 10-15 min curing-cycle range. That
makes A a category (c) value *anchored to verified cycle durations*, and it
is recomputed (bisection) rather than hard-coded, so changing Ea/R or m/n
keeps the model consistent with the documented cycle envelope.

Leak -> temperature coupling [(b) — steam tables]
-------------------------------------------------
For direct-steam vulcanization the mold cavity sits at the SATURATION
temperature of the steam: lose pressure, lose temperature. Around 15 bar the
saturation slope from standard steam tables is ~3.2 K/bar (10 bar/180 degC ->
15 bar/198 degC -> 20 bar/212 degC). A mid-hold seal leak therefore not only
drops pressure but slows the cure — which is exactly why an under-cure
estimate is the physically meaningful anomaly output, not just "pressure low".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- constants with explicit confidence tags -------------------------------
EA_OVER_R_K = 12_000.0        # (b, indicative) mid of 10k-14k K, patent US4022555
KAMAL_M = 1.0                 # (c) compound-specific, common literature range 0.5-2
KAMAL_N = 1.5                 # (c) compound-specific, common literature range 1-2
K2_OVER_K1 = 2.0              # (c) autocatalytic strength, to fit on ODR data
T99_CALIBRATION_S = 720.0     # (b-anchored) 12 min = midpoint of verified 10-15 min cycles
T99_CALIBRATION_C = 195.0     # (b) inside the verified 180-210 degC envelope
SAT_STEAM_SLOPE_K_PER_BAR = 3.2  # (b) standard steam tables, ~10-20 bar region
ALPHA_CURED_THRESHOLD = 0.90  # (c) demo threshold for "acceptably cured"


@dataclass(frozen=True)
class CureParams:
    ea_over_r: float = EA_OVER_R_K
    m: float = KAMAL_M
    n: float = KAMAL_N
    k2_over_k1: float = K2_OVER_K1
    prefactor_A: float | None = None  # None -> use calibrated module default


def kamal_rate(alpha, temp_K, A: float, p: CureParams) -> np.ndarray:
    """dalpha/dt of the Kamal-Sourour model at temperature temp_K [K]."""
    alpha = np.clip(np.asarray(alpha, dtype=float), 0.0, 1.0)
    k1 = A * np.exp(-p.ea_over_r / np.asarray(temp_K, dtype=float))
    k2 = p.k2_over_k1 * k1
    return (k1 + k2 * alpha**p.m) * (1.0 - alpha) ** p.n


def integrate_cure(temp_C: np.ndarray, dt_s: float, A: float | None = None,
                   p: CureParams = CureParams()) -> np.ndarray:
    """Euler-integrate alpha(t) along a temperature trace [degC] -> alpha array."""
    A = A if A is not None else calibrated_prefactor()
    temp_K = np.asarray(temp_C, dtype=float) + 273.15
    alpha = np.empty(len(temp_K))
    a = 1e-6
    for i, T in enumerate(temp_K):
        a = min(1.0, a + dt_s * float(kamal_rate(a, T, A, p)))
        alpha[i] = a
    return alpha


def _t99_at(A: float, temp_C: float, p: CureParams, dt_s: float = 1.0,
            t_max_s: float = 7200.0) -> float:
    """Time for alpha to reach 0.99 at constant temperature (Euler)."""
    a, t = 1e-6, 0.0
    while a < 0.99 and t < t_max_s:
        a += dt_s * float(kamal_rate(a, temp_C + 273.15, A, p))
        t += dt_s
    return t


_CALIBRATED_A: float | None = None


def calibrated_prefactor(p: CureParams = CureParams()) -> float:
    """Solve A by bisection so t99(195 degC) = 12 min. Cached after first call."""
    global _CALIBRATED_A
    if _CALIBRATED_A is not None and p == CureParams():
        return _CALIBRATED_A
    lo, hi = 1e4, 1e14
    for _ in range(60):
        mid = np.sqrt(lo * hi)  # geometric bisection over orders of magnitude
        if _t99_at(mid, T99_CALIBRATION_C, p) > T99_CALIBRATION_S:
            lo = mid            # too slow -> need bigger A
        else:
            hi = mid
    A = float(np.sqrt(lo * hi))
    if p == CureParams():
        _CALIBRATED_A = A
    return A


def saturation_temp_drop(p_deficit_bar) -> np.ndarray:
    """Temperature drop [K] for a steam-pressure deficit [bar] (steam tables)."""
    return SAT_STEAM_SLOPE_K_PER_BAR * np.maximum(0.0, np.asarray(p_deficit_bar, dtype=float))
