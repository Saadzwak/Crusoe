"""Curing (vulcanization) press cycle model — pressure head.

============================ SYNTHETIC DATA NOTICE ============================
No public dataset of tire-curing-press pressure traces exists. Everything the
pressure head trains on is GENERATED from this module and must be labeled
"synthetic" wherever it appears (code, docs, demo output). This is expected,
accepted hackathon practice — but never present it as measured data.
===============================================================================

Operating envelope (real, public engineering values for tire curing presses —
provided and verified by the team brief; not from any confidential source):
- cure temperature: 180-210 degC
- pressure: > 20 bar press clamping; 1.6-1.9 MPa (16-19 bar) for
  direct-steam-in-bladder vulcanization
- cycle duration: 10-15 minutes

Governing equation used for the nominal cycle and as the PINN residual — a
first-order pressure response (standard lumped model of a pneumatic/steam
volume being filled through a restriction; the same form as an RC charging
curve, from mass balance through an orifice linearized around the setpoint):

    ramp  (0 <= t < t_ramp):    dP/dt = (P_set - P) / tau_ramp
    hold  (t_ramp <= t < t_rel): dP/dt ~= 0        (P ~= P_set, small noise)
    release (t >= t_rel):        dP/dt = -P / tau_release

Why this is legitimate physics rather than decoration: steam/gas flow into a
fixed mold volume through a valve behaves, to first order, like a first-order
lag — the fill rate is proportional to the upstream/downstream difference.
The exact time constants of a real Michelin press are unknown to us (and
confidential); the *form* of the dynamics is textbook. We therefore constrain
the model to the ODE form, with time constants chosen inside plausible ranges,
and we say so explicitly.

Anomaly scenario injected for training (the failure mode the PINN must catch):
a mid-hold seal/bladder leak, modeled as an additional loss term in the mass
balance:

    leak:   dP/dt = -k_leak * P     (during hold, from t_leak onward)

which produces the characteristic slow pressure decay during what should be a
flat hold phase.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CuringCycleParams:
    """Parameters of one synthetic curing cycle. Units: seconds, bar, degC."""

    cycle_s: float = 780.0        # 13 min, inside the documented 10-15 min range
    p_set_bar: float = 18.0       # inside 16-19 bar steam range (brief-verified)
    t_ramp_s: float = 90.0        # ramp time constant regime below
    tau_ramp_s: float = 25.0      # first-order fill time constant (plausible, NOT measured)
    t_release_s: float = 60.0     # release duration at end of cycle
    tau_release_s: float = 12.0   # first-order vent time constant (plausible, NOT measured)
    temp_set_C: float = 195.0     # inside 180-210 degC documented range
    tau_temp_s: float = 60.0      # thermal lag (slower than pressure, plausible)
    noise_bar: float = 0.05       # sensor noise std
    dt_s: float = 1.0             # 1 Hz sampling


def nominal_pressure_ode(p: np.ndarray, t: np.ndarray, params: CuringCycleParams,
                         leak_rate: float = 0.0, t_leak_s: float | None = None) -> np.ndarray:
    """Right-hand side dP/dt of the cycle ODE, vectorized over time.

    This same RHS is what the training loss re-implements in torch as the
    PINN residual: residual = dP_hat/dt - rhs(P_hat, t).
    """
    p = np.asarray(p, dtype=float)
    t = np.asarray(t, dtype=float)
    t_rel_start = params.cycle_s - params.t_release_s

    dpdt = np.zeros_like(p)
    ramp = t < params.t_ramp_s
    hold = (t >= params.t_ramp_s) & (t < t_rel_start)
    release = t >= t_rel_start

    dpdt[ramp] = (params.p_set_bar - p[ramp]) / params.tau_ramp_s
    dpdt[hold] = 0.0
    dpdt[release] = -p[release] / params.tau_release_s

    if leak_rate > 0.0 and t_leak_s is not None:
        leaking = hold & (t >= t_leak_s)
        dpdt[leaking] += -leak_rate * p[leaking]
    return dpdt


def phase_masks(t: np.ndarray, params: CuringCycleParams) -> dict[str, np.ndarray]:
    """Boolean masks for ramp / hold / release given timestamps [s]."""
    t = np.asarray(t, dtype=float)
    t_rel_start = params.cycle_s - params.t_release_s
    return {
        "ramp": t < params.t_ramp_s,
        "hold": (t >= params.t_ramp_s) & (t < t_rel_start),
        "release": t >= t_rel_start,
    }
