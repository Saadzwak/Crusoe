"""Remaining Useful Life (RUL) structure — degradation head (C-MAPSS FD001).

Two pieces of domain structure are used, both standard in the PHM literature:

1. Piecewise-linear RUL target (Heimes 2008, PHM data challenge; used by
   virtually all C-MAPSS work since): early in life the engine shows no
   measurable degradation, so 'true' RUL is unobservable from sensors and is
   capped at a constant; past the knee it decreases linearly to 0 at failure:

       RUL(cycle) = min(RUL_max, n_cycles_total - cycle)

   with RUL_max = 125 cycles (the conventional value for FD001).

2. Degradation monotonicity as the physics constraint. By definition of a
   run-to-failure trajectory measured in cycles,

       d(RUL)/d(cycle) = -1   (exactly, once degradation is observable)

   Damage accumulation is irreversible (no repair happens mid-trajectory in
   C-MAPSS), so a physically-consistent RUL estimate along a window of
   consecutive cycles must decrease by ~1 per cycle. The head predicts RUL at
   *every* timestep of its input window; the loss penalizes deviation of the
   finite difference RUL_hat[t+1] - RUL_hat[t] from -1. This is the legitimate
   'physics' of the quantity itself — not thermodynamics of the turbofan, but
   the defining dynamics of irreversible damage measured in cycle time.

Note honestly flagged: unlike the bearing-frequency or pressure-ODE
constraints, this one is a *structural* prior on the target rather than a law
of nature about the sensors. It is still physics-informed in the accepted
PINN-for-PHM sense (see e.g. surveys of physics-informed RUL estimation), and
it measurably regularizes noisy per-window RUL estimates.
"""

from __future__ import annotations

import numpy as np

RUL_CAP_FD001 = 125.0  # cycles — conventional piecewise-linear cap for FD001

# The 7 constant/no-signal columns in FD001, per the team brief (verified
# during exploration: their std over the whole train set is ~0).
FD001_DROP_COLUMNS = ["op_setting_3", "sensor_1", "sensor_5", "sensor_10",
                      "sensor_16", "sensor_18", "sensor_19"]


def piecewise_linear_rul(cycles: np.ndarray, n_total: int,
                         cap: float = RUL_CAP_FD001) -> np.ndarray:
    """RUL target for one unit given its total lifetime in cycles."""
    cycles = np.asarray(cycles, dtype=float)
    return np.minimum(cap, n_total - cycles)
