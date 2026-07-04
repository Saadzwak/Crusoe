"""Closed-form failure rules of the AI4I 2020 dataset (thermal/power head).

Source: the dataset's own documentation (S. Matzka, "Explainable Artificial
Intelligence for Predictive Maintenance Applications", 2020; UCI repository
dataset #601). AI4I is *synthetic by design* and the authors publish the exact
generative rules — which is precisely why it is useful to a physics-informed
head: the "physics" of this dataset is known in closed form, so we can
penalize the model for disagreeing with the very rules that generated the
labels.

Rules (as documented by the authors):

- HDF (heat dissipation failure):
      (T_process - T_air) < 8.6 K   AND   rotational speed < 1380 rpm
  Physical reading: heat dissipation from the tool depends on the temperature
  gradient to ambient (Newton's law of cooling, q = h*A*dT) and on convection
  driven by rotation; low gradient + low speed = insufficient cooling.

- PWF (power failure):
      P = torque * omega,  omega = 2*pi*rpm/60  [rad/s]
      fails if P < 3500 W or P > 9000 W
  Physical reading: mechanical power transmitted by a rotating shaft is
  exactly tau*omega; the process only works inside a power window.

- OSF (overstrain failure):
      tool_wear [min] * torque [Nm] > 11000 (L) / 12000 (M) / 13000 (H)
  Physical reading: a worn tool under high torque overstrains.

- TWF (tool wear failure): tool replaced/fails at a wear time drawn uniformly
  in 200-240 min (stochastic — NOT usable as a deterministic constraint).
- RNF (random failure): 0.1% chance regardless of inputs (pure noise — same).

The thermal/power head is therefore physics-constrained on HDF and PWF (and
optionally OSF), the deterministic rules; TWF/RNF remain data-only.
"""

from __future__ import annotations

import math

import numpy as np

HDF_DELTA_T_K = 8.6        # [K]   documented HDF temperature-difference threshold
HDF_RPM = 1380.0           # [rpm] documented HDF speed threshold
PWF_LOW_W = 3500.0         # [W]   documented power window
PWF_HIGH_W = 9000.0        # [W]
OSF_MINNM = {"L": 11000.0, "M": 12000.0, "H": 13000.0}  # [min*Nm] per product type


def rpm_to_rad_s(rpm):
    return 2.0 * math.pi * np.asarray(rpm, dtype=float) / 60.0


def ai4i_hard_rules(air_temp_K, process_temp_K, rpm, torque_Nm, tool_wear_min,
                    product_type: str | np.ndarray = "L") -> dict[str, np.ndarray]:
    """Boolean (0/1) rule evaluations, exactly as documented. numpy, non-differentiable.

    Used for (a) validating the dataset against its own documentation and
    (b) building rule-based pseudo-labels. The differentiable soft versions
    used in the training loss live in `pinn.losses`.
    """
    air = np.asarray(air_temp_K, dtype=float)
    proc = np.asarray(process_temp_K, dtype=float)
    speed = np.asarray(rpm, dtype=float)
    tau = np.asarray(torque_Nm, dtype=float)
    wear = np.asarray(tool_wear_min, dtype=float)

    power = tau * rpm_to_rad_s(speed)
    hdf = ((proc - air) < HDF_DELTA_T_K) & (speed < HDF_RPM)
    pwf = (power < PWF_LOW_W) | (power > PWF_HIGH_W)

    if isinstance(product_type, str):
        osf_thr = np.full_like(wear, OSF_MINNM[product_type])
    else:
        osf_thr = np.array([OSF_MINNM[t] for t in product_type], dtype=float)
    osf = (wear * tau) > osf_thr

    return {"HDF": hdf.astype(int), "PWF": pwf.astype(int), "OSF": osf.astype(int),
            "power_W": power}
