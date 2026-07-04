"""Physics knowledge used by the MH-PINN, one module per head.

Each module states the governing equation(s), where they come from, and why
they are a legitimate physical constraint for that head. Everything here is
plain numpy/python (no torch): these are the *equations*; how they enter the
training loss lives in `pinn.losses`.
"""

from pinn.physics.bearing import (  # noqa: F401
    BearingGeometry,
    CWRU_DRIVE_END_6205,
    CWRU_FAN_END_6203,
    IMS_REXNORD_ZA2115,
    fault_frequencies,
)
from pinn.physics.ai4i_rules import ai4i_hard_rules  # noqa: F401
from pinn.physics.curing_press import CuringCycleParams, nominal_pressure_ode  # noqa: F401
from pinn.physics.rul import piecewise_linear_rul  # noqa: F401
