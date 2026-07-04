"""Bearing fault characteristic frequencies (vibration head).

Governing equations — classical rolling-element bearing kinematics (derived
from the rolling-without-slipping condition between rolling elements and
raceways; standard references: Randall & Antoni, "Rolling element bearing
diagnostics — a tutorial", MSSP 2011):

    BPFO = (n / 2) * f_r * (1 - (d / D) * cos(phi))     # outer race defect
    BPFI = (n / 2) * f_r * (1 + (d / D) * cos(phi))     # inner race defect
    BSF  = (D / (2 d)) * f_r * (1 - ((d / D) cos(phi))^2)  # ball spin
    FTF  = (f_r / 2) * (1 - (d / D) * cos(phi))         # cage (train) freq

where n = number of rolling elements, f_r = shaft rotation frequency [Hz],
d = rolling-element diameter, D = pitch diameter, phi = contact angle.

Why this is a legitimate physics constraint: a localized defect on a race or
ball is struck once per pass of each rolling element, so it injects periodic
impulses at exactly these frequencies (and harmonics, with sidebands at f_r
for inner-race faults). The fault *class* therefore has a closed-form
signature in the frequency domain that depends only on published bearing
geometry and shaft speed — not on learned parameters. The vibration head can
be regularized to agree with the energy observed in those bands.

Geometry sources (NOT invented — cross-checked, see `verify_against_published`):
- CWRU drive end: SKF 6205-2RS JEM, from the CWRU Bearing Data Center
  "Apparatus & Procedures" page.
- CWRU fan end: SKF 6203-2RS JEM, same source.
- IMS: Rexnord ZA-2115 double-row bearing, from the IMS/University of
  Cincinnati dataset readme (Qiu et al. 2006, J. Sound & Vibration).

Confidence note: the geometric constants below reproduce the fault-frequency
multipliers *published by the dataset authors themselves* to ~4 significant
figures (run `verify_against_published()`), which is strong evidence they are
correct. Still, they were entered from documentation, not measured by us —
if a demo depends on an exact number, re-check the dataset pages.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BearingGeometry:
    """Geometry of one rolling-element bearing, in consistent units (inches)."""

    name: str
    n_elements: int          # number of rolling elements (per row)
    d_element: float         # rolling element diameter [in]
    d_pitch: float           # pitch diameter [in]
    contact_angle_deg: float # contact angle phi [deg]
    # Fault-frequency multipliers (x shaft frequency) as published by the
    # dataset authors, used only to cross-check our formula + geometry.
    published_multipliers: dict | None = None


# CWRU Bearing Data Center, "Apparatus & Procedures".
# Motor shaft speeds in the dataset: ~1797/1772/1750/1730 rpm (0..3 HP load).
CWRU_DRIVE_END_6205 = BearingGeometry(
    name="CWRU drive end — SKF 6205-2RS JEM",
    n_elements=9,
    d_element=0.3126,
    d_pitch=1.537,
    contact_angle_deg=0.0,
    published_multipliers={"BPFI": 5.4152, "BPFO": 3.5848, "FTF": 0.39828, "BSF": 2.3567},
)

CWRU_FAN_END_6203 = BearingGeometry(
    name="CWRU fan end — SKF 6203-2RS JEM",
    n_elements=8,
    d_element=0.2656,
    d_pitch=1.122,
    contact_angle_deg=0.0,
    published_multipliers={"BPFI": 4.9469, "BPFO": 3.0530, "FTF": 0.38167, "BSF": 1.9938},
)

# IMS readme (Qiu et al. 2006): shaft held at 2000 rpm; published characteristic
# frequencies BPFO ~236.4 Hz and BPFI ~296.9 Hz at that speed.
IMS_REXNORD_ZA2115 = BearingGeometry(
    name="IMS — Rexnord ZA-2115 double row",
    n_elements=16,
    d_element=0.331,
    d_pitch=2.815,
    contact_angle_deg=15.17,
    published_multipliers={"BPFO": 236.4 / (2000 / 60), "BPFI": 296.9 / (2000 / 60)},
)


def fault_frequencies(geom: BearingGeometry, shaft_hz: float) -> dict[str, float]:
    """Return {BPFO, BPFI, BSF, FTF} in Hz for a shaft speed in Hz."""
    ratio = (geom.d_element / geom.d_pitch) * math.cos(math.radians(geom.contact_angle_deg))
    n = geom.n_elements
    return {
        "BPFO": (n / 2.0) * shaft_hz * (1.0 - ratio),
        "BPFI": (n / 2.0) * shaft_hz * (1.0 + ratio),
        "BSF": (geom.d_pitch / (2.0 * geom.d_element)) * shaft_hz * (1.0 - ratio**2),
        "FTF": (shaft_hz / 2.0) * (1.0 - ratio),
    }


def verify_against_published(rtol: float = 5e-3) -> list[str]:
    """Cross-check geometry + formulas against dataset-published multipliers.

    Returns a list of human-readable check results; raises AssertionError on
    mismatch beyond `rtol`. Run at import time of the training script so a
    wrong constant can never silently enter training.
    """
    results = []
    for geom in (CWRU_DRIVE_END_6205, CWRU_FAN_END_6203, IMS_REXNORD_ZA2115):
        freqs_at_1hz = fault_frequencies(geom, shaft_hz=1.0)
        for key, published in (geom.published_multipliers or {}).items():
            computed = freqs_at_1hz[key]
            ok = abs(computed - published) / published <= rtol
            results.append(
                f"{geom.name}: {key} computed {computed:.4f} vs published {published:.4f} "
                f"-> {'OK' if ok else 'MISMATCH'}"
            )
            if not ok:
                raise AssertionError(results[-1])
    return results
