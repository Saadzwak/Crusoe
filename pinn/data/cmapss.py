"""C-MAPSS FD001 loader (degradation/RUL head — SIMULATED run-to-failure).

Space-separated, no header, 26 columns:
    unit, cycle, op_setting_1..3, sensor_1..21
Verified: train 20,631 rows (100 units), test 13,096 rows, RUL 100 rows.

Preprocessing (standard for FD001, per team brief + Heimes 2008 convention):
- drop the 7 constant columns (op_setting_3, sensors 1,5,10,16,18,19) -> 14
  informative sensors remain
- z-score with *train* statistics
- sliding windows of `window` consecutive cycles per unit
- target: piecewise-linear RUL capped at 125, at EVERY timestep of the window
  (the per-timestep targets enable the monotonicity physics loss).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pinn.physics.rul import RUL_CAP_FD001, estimate_knee_rul, piecewise_linear_rul

COLUMNS = (["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
           + [f"sensor_{i}" for i in range(1, 22)])
# The brief's 7 columns are exactly constant in FD001 (verified: std == 0).
# sensor_6 is additionally dropped: near-constant (verified: std ~= 0.0014,
# no signal), which is what makes the conventional "14 informative sensors"
# count work out (the brief's list alone would leave 15). Flagged in NOTES.
DROP = ["op_setting_3", "sensor_1", "sensor_5", "sensor_6", "sensor_10",
        "sensor_16", "sensor_18", "sensor_19"]
FEATURES = [c for c in COLUMNS if c.startswith("sensor_") and c not in DROP]     # 14 sensors
assert len(FEATURES) == 14, f"expected 14 informative sensors, got {len(FEATURES)}"


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = COLUMNS
    return df


def load(root: str | Path = "data/cmapss", window: int = 30, cap: float = RUL_CAP_FD001):
    """Return train windows/targets + full test trajectories + true test RUL."""
    root = Path(root)
    train = _read(root / "train_FD001.txt")
    test = _read(root / "test_FD001.txt")
    rul_true = np.loadtxt(root / "RUL_FD001.txt")

    mean = train[FEATURES].mean()
    std = train[FEATURES].std() + 1e-8

    X, Y, KNEE = [], [], []
    for unit, g in train.groupby("unit"):
        g = g.sort_values("cycle")
        feats = ((g[FEATURES] - mean) / std).to_numpy(dtype=np.float32)
        n_total = int(g["cycle"].max())
        rul = piecewise_linear_rul(g["cycle"].to_numpy(), n_total=n_total,
                                   cap=cap).astype(np.float32)
        # Per-unit degradation knee (Step-7): orient sensors so end-of-life
        # drift is positive, average -> composite health, two-segment fit.
        sign = np.sign(feats[-15:].mean(axis=0) - feats[:15].mean(axis=0))
        knee = min(cap, estimate_knee_rul((feats * sign).mean(axis=1)))
        rul_uncapped = (n_total - g["cycle"].to_numpy()).astype(np.float32)
        below_knee = (rul_uncapped < knee)
        for s in range(0, len(g) - window + 1):
            X.append(feats[s:s + window])
            Y.append(rul[s:s + window])
            KNEE.append(below_knee[s:s + window])
    X = np.stack(X)                     # [N, window, 14]
    Y = np.stack(Y)                     # [N, window] per-timestep RUL target
    KNEE = np.stack(KNEE)               # [N, window] slope -1 enforceable here

    # Test: last `window` cycles of each unit (padded by repetition if shorter),
    # target = provided true RUL at the last observed cycle (capped).
    Xt, Yt = [], []
    for i, (unit, g) in enumerate(test.groupby("unit")):
        g = g.sort_values("cycle")
        feats = ((g[FEATURES] - mean) / std).to_numpy(dtype=np.float32)
        if len(feats) < window:
            pad = np.repeat(feats[:1], window - len(feats), axis=0)
            feats = np.concatenate([pad, feats], axis=0)
        Xt.append(feats[-window:])
        Yt.append(min(cap, float(rul_true[i])))
    return {
        "X_train": X, "Y_train": Y, "knee_mask": KNEE,
        "X_test": np.stack(Xt), "rul_test": np.asarray(Yt, dtype=np.float32),
        "scaler": (mean.to_numpy(), std.to_numpy()), "features": FEATURES,
        "n_units_train": train["unit"].nunique(), "n_units_test": test["unit"].nunique(),
    }
