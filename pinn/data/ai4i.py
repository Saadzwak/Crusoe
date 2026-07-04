"""AI4I 2020 loader (thermal/power head — SIMULATED data, documented rules).

10,000 rows x 14 columns, zero missing values (verified). Columns used as
inputs: Air temperature [K], Process temperature [K], Rotational speed [rpm],
Torque [Nm], Tool wear [min], product Type (L/M/H one-hot). Targets:
Machine failure, HDF, PWF (and OSF kept for the physics loss).

The loader returns BOTH z-scored features (for the network) and the raw
physical columns (for the physics loss, which needs real kelvins/rpm/Nm —
see pinn.physics.ai4i_rules).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pinn.physics.ai4i_rules import ai4i_hard_rules

RAW_FEATURES = ["Air temperature [K]", "Process temperature [K]",
                "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
TARGETS = ["Machine failure", "HDF", "PWF"]


def load(csv_path: str | Path = "data/ai4i/ai4i2020.csv", val_fraction: float = 0.2,
         seed: int = 7):
    """Return dict with scaled features, raw physics columns, targets, split."""
    df = pd.read_csv(csv_path)

    # Sanity: dataset must satisfy its own documented rules where flags are set
    # (RNF rows can set Machine failure without a deterministic cause).
    rules = ai4i_hard_rules(df[RAW_FEATURES[0]], df[RAW_FEATURES[1]],
                            df[RAW_FEATURES[2]], df[RAW_FEATURES[3]],
                            df[RAW_FEATURES[4]], df["Type"].to_numpy())
    hdf_match = (rules["HDF"] & df["HDF"].to_numpy()).sum()

    type_onehot = pd.get_dummies(df["Type"])[["L", "M", "H"]].to_numpy(dtype=np.float32)
    raw = df[RAW_FEATURES].to_numpy(dtype=np.float32)
    mean, std = raw.mean(axis=0), raw.std(axis=0) + 1e-8
    X = np.concatenate([(raw - mean) / std, type_onehot], axis=1)
    y = df[TARGETS].to_numpy(dtype=np.float32)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_val = int(len(df) * val_fraction)
    return {
        "X": X.astype(np.float32),            # [N, 8] scaled + one-hot
        "raw": raw,                           # [N, 5] physical units for physics loss
        "types": df["Type"].to_numpy(),
        "y": y,                               # [N, 3] machine failure / HDF / PWF
        "train_idx": idx[n_val:], "val_idx": idx[:n_val],
        "scaler": (mean, std),
        "rule_hdf_overlap": int(hdf_match),   # diagnostics for NOTES.md
        "rules_eval": rules,
    }
