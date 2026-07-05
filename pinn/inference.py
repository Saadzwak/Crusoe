"""PinnRuntime — the TRAINED MH-PINN as the live physical-state source.

Loads pinn/models/mh_pinn_v2.pt once (CPU) and turns a C-MAPSS FD001 unit
replay into the PinnState fields the information layer consumes:

    health_index  = clamp(RUL_hat / 125, 0, 1)   (fraction of capped life left)
    rul_cycles    = RUL head prediction at the current cycle (30-cycle window)
    residual      = the model's own physics-consistency residual: mean squared
                    deviation of the predicted per-cycle slope from the
                    physical -1 in the sub-cap region (higher = worse), i.e.
                    exactly the quantity the training loss constrains
    failure_mode_probs = {"degradation_wearout": 1 - health} once observable

HONESTY BOUNDARY (stated, not hidden): only the RUL/degradation head applies
to this feed — the C-MAPSS replay carries turbofan-mapped scalar sensors, not
the vibration waveforms / AI4I features / pressure traces the other heads
were trained on. Those heads run in `pinn.train` evaluations, not here.
Normalization uses the SAME convention as training (z-score with full-train
statistics over the 14 informative sensors, sensor_6 dropped — see
docs/physics_heads.md §5).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

RUL_CAP = 125.0
WINDOW = 30


class PinnRuntime:
    def __init__(self, repo_root: str | Path | None = None, unit: int = 1):
        import pandas as pd
        import torch

        from pinn.data.cmapss import COLUMNS, FEATURES
        from pinn.model import MHPinn

        root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
        data = root / "pinn" / "data" / "train_FD001.txt"
        ckpt = root / "pinn" / "models" / "mh_pinn_v2.pt"

        df = pd.read_csv(data, sep=r"\s+", header=None, names=COLUMNS)
        mean = df[FEATURES].mean()
        std = df[FEATURES].std() + 1e-8
        g = df[df["unit"] == unit].sort_values("cycle")
        if g.empty:
            raise ValueError(f"FD001 unit {unit} absent de {data}")
        self._feats = ((g[FEATURES] - mean) / std).to_numpy(dtype=np.float32)
        self.n_cycles = len(self._feats)
        self.unit = unit

        self._torch = torch
        self.model = MHPinn()
        self.model.load_state_dict(torch.load(ckpt, weights_only=True))
        self.model.eval()

    def infer_at_cycle(self, cycle: int) -> dict:
        """PinnState fields from the trained RUL head at a given unit cycle."""
        torch = self._torch
        end = int(max(1, min(cycle, self.n_cycles)))
        w = self._feats[max(0, end - WINDOW):end]
        if len(w) < WINDOW:                      # early life: repeat-pad the front
            w = np.concatenate([np.repeat(w[:1], WINDOW - len(w), axis=0), w])
        with torch.no_grad():
            rul_seq = self.model("rul", torch.from_numpy(w[None]))["rul_seq"][0]
        rul = float(rul_seq[-1])
        diffs = rul_seq[1:] - rul_seq[:-1]
        below = rul_seq[1:] < (RUL_CAP - 0.5)
        if bool(below.any()):
            residual = float(((diffs[below] + 1.0) ** 2).mean())
        else:                                     # capped region: only non-increase
            residual = float((torch.relu(diffs) ** 2).mean())
        health = max(0.0, min(1.0, rul / RUL_CAP))
        urgency = round(1.0 - health, 3)
        return {
            "health_index": round(health, 3),
            "rul_cycles": round(rul, 1),
            "residual": round(residual, 4),
            "failure_mode_probs": ({"degradation_wearout": urgency}
                                   if urgency > 0.15 else {}),
        }
