"""Physics-informed losses, one block per head. All torch, all differentiable.

total_loss(head) = data_loss + lambda_phys * physics_residual

Each physics term is the torch re-statement of an equation documented in
pinn/physics/ — see those modules for sources and justification.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

# ---------------------------------------------------------------- vibration --
def vibration_loss(out: dict, y: torch.Tensor, band_energies: torch.Tensor,
                   lambda_phys: float = 0.1) -> dict:
    """CE on 4 classes + spectral-consistency physics term.

    Physics term: for windows labeled faulty, the model's relative belief
    among {IR, OR, B} should match the relative envelope-spectrum energy
    observed at {BPFI, BPFO, BSF} (bearing kinematics, pinn.physics.bearing).
    Soft cross-entropy between the two distributions.
    Head class order is [Normal, IR, OR, B]; band order is [BPFO, BPFI, BSF]
    -> faults [IR, OR, B] correspond to bands [BPFI, BPFO, BSF] = indices [1, 0, 2].
    """
    logits = out["fault_logits"]
    data = F.cross_entropy(logits, y)

    faulty = y > 0
    if faulty.any():
        fault_logits = logits[faulty][:, 1:]                      # [n, 3] IR,OR,B
        e = band_energies[faulty][:, [1, 0, 2]]                   # match order
        band_p = (e + 1e-6) / (e + 1e-6).sum(dim=1, keepdim=True)
        phys = -(band_p * F.log_softmax(fault_logits, dim=1)).sum(dim=1).mean()
    else:
        phys = torch.zeros((), device=logits.device)
    return {"data": data, "phys": phys, "total": data + lambda_phys * phys}


# ------------------------------------------------------------ thermal/power --
def _soft_hdf(raw: torch.Tensor) -> torch.Tensor:
    """Differentiable AI4I HDF rule. raw columns: [air_K, proc_K, rpm, torque, wear]."""
    dT = raw[:, 1] - raw[:, 0]
    return torch.sigmoid((8.6 - dT) / 0.5) * torch.sigmoid((1380.0 - raw[:, 2]) / 20.0)


def _soft_pwf(raw: torch.Tensor) -> torch.Tensor:
    """Differentiable AI4I PWF rule: power outside [3500, 9000] W."""
    power = raw[:, 3] * raw[:, 2] * (2.0 * torch.pi / 60.0)
    return torch.clamp(torch.sigmoid((3500.0 - power) / 100.0)
                       + torch.sigmoid((power - 9000.0) / 100.0), max=1.0)


THERMAL_POS_WEIGHT = torch.tensor([28.0, 85.0, 104.0])  # ~neg/pos ratios in AI4I


def thermal_loss(out: dict, y: torch.Tensor, raw: torch.Tensor,
                 lambda_phys: float = 0.5) -> dict:
    """Weighted BCE on [MF, HDF, PWF] + rule-consistency physics term.

    Physics term: the HDF/PWF probability predicted by the head must agree
    with the soft (sigmoid-relaxed) closed-form rules evaluated on the raw
    physical inputs (pinn.physics.ai4i_rules — the dataset's own documented
    generative physics).
    """
    logits = out["failure_logits"]
    data = F.binary_cross_entropy_with_logits(
        logits, y, pos_weight=THERMAL_POS_WEIGHT.to(logits.device))
    p = torch.sigmoid(logits)
    phys = F.mse_loss(p[:, 1], _soft_hdf(raw)) + F.mse_loss(p[:, 2], _soft_pwf(raw))
    return {"data": data, "phys": phys, "total": data + lambda_phys * phys}


# -------------------------------------------------------------------- RUL ----
def rul_loss(out: dict, y_seq: torch.Tensor, lambda_phys: float = 0.1,
             rul_scale: float = 125.0) -> dict:
    """MSE on per-timestep RUL + degradation-monotonicity physics term.

    Physics term (pinn.physics.rul), piecewise like the target itself:
    - below the cap (degradation observable): RUL[t+1] - RUL[t] must be -1
      (irreversible damage, cycle clock);
    - in the capped early-life region the true slope is 0, so forcing -1
      there would fight the data term — we only require non-INCREASE
      (relu(diff)^2), i.e. RUL never goes back up.
    Both terms normalized by the cap so they share scale.
    """
    rul = out["rul_seq"] / rul_scale
    target = y_seq / rul_scale
    data = F.mse_loss(rul, target)
    diffs = (out["rul_seq"][:, 1:] - out["rul_seq"][:, :-1]) / rul_scale
    below_cap = (y_seq[:, 1:] < rul_scale - 0.5).float()
    phys = (below_cap * (diffs + 1.0 / rul_scale) ** 2
            + (1.0 - below_cap) * F.relu(diffs) ** 2).mean()
    return {"data": data, "phys": phys, "total": data + lambda_phys * phys}


# --------------------------------------------------------------- pressure ----
def pressure_loss(out: dict, batch: dict, lambda_phys: float = 0.2,
                  lambda_recon: float = 1.0) -> dict:
    """[SYNTHETIC-data head] anomaly BCE + healthy-cycle recon + ODE residual.

    - recon MSE only on NOMINAL cycles: the head learns the healthy trace.
    - ODE residual (pinn.physics.curing_press) on the reconstruction for ALL
      cycles, using each cycle's known generation params (training-time
      physics): ramp dP/dt=(P_set-P)/tau_r, hold dP/dt=0, release dP/dt=-P/tau_rel.
      Finite differences on the token grid (dt_token = 8 * dt_grid).
    """
    recon = out["pressure_recon"]                       # [B, K] tokens
    frame = batch["frame"]                              # 8
    p_obs = batch["X"][:, :, 0]                         # [B, seq]
    k = recon.shape[1]
    p_obs_tok = p_obs[:, : k * frame].reshape(p_obs.shape[0], k, frame).mean(dim=2)
    nominal = batch["y"] == 0

    data_anom = F.binary_cross_entropy_with_logits(out["anomaly_logit"], batch["y"])
    data_recon = F.mse_loss(recon[nominal], p_obs_tok[nominal]) if nominal.any() \
        else torch.zeros((), device=recon.device)

    dt_tok = batch["dt"].unsqueeze(1) * frame           # [B,1] seconds per token
    dpdt = (recon[:, 1:] - recon[:, :-1]) / dt_tok
    def tok_mask(m):
        mt = m[:, : k * frame].reshape(m.shape[0], k, frame).float().mean(dim=2) > 0.5
        return mt[:, :-1]
    rhs = torch.zeros_like(dpdt)
    ramp, hold, rel = tok_mask(batch["mask_ramp"]), tok_mask(batch["mask_hold"]), tok_mask(batch["mask_release"])
    p_left = recon[:, :-1]
    rhs = torch.where(ramp, (batch["p_set"].unsqueeze(1) - p_left) / batch["tau_ramp"].unsqueeze(1), rhs)
    rhs = torch.where(rel, -p_left / batch["tau_release"].unsqueeze(1), rhs)
    # hold: rhs stays 0
    phys = F.mse_loss(dpdt, rhs)

    total = data_anom + lambda_recon * data_recon + lambda_phys * phys
    return {"data": data_anom + data_recon, "phys": phys, "total": total}
