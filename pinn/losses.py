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


# ---------------------------------------------------------------- fatigue ----
def fatigue_loss(out: dict, d_target: torch.Tensor, order: torch.Tensor,
                 lambda_phys: float = 0.3) -> dict:
    """MSE on Miner damage D + within-batch monotonicity physics term.

    Physics (pinn.physics.fatigue): damage is irreversible — for two windows
    of the SAME run-to-failure test, the later one cannot have lower damage.
    `order` is the chronological index of each window; the penalty is
    relu(D_early - D_late) over all ordered pairs in the batch (the batch
    comes from one test set in v1, so all pairs are comparable).
    """
    d = out["damage"]
    data = F.mse_loss(d, d_target)
    idx = torch.argsort(order)
    d_sorted = d[idx]
    viol = F.relu(d_sorted[:-1].unsqueeze(1) - d_sorted[1:].unsqueeze(0))
    # only pairs (i earlier than j): upper triangle of the sorted matrix
    mask = torch.triu(torch.ones_like(viol), diagonal=0)
    phys = (viol * mask).sum() / mask.sum().clamp(min=1.0)
    return {"data": data, "phys": phys, "total": data + lambda_phys * phys}


# ---------------------------------------------------- virtual sensor: AI4I ---
def thermal_recon_loss(out: dict, temp_true_K: torch.Tensor, raw: torch.Tensor,
                       hdf_label: torch.Tensor, lambda_phys: float = 0.3,
                       temp_scale: float = 2.0) -> dict:
    """VIRTUAL SENSOR loss: reconstruct withheld process temperature.

    Data term: MSE in kelvin (scaled by ~the dataset's temp std, 2 K).
    Physics term: the HDF rule (pinn.physics.ai4i_rules) evaluated with the
    RECONSTRUCTED temperature must reproduce the row's true HDF label — the
    documented failure physics is what pins the reconstruction where plain
    regression would drift.
    """
    t_hat = out["process_temp_K"]
    data = F.mse_loss(t_hat / temp_scale, temp_true_K / temp_scale)
    dT_hat = t_hat - raw[:, 0]                       # reconstructed - air temp
    hdf_soft = torch.sigmoid((8.6 - dT_hat) / 0.5) \
        * torch.sigmoid((1380.0 - raw[:, 2]) / 20.0)
    phys = F.binary_cross_entropy(hdf_soft.clamp(1e-6, 1 - 1e-6), hdf_label)
    return {"data": data, "phys": phys, "total": data + lambda_phys * phys}


# ---------------------------------------------------- virtual sensor: CWRU ---
def fe_recon_loss(out: dict, fe_bands_true: torch.Tensor, fe_rms_true: torch.Tensor,
                  de_bands: torch.Tensor, lambda_phys: float = 0.2) -> dict:
    """VIRTUAL SENSOR loss: reconstruct withheld fan-end features from drive-end.

    Data terms: MSE against the REAL (withheld) fan-end measurements.
    Physics term: both accelerometers watch the same shaft/fault — the
    RELATIVE distribution of predicted fan-end band energies must match the
    drive-end's (bearing kinematics puts the same characteristic frequencies
    in both, amplitudes differ with transmission path).
    """
    data = F.mse_loss(out["fe_bands"], fe_bands_true) \
        + F.mse_loss(out["fe_rms"], fe_rms_true)
    p_fe = (out["fe_bands"] + 1e-6) / (out["fe_bands"] + 1e-6).sum(dim=1, keepdim=True)
    p_de = (de_bands + 1e-6) / (de_bands + 1e-6).sum(dim=1, keepdim=True)
    phys = F.kl_div(p_fe.log(), p_de, reduction="batchmean")
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

    # Mild pos_weight (~neg/pos at 35% anomalous). The real degeneracy fix is
    # architectural — the head classifies from the physics residual, see
    # PressureHead — 2.5 overshot to all-anomalous in the second v1 run.
    data_anom = F.binary_cross_entropy_with_logits(
        out["anomaly_logit"], batch["y"],
        pos_weight=torch.tensor(1.8, device=recon.device))
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

    # Final degree of cure (SYNTHETIC target from Kamal-Sourour integration).
    # alpha_end lives in ~[0.73, 0.99]; x10 scaling puts its MSE on the same
    # order as the other terms (magnitude-balanced fixed weighting, see docs).
    # Depth-weighted: nominal cycles cluster at ~0.985 and dominated the plain
    # MSE, letting the head miss the rare DEEP under-cures (0/3 caught in the
    # first v1 run) — the one output that must never be missed. Weight grows
    # ~1 (fully cured) -> ~9 (alpha 0.73).
    a_true = batch["alpha_end"]
    w = 1.0 + 30.0 * (0.99 - a_true).clamp(min=0.0)
    data_alpha = (w * (out["alpha_end"] * 10.0 - a_true * 10.0) ** 2).mean()

    total = data_anom + lambda_recon * data_recon + data_alpha + lambda_phys * phys
    return {"data": data_anom + data_recon + data_alpha, "phys": phys, "total": total}
