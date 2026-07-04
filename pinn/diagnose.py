"""Step-6 diagnostic: is the model leaning on data more than physics?

Three instrumented answers (no assumptions):

A. Per head: fraction of the total loss that is lambda*physics at the final
   epoch (from runs/v1/history.json) AND a live gradient probe — backward the
   data term and the lambda*physics term SEPARATELY on one validation batch
   and measure the gradient norm each induces on the SHARED BODY parameters.
   "Does the physics term actually receive gradients?" answered directly.

B. Vibration "static physics term": the term is a cross-entropy against SOFT
   targets (envelope band-energy distributions), so its minimum is not 0 but
   the mean entropy H(band_p) of those targets. We report the observed term,
   the entropy floor, and their difference KL(band_p || model_p) — the part
   that can actually reach 0 — plus the pseudo-label quality (how often the
   dominant band agrees with the true fault class).

C. RUL lambda sweep + monotone-by-construction head: run separately via
     python -m pinn.train --heads rul --rul-lambda {0,0.5,5} [--rul-monotone]
   and compared in the synthesis doc.

D. Pressure under-cure: compute the PHYSICS FEATURE the model was never
   given — the integral of saturation-temperature deficit over the hold phase
   (3.2 K/bar x pressure deficit, steam tables; cure loss is ~proportional) —
   and check (i) how well that single feature explains true alpha_end on
   leaky cycles, (ii) whether the head's alpha_hat is compressed toward the
   nominal cluster (regression slope < 1 on leaky cycles).

Usage:  python -m pinn.diagnose          (expects runs/v1/mh_pinn_v1.pt)
Writes: runs/v1/diagnosis.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pinn import losses as L
from pinn.model import MHPinn
from pinn.train import (build_pressure, build_rul, build_thermal,
                        build_vibration, head_step, task_list)

RUNS = Path("runs/v1")

# lambda_phys actually used in training — keep in sync with pinn/losses.py
# defaults and train.py's RUL_LAMBDA_PHYS.
LAMBDAS = {"vibration": 0.1, "fe_recon": 0.2, "fatigue": 0.3, "thermal": 0.5,
           "thermal_recon": 0.3, "rul": 0.5, "pressure": 0.2}


def body_grad_norm(model: MHPinn) -> float:
    total = 0.0
    for p in model.body.parameters():
        if p.grad is not None:
            total += float((p.grad ** 2).sum())
    return float(np.sqrt(total))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--checkpoint", default=str(RUNS / "mh_pinn_v1.pt"))
    args = ap.parse_args()
    args.smoke = False  # builders expect the attribute

    print("== rebuilding datasets (same seeds as training) ==", flush=True)
    tasks = {"vibration": build_vibration(args), "thermal": build_thermal(args),
             "rul": build_rul(args), "pressure": build_pressure(args)}
    tasks = {k: v for k, v in tasks.items() if v is not None}

    model = MHPinn()
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    model.train()  # need grads; no optimizer step will be taken

    diagnosis: dict = {}

    # ---------------------------------------------------------------- A ----
    print("\n== A. loss fractions (final epoch) + gradient probe ==", flush=True)
    hist = json.loads((RUNS / "history.json").read_text())[-1]
    frac_table = {}
    for name, _src in task_list(tasks):
        lam = LAMBDAS[name]
        data_l, phys_l = hist.get(f"{name}_data"), hist.get(f"{name}_phys")
        if data_l is None or phys_l is None:
            continue
        frac = lam * phys_l / (data_l + lam * phys_l)
        frac_table[name] = {"data_loss": round(data_l, 5),
                            "phys_loss_raw": round(phys_l, 5),
                            "lambda": lam,
                            "phys_fraction_of_total": round(frac, 4)}
    grad_table = {}
    srcs = dict(task_list(tasks))
    for name, src in task_list(tasks):
        d = tasks[src]
        bidx = d["val"][:64]
        lam = LAMBDAS[name]
        terms = head_step(model, name, d, bidx)
        model.zero_grad()
        terms["data"].backward(retain_graph=True)
        g_data = body_grad_norm(model)
        model.zero_grad()
        (lam * terms["phys"]).backward()
        g_phys = body_grad_norm(model)
        grad_table[name] = {
            "body_grad_norm_data": round(g_data, 6),
            "body_grad_norm_lambda_phys": round(g_phys, 6),
            "phys_to_data_grad_ratio": round(g_phys / (g_data + 1e-12), 4),
            "phys_receives_gradient": bool(g_phys > 1e-9),
        }
        print(f"  {name:14s} |g_data|={g_data:.5f} |g_lam*phys|={g_phys:.5f} "
              f"ratio={g_phys / (g_data + 1e-12):.3f}")
    diagnosis["A_loss_fractions_final_epoch"] = frac_table
    diagnosis["A_gradient_probe_shared_body"] = grad_table

    # ---------------------------------------------------------------- B ----
    print("\n== B. vibration soft-target entropy floor ==", flush=True)
    d = tasks["vibration"]
    v = d["val"]
    faulty = d["y"][v] > 0
    e = d["bands"][v][faulty][:, [1, 0, 2]]           # order [IR, OR, B]
    band_p = (e + 1e-6) / (e + 1e-6).sum(dim=1, keepdim=True)
    floor = float(-(band_p * band_p.log()).sum(dim=1).mean())
    with torch.no_grad():
        out = model("vibration", d["X"][v])
        terms = L.vibration_loss(out, d["y"][v], d["bands"][v])
    observed = float(terms["phys"])
    # pseudo-label quality: dominant band -> class in {IR=1, OR=2, B=3}
    pseudo = band_p.argmax(dim=1) + 1
    agree = float((pseudo == d["y"][v][faulty]).float().mean())
    diagnosis["B_vibration_entropy_floor"] = {
        "observed_phys_term": round(observed, 4),
        "soft_target_entropy_floor": round(floor, 4),
        "achievable_KL_distance": round(observed - floor, 4),
        "pseudo_label_agreement_with_true_class": round(agree, 4),
        "reading": "the term cannot go below the floor; only the KL part is "
                   "optimizable — if KL is small the term IS optimized, its "
                   "plateau is the soft-target entropy, not a dead gradient",
    }
    print(f"  observed={observed:.4f} floor={floor:.4f} "
          f"KL={observed - floor:.4f} pseudo-label agreement={agree:.3f}")

    # ---------------------------------------------------------------- D ----
    print("\n== D. pressure: the physics feature the head never saw ==", flush=True)
    d = tasks["pressure"]
    v = d["val"]
    X, dt = d["X"][v], d["dt"][v]
    p_set = d["p_set"][v]
    hold = d["mask_hold"][v]
    y, a_true = d["y"][v], d["alpha_end"][v]
    deficit = torch.clamp(p_set.unsqueeze(1) - X[:, :, 0], min=0.0) * hold
    # [K*s] — proportional to lost cure (3.2 K/bar saturation slope x time)
    feature = (3.2 * deficit * dt.unsqueeze(1)).sum(dim=1)
    with torch.no_grad():
        a_hat = model("pressure", X)["alpha_end"]
    leaky = y > 0
    f_np = feature[leaky].numpy()
    at_np = a_true[leaky].numpy()
    ah_np = a_hat[leaky].numpy()
    corr_feat_true = float(np.corrcoef(f_np, at_np)[0, 1])
    corr_feat_pred = float(np.corrcoef(f_np, ah_np)[0, 1])
    slope, intercept = np.polyfit(at_np, ah_np, 1)
    diagnosis["D_pressure_under_cure"] = {
        "n_leaky_val_cycles": int(leaky.sum()),
        "corr_physics_feature_vs_true_alpha": round(corr_feat_true, 4),
        "corr_physics_feature_vs_predicted_alpha": round(corr_feat_pred, 4),
        "regression_slope_pred_vs_true_on_leaky": round(float(slope), 4),
        "reading": "|corr(feature, true)| ~ 1 means one computable physics "
                   "feature carries the answer; slope << 1 means the head "
                   "compresses toward the nominal cluster (regression-to-"
                   "mean), i.e. a wiring/calibration gap, not a lambda gap",
    }
    print(f"  corr(feature, alpha_true)={corr_feat_true:.4f}  "
          f"corr(feature, alpha_hat)={corr_feat_pred:.4f}  "
          f"pred-vs-true slope={slope:.3f}")

    (RUNS / "diagnosis.json").write_text(json.dumps(diagnosis, indent=2))
    print(f"\nSaved {RUNS / 'diagnosis.json'}")


if __name__ == "__main__":
    main()
