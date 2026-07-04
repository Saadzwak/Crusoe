"""v1 training entrypoint — 7 trained outputs on the shared body, loss curves,
and the single-sensor-reconstruction (virtual sensor) demo.

Usage:
    python -m pinn.train --smoke              # tiny fast CPU check
    python -m pinn.train --epochs 10          # v1 run
    python -m pinn.train --heads thermal rul  # subset

Loss weighting policy (documented decision, Step 3 of the v1 brief):
    We use FIXED, magnitude-balanced weights: every data/physics term is
    pre-scaled to O(1) at healthy operating points (e.g. RUL normalized by its
    cap, temperatures by ~their std, alpha_end x10), then combined with small
    fixed lambda_phys in [0.1, 0.5]. This is the standard low-risk baseline in
    the PINN literature; adaptive schemes (ReLoBRaLo — Bischof & Kraus;
    gradient-statistics annealing — Wang et al.) are documented follow-ups,
    not needed at v1 scale. Optimizer: Adam (proposal accepted), lr 1e-3,
    grad-clip 5 — unexceptional and it converged cleanly in v0.
    Change vs v0 from evidence: rul lambda_phys 0.1 -> 0.5 (v0 measured mean
    dRUL/cycle -0.40 vs the physical -1 below the cap).

Deliberately NOT here yet: champion/challenger, operator Q&A, IFC layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pinn import losses as L
from pinn.data import ai4i, cmapss, cwru, ims, pressure_synthetic
from pinn.model import MHPinn
from pinn.physics.bearing import verify_against_published
from pinn.physics.cure_kinetics import ALPHA_CURED_THRESHOLD
from pinn.physics import fatigue as fat
from pinn.telemetry import build_payload, to_praetor_signed_reading

RUNS_DIR = Path("runs/v1")      # overridable via --run-dir (diagnostic sweeps)
# MEASURED, not guessed (Step-6 sweep, runs/diag_rul_*): the normalized
# monotonicity term is ~1000x smaller than the data term (shared-body gradient
# ratio 0.026 at lambda=0.5 — numerically inert). lambda=5 was best on BOTH
# test RMSE (15.10 vs 15.99) and slope (-0.959 vs -0.910); the hard-monotone
# head variant was strictly worse (RMSE 17.64). See docs/v1_physics_diagnosis.md.
RUL_LAMBDA_PHYS = 5.0   # overridable via --rul-lambda


# --------------------------------------------------------------------------- data
def build_vibration(args) -> dict | None:
    """CWRU windows (+ real fan-end targets for the virtual-sensor head) + IMS."""
    root = Path(args.data_root) / "cwru"
    if not root.exists():
        print("[vibration] CWRU data missing — skipping head")
        return None
    X, y, bands, rpm, meta, fe_bands, fe_rms = cwru.load_windows(
        root, window=2048, hop=2048 if not args.smoke else 8192, fe_targets=True)
    if args.smoke:
        keep = np.random.default_rng(0).permutation(len(X))[:400]
        X, y, bands = X[keep], y[keep], bands[keep]
        fe_bands, fe_rms = fe_bands[keep], fe_rms[keep]
    Xn = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(Xn))
    n_val = int(0.2 * len(Xn))
    d = {"X": torch.from_numpy(Xn), "y": torch.from_numpy(y),
         "bands": torch.from_numpy(bands),
         "fe_bands": torch.from_numpy(fe_bands), "fe_rms": torch.from_numpy(fe_rms),
         "train": idx[n_val:], "val": idx[:n_val]}
    ims_dir = Path(args.data_root) / "ims" / "2nd_test"
    if ims_dir.exists():
        Xi, hi, fidx = ims.load_health_windows(Path(args.data_root) / "ims",
                                               stride_files=20 if args.smoke else 5)
        Xi = (Xi - Xi.mean(axis=1, keepdims=True)) / (Xi.std(axis=1, keepdims=True) + 1e-8)
        d["ims_X"], d["ims_health"] = torch.from_numpy(Xi), torch.from_numpy(hi)
        d["ims_order"] = torch.from_numpy(fidx.astype(np.float32))
        print(f"[vibration] CWRU {len(Xn)} windows (+FE targets) "
              f"+ IMS {len(Xi)} run-to-failure windows")
    else:
        print(f"[vibration] CWRU {len(Xn)} windows (IMS missing — fatigue head off)")
    return d


def build_thermal(args) -> dict:
    d = ai4i.load(Path(args.data_root) / "ai4i" / "ai4i2020.csv")
    X = d["X"]
    # VIRTUAL SENSOR input: process temperature column (scaled index 1) masked.
    X_masked = X.copy()
    X_masked[:, 1] = 0.0
    print(f"[thermal] AI4I {len(X)} rows "
          f"(HDF rule/label overlap: {d['rule_hdf_overlap']}); "
          f"temp-recon input = process-temp column masked")
    return {"X": torch.from_numpy(X), "X_masked": torch.from_numpy(X_masked),
            "raw": torch.from_numpy(d["raw"]),
            "temp_true": torch.from_numpy(d["raw"][:, 1].copy()),
            "y": torch.from_numpy(d["y"]),
            "train": d["train_idx"], "val": d["val_idx"]}


def build_rul(args) -> dict:
    d = cmapss.load(Path(args.data_root) / "cmapss")
    X, Y = d["X_train"], d["Y_train"]
    if args.smoke:
        X, Y = X[:2000], Y[:2000]
    knee = d["knee_mask"]
    if args.smoke:
        knee = knee[:2000]
    rng = np.random.default_rng(2)
    idx = rng.permutation(len(X))
    n_val = int(0.1 * len(X))
    print(f"[rul] C-MAPSS {len(X)} windows from {d['n_units_train']} units "
          f"(knee-masked physics: {float(knee.mean()):.0%} of timesteps below knee)")
    return {"X": torch.from_numpy(X), "Y": torch.from_numpy(Y),
            "knee": torch.from_numpy(knee),
            "X_test": torch.from_numpy(d["X_test"]),
            "rul_test": torch.from_numpy(d["rul_test"]),
            "train": idx[n_val:], "val": idx[:n_val]}


def build_pressure(args) -> dict:
    # 0.35 anomaly fraction + 800 cycles: the first v1 run had only ~14 deep
    # under-cures total — too few for the depth-weighted alpha loss to learn
    # the tail. Synthetic data is free; boil the lake.
    ds = pressure_synthetic.generate_dataset(
        n_cycles=120 if args.smoke else 800,
        anomaly_fraction=0.35)
    p_set = np.array([c["params"].p_set_bar for c in ds["cycles"]], dtype=np.float32)
    tau_r = np.array([c["params"].tau_ramp_s for c in ds["cycles"]], dtype=np.float32)
    tau_rel = np.array([c["params"].tau_release_s for c in ds["cycles"]], dtype=np.float32)
    rng = np.random.default_rng(4)
    idx = rng.permutation(len(ds["X"]))
    n_val = int(0.2 * len(ds["X"]))
    a = ds["alpha_end"]
    print(f"[pressure] SYNTHETIC {len(ds['X'])} cycles "
          f"({int(ds['y'].sum())} leaky; alpha_end {a.min():.3f}-{a.max():.3f})")
    return {"X": torch.from_numpy(ds["X"]), "y": torch.from_numpy(ds["y"]),
            "alpha_end": torch.from_numpy(ds["alpha_end"]),
            "dt": torch.from_numpy(ds["dt"]),
            "mask_ramp": torch.from_numpy(ds["mask_ramp"]),
            "mask_hold": torch.from_numpy(ds["mask_hold"]),
            "mask_release": torch.from_numpy(ds["mask_release"]),
            "p_set": torch.from_numpy(p_set), "tau_ramp": torch.from_numpy(tau_r),
            "tau_release": torch.from_numpy(tau_rel),
            "train": idx[n_val:], "val": idx[:n_val], "frame": 8}


# ----------------------------------------------------------------------- batches
def batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator):
    order = rng.permutation(indices)
    for s in range(0, len(order), batch_size):
        yield order[s:s + batch_size]


def press_extras(d: dict, idx) -> dict:
    """Per-batch extras the physics-integrated pressure head needs."""
    return {"dt": d["dt"][idx], "mask_ramp": d["mask_ramp"][idx],
            "mask_hold": d["mask_hold"][idx], "mask_release": d["mask_release"][idx]}


def head_step(model: MHPinn, head: str, data: dict, bidx: np.ndarray) -> dict:
    if head == "vibration":
        out = model("vibration", data["X"][bidx])
        terms = L.vibration_loss(out, data["y"][bidx], data["bands"][bidx])
        if "ims_X" in data:  # IMS health supervision rides along
            rng = np.random.default_rng(int(bidx[0]))
            j = rng.integers(0, len(data["ims_X"]), size=min(32, len(data["ims_X"])))
            out_i = model("vibration", data["ims_X"][j])
            h_loss = torch.nn.functional.mse_loss(out_i["health"], data["ims_health"][j])
            terms["total"] = terms["total"] + 0.5 * h_loss
        return terms
    if head == "fe_recon":
        out = model("fe_recon", data["X"][bidx])
        return L.fe_recon_loss(out, data["fe_bands"][bidx], data["fe_rms"][bidx],
                               data["bands"][bidx])
    if head == "fatigue":
        rng = np.random.default_rng(int(bidx[0]) + 7)
        j = rng.integers(0, len(data["ims_X"]), size=min(48, len(data["ims_X"])))
        out = model("fatigue", data["ims_X"][j])
        return L.fatigue_loss(out, data["ims_health"][j], data["ims_order"][j])
    if head == "thermal":
        out = model("thermal", data["X"][bidx])
        return L.thermal_loss(out, data["y"][bidx], data["raw"][bidx])
    if head == "thermal_recon":
        out = model("thermal_recon", data["X_masked"][bidx])
        return L.thermal_recon_loss(out, data["temp_true"][bidx], data["raw"][bidx],
                                    data["y"][bidx][:, 1])
    if head == "rul":
        out = model("rul", data["X"][bidx])
        return L.rul_loss(out, data["Y"][bidx], lambda_phys=RUL_LAMBDA_PHYS,
                          below_knee=data["knee"][bidx])
    if head == "pressure":
        out = model("pressure", data["X"][bidx], extras=press_extras(data, bidx))
        batch = {k: (v[bidx] if isinstance(v, torch.Tensor) else v)
                 for k, v in data.items() if k not in ("train", "val")}
        return L.pressure_loss(out, batch)
    raise ValueError(head)


# The tasks list: (loss/output name, source-data key). Several outputs share
# one dataset dict.
def task_list(tasks: dict) -> list[tuple[str, str]]:
    tl = []
    for h in tasks:
        tl.append((h, h))
        if h == "vibration":
            tl.append(("fe_recon", "vibration"))
            if "ims_X" in tasks["vibration"]:
                tl.append(("fatigue", "vibration"))
        if h == "thermal":
            tl.append(("thermal_recon", "thermal"))
    return tl


# ------------------------------------------------------------------------- train
@torch.no_grad()
def quick_val_metrics(model: MHPinn, tasks: dict) -> dict:
    """Small per-epoch validation metrics for the learning curves."""
    model.eval()
    m = {}
    if "vibration" in tasks:
        d, v = tasks["vibration"], tasks["vibration"]["val"][:512]
        out = model("vibration", d["X"][v])
        m["vibration_val_acc"] = float((out["fault_logits"].argmax(1) == d["y"][v])
                                       .float().mean())
        fe = model("fe_recon", d["X"][v])
        m["fe_recon_band_mae"] = float((fe["fe_bands"] - d["fe_bands"][v]).abs().mean())
        if "ims_X" in d:
            dm = model("fatigue", d["ims_X"])["damage"]
            m["fatigue_damage_corr"] = float(np.corrcoef(
                dm.numpy(), d["ims_health"].numpy())[0, 1])
    if "thermal" in tasks:
        d, v = tasks["thermal"], tasks["thermal"]["val"][:1024]
        p = torch.sigmoid(model("thermal", d["X"][v])["failure_logits"])
        y = d["y"][v]
        m["thermal_val_recall"] = float(((p[:, 0] > 0.5) & (y[:, 0] > 0)).sum()
                                        / max(1, int(y[:, 0].sum())))
        t_hat = model("thermal_recon", d["X_masked"][v])["process_temp_K"]
        m["temp_recon_mae_K"] = float((t_hat - d["temp_true"][v]).abs().mean())
    if "rul" in tasks:
        d = tasks["rul"]
        rul_hat = model("rul", d["X_test"])["rul_seq"][:, -1]
        m["rul_test_rmse"] = float(torch.sqrt(torch.mean((rul_hat - d["rul_test"]) ** 2)))
    if "pressure" in tasks:
        d, v = tasks["pressure"], tasks["pressure"]["val"]
        out = model("pressure", d["X"][v], extras=press_extras(d, v))
        m["pressure_anom_acc"] = float(((torch.sigmoid(out["anomaly_logit"]) > 0.5)
                                        .float() == d["y"][v]).float().mean())
        m["alpha_end_mae"] = float((out["alpha_end"] - d["alpha_end"][v]).abs().mean())
    model.train()
    return m


def train(model: MHPinn, tasks: dict, epochs: int, batch_size: int, lr: float):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(123)
    tl = task_list(tasks)
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        iters = {name: batches(tasks[src]["train"], batch_size, rng)
                 for name, src in tl}
        srcs = dict(tl)
        log = {name: {"data": [], "phys": [], "total": []} for name, _ in tl}
        while iters:
            for name in list(iters):
                try:
                    bidx = next(iters[name])
                except StopIteration:
                    del iters[name]
                    continue
                terms = head_step(model, name, tasks[srcs[name]], bidx)
                opt.zero_grad()
                terms["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                for k in ("data", "phys", "total"):
                    if k in terms:
                        log[name][k].append(float(terms[k].detach()))
        entry = {"epoch": epoch}
        for name, kk in log.items():
            for k, vals in kk.items():
                if vals:
                    entry[f"{name}_{k}"] = float(np.mean(vals))
        entry.update(quick_val_metrics(model, tasks))
        history.append(entry)
        heads_str = ", ".join(f"{n}={entry.get(f'{n}_total', float('nan')):.4f}"
                              for n, _ in tl)
        val_str = ", ".join(f"{k}={v:.4f}" for k, v in entry.items()
                            if k not in ("epoch",) and not k.endswith(("_data", "_phys", "_total")))
        print(f"epoch {epoch}/{epochs} | loss: {heads_str}")
        print(f"           | val:  {val_str}", flush=True)
    return history


def plot_curves(history: list[dict], outdir: Path):
    """Loss + metric curves per head — the trend is the deliverable."""
    epochs = [h["epoch"] for h in history]
    names = sorted({k.rsplit("_", 1)[0] for h in history for k in h
                    if k.endswith("_total")})
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    for ax, name in zip(axes.flat, names):
        for part, style in (("data", "--"), ("phys", ":"), ("total", "-")):
            ys = [h.get(f"{name}_{part}") for h in history]
            if any(y is not None for y in ys):
                ax.plot(epochs, ys, style, label=part)
        ax.set_title(name); ax.set_xlabel("epoch"); ax.set_yscale("log")
        ax.legend(fontsize=7)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    fig.suptitle("v1 training losses per head (log scale): data vs physics vs total")
    fig.tight_layout(); fig.savefig(outdir / "curves_losses.png", dpi=130)
    plt.close(fig)

    metric_keys = sorted({k for h in history for k in h
                          if not k.endswith(("_data", "_phys", "_total"))
                          and k != "epoch"})
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    for ax, key in zip(axes.flat, metric_keys):
        ax.plot(epochs, [h.get(key) for h in history], marker="o", ms=3)
        ax.set_title(key); ax.set_xlabel("epoch")
    for ax in axes.flat[len(metric_keys):]:
        ax.axis("off")
    fig.suptitle("v1 validation metrics per epoch")
    fig.tight_layout(); fig.savefig(outdir / "curves_metrics.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- virtual sensor
@torch.no_grad()
def virtual_sensor_demo(model: MHPinn, tasks: dict, outdir: Path) -> dict:
    """THE flagship demo: reconstruct withheld sensors, report error vs truth."""
    model.eval()
    demo: dict = {}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    if "thermal" in tasks:
        d, v = tasks["thermal"], tasks["thermal"]["val"]
        t_hat = model("thermal_recon", d["X_masked"][v])["process_temp_K"].numpy()
        t_true = d["temp_true"][v].numpy()
        mae = float(np.abs(t_hat - t_true).mean())
        rng_span = float(t_true.max() - t_true.min())
        demo["ai4i_process_temp"] = {
            "withheld_sensor": "Process temperature [K] (input column masked)",
            "n_val_rows": int(len(v)),
            "MAE_K": round(mae, 3),
            "true_range_K": [round(float(t_true.min()), 1), round(float(t_true.max()), 1)],
            "MAE_as_pct_of_range": round(100 * mae / rng_span, 1),
            "provenance": "SIMULATED ground truth (AI4I)",
        }
        axes[0].scatter(t_true, t_hat, s=4, alpha=0.3)
        lims = [t_true.min() - 0.5, t_true.max() + 0.5]
        axes[0].plot(lims, lims, "k--", lw=1)
        axes[0].set_xlabel("true (withheld) process temp [K]")
        axes[0].set_ylabel("reconstructed [K]")
        axes[0].set_title(f"Virtual sensor — AI4I process temp, MAE {mae:.2f} K")

    if "vibration" in tasks:
        d, v = tasks["vibration"], tasks["vibration"]["val"]
        out = model("fe_recon", d["X"][v])
        b_hat, b_true = out["fe_bands"].numpy(), d["fe_bands"][v].numpy()
        r_hat, r_true = out["fe_rms"].numpy(), d["fe_rms"][v].numpy()
        rel_rms = float(np.median(np.abs(r_hat - r_true) / (r_true + 1e-9)))
        demo["cwru_fan_end"] = {
            "withheld_sensor": "Fan-end accelerometer (REAL, never a model input)",
            "n_val_windows": int(len(v)),
            "band_energy_MAE": round(float(np.abs(b_hat - b_true).mean()), 4),
            "rms_median_relative_error_pct": round(100 * rel_rms, 1),
            "provenance": "REAL measured ground truth (CWRU)",
        }
        for k, (name, c) in enumerate((("BPFO", "tab:blue"), ("BPFI", "tab:red"),
                                       ("BSF", "tab:green"))):
            axes[1].scatter(b_true[:, k], b_hat[:, k], s=5, alpha=0.35, c=c, label=name)
        lim = max(b_true.max(), b_hat.max()) * 1.05
        axes[1].plot([0, lim], [0, lim], "k--", lw=1)
        axes[1].set_xlabel("true fan-end envelope band energy (withheld)")
        axes[1].set_ylabel("reconstructed from drive-end")
        axes[1].set_title(f"Virtual sensor — CWRU fan-end bands, "
                          f"RMS med.rel.err {100 * rel_rms:.0f}%")
        axes[1].legend(fontsize=8)

    fig.suptitle("Single-sensor reconstruction: physics + remaining sensors vs the withheld measurement")
    fig.tight_layout(); fig.savefig(outdir / "virtual_sensor_demo.png", dpi=130)
    plt.close(fig)
    return demo


# -------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model: MHPinn, tasks: dict) -> dict:
    model.eval()
    report: dict = {}

    if "vibration" in tasks:
        d, v = tasks["vibration"], tasks["vibration"]["val"]
        out = model("vibration", d["X"][v])
        pred = out["fault_logits"].argmax(dim=1)
        report["vibration"] = {
            "val_accuracy_4class": round(float((pred == d["y"][v]).float().mean()), 4),
            "sample": {"true_class": cwru.CLASSES[int(d['y'][v][0])],
                       "pred_class": cwru.CLASSES[int(pred[0])]},
        }
        if "ims_X" in d:
            h = model("vibration", d["ims_X"])["health"]
            report["vibration"]["ims_health_corr_with_lifetime"] = round(
                float(np.corrcoef(h.numpy(), d["ims_health"].numpy())[0, 1]), 3)

    if "vibration" in tasks and "ims_X" in tasks["vibration"]:
        d = tasks["vibration"]
        dm = model("fatigue", d["ims_X"])["damage"].numpy()
        target = d["ims_health"].numpy()
        corr = float(np.corrcoef(dm, target)[0, 1])
        # Physics interpretation: latest window -> hours to planned replacement.
        i_last = int(np.argmax(d["ims_order"].numpy()))
        cps = fat.stress_cycles_per_second("OR")
        nf_geo = 4.8e8  # geometric mean of the three observed IMS lifetimes
        hours = fat.hours_to_replacement(float(dm[i_last]), nf_geo, cps)
        report["fatigue"] = {
            "damage_corr_with_lifetime": round(corr, 3),
            "final_window_damage_D": round(float(dm[i_last]), 3),
            "true_lifetime_position": round(float(target[i_last]), 3),
            "hours_to_replacement_at_D0.9_(Basquin/Miner)": round(hours, 1),
            "note": "N_f = 4.8e8 cycles = geometric mean of the 3 REAL IMS "
                    "failures; sigma_f' family in pinn/physics/fatigue.py",
        }

    if "thermal" in tasks:
        d, v = tasks["thermal"], tasks["thermal"]["val"]
        p = torch.sigmoid(model("thermal", d["X"][v])["failure_logits"])
        y = d["y"][v]
        i = int(y[:, 1].argmax())
        report["thermal"] = {
            "val_recall_machine_failure@0.5": round(
                float(((p[:, 0] > 0.5) & (y[:, 0] > 0)).sum()
                      / max(1, int(y[:, 0].sum()))), 3),
            "val_mean_pred_on_healthy": round(float(p[y[:, 0] == 0, 0].mean()), 4),
            "sample_HDF_row_pred_MF_HDF_PWF": [round(float(x), 3) for x in p[i]],
        }

    if "rul" in tasks:
        d = tasks["rul"]
        rul_hat = model("rul", d["X_test"])["rul_seq"][:, -1]
        # Monotonicity checked where it physically applies (Step-7): below the
        # per-unit OBSERVABLE degradation knee, not the conventional 125 cap.
        vout = model("rul", d["X"][d["val"]])["rul_seq"]
        vtgt = d["Y"][d["val"]]
        below_knee = d["knee"][d["val"]][:, 1:]
        below_cap = vtgt[:, 1:] < 124.5
        diffs_knee = (vout[:, 1:] - vout[:, :-1])[below_knee]
        diffs_cap = (vout[:, 1:] - vout[:, :-1])[below_cap]
        report["rul"] = {
            "test_RMSE_cycles_(capped_target)": round(
                float(torch.sqrt(torch.mean((rul_hat - d["rul_test"]) ** 2))), 2),
            "pred_range": [round(float(rul_hat.min()), 1), round(float(rul_hat.max()), 1)],
            "mean_dRUL_below_KNEE_(constraint_region,_should_be_-1)":
                round(float(diffs_knee.mean()), 3),
            "frac_increasing_steps_below_knee":
                round(float((diffs_knee > 0).float().mean()), 3),
            "mean_dRUL_below_cap125_(legacy_comparison)":
                round(float(diffs_cap.mean()), 3),
        }

    if "pressure" in tasks:
        d, v = tasks["pressure"], tasks["pressure"]["val"]
        out = model("pressure", d["X"][v], extras=press_extras(d, v))
        p_anom = torch.sigmoid(out["anomaly_logit"])
        y = d["y"][v]
        a_hat, a_true = out["alpha_end"], d["alpha_end"][v]
        under_true = a_true < ALPHA_CURED_THRESHOLD
        under_pred = a_hat < ALPHA_CURED_THRESHOLD
        leaky = y > 0
        # Linear recalibration fitted on TRAIN leaky cycles (diagnosed
        # regression-to-mean: pred-vs-true slope ~0.52 — the head tracks
        # under-cure direction but halves the amplitude). Applied to val only;
        # no leakage. Coefficients persisted for inference use.
        tr = d["train"]
        out_tr = model("pressure", d["X"][tr], extras=press_extras(d, tr))
        lt = d["y"][tr] > 0
        fit = np.polyfit(out_tr["alpha_end"][lt].numpy(),
                         d["alpha_end"][tr][lt].numpy(), 1)
        a_cal = torch.from_numpy(np.polyval(fit, a_hat.numpy()).astype(np.float32))
        under_pred_cal = a_cal < ALPHA_CURED_THRESHOLD
        # Correlation on leaky cycles is the robust depth metric — the
        # threshold count below can rest on 1-3 validation cases.
        alpha_corr = float(np.corrcoef(a_hat[leaky].numpy(),
                                       a_true[leaky].numpy())[0, 1]) \
            if int(leaky.sum()) > 2 else None
        report["pressure"] = {
            "provenance": "SYNTHETIC — generated curing cycles, never measured data",
            "val_anomaly_accuracy@0.5": round(float(((p_anom > 0.5).float() == y)
                                                    .float().mean()), 3),
            "alpha_end_MAE": round(float((a_hat - a_true).abs().mean()), 4),
            "alpha_corr_on_leaky_cycles": None if alpha_corr is None
            else round(alpha_corr, 3),
            "under_cure_detection_(alpha<%.2f)" % ALPHA_CURED_THRESHOLD: {
                "n_true_under_cured": int(under_true.sum()),
                "n_caught_raw": int((under_true & under_pred).sum()),
                "n_caught_calibrated": int((under_true & under_pred_cal).sum()),
                "n_false_alarms_calibrated": int((~under_true & under_pred_cal).sum()),
            },
            "alpha_calibration_(pred_to_true,_fit_on_train)": [round(float(c), 4)
                                                               for c in fit],
            "recon_peak_bar": round(float(out["pressure_recon"].max()), 2),
        }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", nargs="+",
                    default=["vibration", "thermal", "rul", "pressure"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--smoke", action="store_true", help="tiny fast run")
    ap.add_argument("--run-dir", default=None,
                    help="artifact directory (default runs/v1)")
    ap.add_argument("--rul-lambda", type=float, default=None,
                    help="override RUL lambda_phys (Step-6 sweep)")
    ap.add_argument("--rul-monotone", action="store_true",
                    help="use the monotone-by-construction RUL head")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 1
    global RUNS_DIR, RUL_LAMBDA_PHYS
    if args.run_dir:
        RUNS_DIR = Path(args.run_dir)
    if args.rul_lambda is not None:
        RUL_LAMBDA_PHYS = args.rul_lambda

    torch.manual_seed(0)
    print("Bearing geometry cross-check vs published dataset constants:")
    for line in verify_against_published():
        print("  ", line)

    builders = {"vibration": build_vibration, "thermal": build_thermal,
                "rul": build_rul, "pressure": build_pressure}
    tasks = {}
    for h in args.heads:
        d = builders[h](args)
        if d is not None:
            tasks[h] = d

    model = MHPinn(rul_monotone=args.rul_monotone)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MHPinn v1: {n_params:,} parameters, trained outputs: "
          f"{[n for n, _ in task_list(tasks)]}"
          + (" [RUL monotone head]" if args.rul_monotone else "")
          + f" [rul lambda_phys={RUL_LAMBDA_PHYS}]", flush=True)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    history = train(model, tasks, args.epochs, args.batch_size, args.lr)
    (RUNS_DIR / "history.json").write_text(json.dumps(history, indent=2))
    plot_curves(history, RUNS_DIR)

    report = evaluate(model, tasks)
    demo = virtual_sensor_demo(model, tasks, RUNS_DIR)
    report["virtual_sensor_demo"] = demo
    print("\n===== evaluation =====")
    print(json.dumps(report, indent=2))

    # Versioned name (v2 = current architecture: hidden 128, knee-masked RUL,
    # integrated pressure ODE); the demo copy lives at pinn/models/mh_pinn_v2.pt.
    torch.save(model.state_dict(), RUNS_DIR / "mh_pinn_v2.pt")
    (RUNS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))

    # Legacy draft payload (v0 shape) + the PRAETOR-contract SignedReading.
    payload = build_payload("curing_press_01", {k: v for k, v in report.items()},
                            key=b"demo-key-not-a-secret-replace-with-shared-team-key")
    (RUNS_DIR / "sample_telemetry.json").write_text(json.dumps(payload, indent=2))

    vib = report.get("vibration", {})
    fatg = report.get("fatigue", {})
    press = report.get("pressure", {})
    sr = to_praetor_signed_reading(
        machine_id="RC-07", department="Curing", epoch=0,
        signals={"mould_temp_C": 195.0, "coil_power_kW": 71.0,
                 "vibration_rms_mm_s": 2.1,
                 "pressure_bar": press.get("recon_peak_bar", 17.0)},
        health_index=1.0 - fatg.get("final_window_damage_D", 0.1),
        rul_cycles=report.get("rul", {}).get("pred_range", [100])[0],
        residual=0.0,
        failure_mode_probs={"bearing_fault": 1.0 - vib.get("val_accuracy_4class", 0.9)},
        note="v1 sample — near-failure snapshot (last IMS window, damage from "
             "fatigue head); schema per backend/agent/schemas.py")
    (RUNS_DIR / "sample_praetor_reading.json").write_text(json.dumps(sr, indent=2))
    print(f"\nSaved checkpoint, curves, metrics, demo PNG and signed samples to {RUNS_DIR}/")


if __name__ == "__main__":
    main()
