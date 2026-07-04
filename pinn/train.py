"""v0 training entrypoint: each head loads its data, trains a few epochs on the
shared body, and prints an inspectable sample output + physical sanity checks.

Usage:
    python -m pinn.train --smoke          # tiny subset, 1 epoch, fast CPU check
    python -m pinn.train --epochs 3       # v0 run
    python -m pinn.train --heads thermal rul pressure   # skip vibration/IMS

Deliberately NOT here yet (later phases): champion/challenger continuous
learning, operator Q&A, IFC visual layer, hyperparameter tuning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pinn import losses as L
from pinn.data import ai4i, cmapss, cwru, ims, pressure_synthetic
from pinn.model import MHPinn
from pinn.physics.bearing import verify_against_published
from pinn.telemetry import build_payload

RUNS_DIR = Path("runs/v0")


# --------------------------------------------------------------------------- data
def build_vibration(args) -> dict | None:
    root = Path(args.data_root) / "cwru"
    if not root.exists():
        print("[vibration] CWRU data missing — skipping head")
        return None
    X, y, bands, rpm, _meta = cwru.load_windows(
        root, window=2048, hop=2048 if not args.smoke else 8192)
    if args.smoke:
        keep = np.random.default_rng(0).permutation(len(X))[:400]
        X, y, bands = X[keep], y[keep], bands[keep]
    # per-window standardization (amplitude varies across files/loads)
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(X))
    n_val = int(0.2 * len(X))
    d = {"X": torch.from_numpy(X), "y": torch.from_numpy(y),
         "bands": torch.from_numpy(bands),
         "train": idx[n_val:], "val": idx[:n_val]}
    # Optional IMS health supervision (if extracted)
    ims_dir = Path(args.data_root) / "ims" / "2nd_test"
    if ims_dir.exists():
        Xi, hi, _ = ims.load_health_windows(Path(args.data_root) / "ims",
                                            stride_files=20 if args.smoke else 5)
        Xi = (Xi - Xi.mean(axis=1, keepdims=True)) / (Xi.std(axis=1, keepdims=True) + 1e-8)
        d["ims_X"], d["ims_health"] = torch.from_numpy(Xi), torch.from_numpy(hi)
        print(f"[vibration] CWRU {len(X)} windows + IMS {len(Xi)} health windows")
    else:
        print(f"[vibration] CWRU {len(X)} windows (IMS not extracted yet — health "
              f"supervision skipped)")
    return d


def build_thermal(args) -> dict:
    d = ai4i.load(Path(args.data_root) / "ai4i" / "ai4i2020.csv")
    print(f"[thermal] AI4I {len(d['X'])} rows "
          f"(HDF rule/label overlap: {d['rule_hdf_overlap']})")
    return {"X": torch.from_numpy(d["X"]), "raw": torch.from_numpy(d["raw"]),
            "y": torch.from_numpy(d["y"]),
            "train": d["train_idx"], "val": d["val_idx"]}


def build_rul(args) -> dict:
    d = cmapss.load(Path(args.data_root) / "cmapss")
    X, Y = d["X_train"], d["Y_train"]
    if args.smoke:
        X, Y = X[:2000], Y[:2000]
    rng = np.random.default_rng(2)
    idx = rng.permutation(len(X))
    n_val = int(0.1 * len(X))
    print(f"[rul] C-MAPSS {len(X)} windows from {d['n_units_train']} units")
    return {"X": torch.from_numpy(X), "Y": torch.from_numpy(Y),
            "X_test": torch.from_numpy(d["X_test"]),
            "rul_test": torch.from_numpy(d["rul_test"]),
            "train": idx[n_val:], "val": idx[:n_val]}


def build_pressure(args) -> dict:
    ds = pressure_synthetic.generate_dataset(n_cycles=120 if args.smoke else 400)
    p_set = np.array([c["params"].p_set_bar for c in ds["cycles"]], dtype=np.float32)
    tau_r = np.array([c["params"].tau_ramp_s for c in ds["cycles"]], dtype=np.float32)
    tau_rel = np.array([c["params"].tau_release_s for c in ds["cycles"]], dtype=np.float32)
    rng = np.random.default_rng(4)
    idx = rng.permutation(len(ds["X"]))
    n_val = int(0.2 * len(ds["X"]))
    print(f"[pressure] SYNTHETIC {len(ds['X'])} cycles "
          f"({int(ds['y'].sum())} anomalous)")
    return {"X": torch.from_numpy(ds["X"]), "y": torch.from_numpy(ds["y"]),
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


def head_step(model: MHPinn, head: str, data: dict, bidx: np.ndarray) -> dict:
    if head == "vibration":
        out = model("vibration", data["X"][bidx])
        terms = L.vibration_loss(out, data["y"][bidx], data["bands"][bidx])
        # IMS health supervision rides along on a random IMS batch when present
        if "ims_X" in data:
            rng = np.random.default_rng(int(bidx[0]))
            j = rng.integers(0, len(data["ims_X"]), size=min(32, len(data["ims_X"])))
            out_i = model("vibration", data["ims_X"][j])
            h_loss = torch.nn.functional.mse_loss(out_i["health"], data["ims_health"][j])
            terms["total"] = terms["total"] + 0.5 * h_loss
            terms["ims_health"] = h_loss
        return terms
    if head == "thermal":
        out = model("thermal", data["X"][bidx])
        return L.thermal_loss(out, data["y"][bidx], data["raw"][bidx])
    if head == "rul":
        out = model("rul", data["X"][bidx])
        return L.rul_loss(out, data["Y"][bidx])
    if head == "pressure":
        out = model("pressure", data["X"][bidx])
        batch = {k: (v[bidx] if isinstance(v, torch.Tensor) else v)
                 for k, v in data.items() if k not in ("train", "val")}
        return L.pressure_loss(out, batch)
    raise ValueError(head)


# ------------------------------------------------------------------------- train
def train(model: MHPinn, tasks: dict, epochs: int, batch_size: int, lr: float):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(123)
    for epoch in range(1, epochs + 1):
        iters = {h: batches(d["train"], batch_size, rng) for h, d in tasks.items()}
        step, log = 0, {h: [] for h in tasks}
        while iters:
            for head in list(iters):
                try:
                    bidx = next(iters[head])
                except StopIteration:
                    del iters[head]
                    continue
                terms = head_step(model, head, tasks[head], bidx)
                opt.zero_grad()
                terms["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                log[head].append(float(terms["total"]))
                step += 1
        means = {h: (float(np.mean(v)) if v else None) for h, v in log.items()}
        print(f"epoch {epoch}/{epochs} mean loss per head: "
              + ", ".join(f"{h}={m:.4f}" for h, m in means.items() if m is not None))
    return means


# -------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model: MHPinn, tasks: dict) -> dict:
    model.eval()
    report: dict = {}

    if "vibration" in tasks:
        d, v = tasks["vibration"], tasks["vibration"]["val"]
        out = model("vibration", d["X"][v])
        pred = out["fault_logits"].argmax(dim=1)
        acc = float((pred == d["y"][v]).float().mean())
        report["vibration"] = {
            "val_accuracy_4class": round(acc, 4),
            "sample": {"true_class": cwru.CLASSES[int(d['y'][v][0])],
                       "pred_class": cwru.CLASSES[int(pred[0])],
                       "band_energies_BPFO_BPFI_BSF": [round(float(x), 5)
                                                       for x in d["bands"][v][0]]},
        }
        if "ims_X" in d:
            h = model("vibration", d["ims_X"])["health"]
            # health should broadly increase with lifetime position
            corr = float(np.corrcoef(h.numpy(), d["ims_health"].numpy())[0, 1])
            report["vibration"]["ims_health_corr_with_lifetime"] = round(corr, 3)

    if "thermal" in tasks:
        d, v = tasks["thermal"], tasks["thermal"]["val"]
        p = torch.sigmoid(model("thermal", d["X"][v])["failure_logits"])
        y = d["y"][v]
        recall_mf = float(((p[:, 0] > 0.5) & (y[:, 0] > 0)).sum() / max(1, int(y[:, 0].sum())))
        i = int(y[:, 1].argmax())  # one real HDF row
        report["thermal"] = {
            "val_recall_machine_failure@0.5": round(recall_mf, 3),
            "val_mean_pred_on_healthy": round(float(p[y[:, 0] == 0, 0].mean()), 4),
            "sample_HDF_row": {"true": y[i].tolist(),
                               "pred_MF_HDF_PWF": [round(float(x), 3) for x in p[i]]},
        }

    if "rul" in tasks:
        d = tasks["rul"]
        rul_hat = model("rul", d["X_test"])["rul_seq"][:, -1]
        rmse = float(torch.sqrt(torch.mean((rul_hat - d["rul_test"]) ** 2)))
        # Monotonicity check where it physically applies: windows below the cap.
        vout = model("rul", d["X"][d["val"]])["rul_seq"]
        vtgt = d["Y"][d["val"]]
        below = vtgt[:, 1:] < 124.5
        diffs = (vout[:, 1:] - vout[:, :-1])[below]
        report["rul"] = {
            "test_RMSE_cycles_(capped_target)": round(rmse, 2),
            "pred_range": [round(float(rul_hat.min()), 1), round(float(rul_hat.max()), 1)],
            "physically_plausible_range_0_125": bool((rul_hat >= 0).all()
                                                     and (rul_hat <= 130).all()),
            "mean_dRUL_per_cycle_below_cap_(should_be_-1)": round(float(diffs.mean()), 3),
            "frac_increasing_steps_below_cap": round(float((diffs > 0).float().mean()), 3),
        }

    if "pressure" in tasks:
        d, v = tasks["pressure"], tasks["pressure"]["val"]
        out = model("pressure", d["X"][v])
        p_anom = torch.sigmoid(out["anomaly_logit"])
        y = d["y"][v]
        acc = float(((p_anom > 0.5).float() == y).float().mean())
        nominal = y == 0
        recon = out["pressure_recon"]
        frame = d["frame"]
        k = recon.shape[1]
        p_obs = d["X"][v][:, : k * frame, 0].reshape(len(v), k, frame).mean(dim=2)
        recon_rmse = float(torch.sqrt(torch.mean((recon[nominal] - p_obs[nominal]) ** 2)))
        report["pressure"] = {
            "provenance": "SYNTHETIC — generated curing cycles, never measured data",
            "val_anomaly_accuracy@0.5": round(acc, 3),
            "recon_RMSE_bar_on_nominal": round(recon_rmse, 3),
            "recon_peak_bar": round(float(recon.max()), 2),
            "peak_inside_documented_16_19_bar": bool(14.0 <= float(recon.max()) <= 21.0),
        }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", nargs="+",
                    default=["vibration", "thermal", "rul", "pressure"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--smoke", action="store_true", help="tiny fast run")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 1

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

    model = MHPinn()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MHPinn: {n_params:,} parameters, heads: {list(tasks)}")

    train(model, tasks, args.epochs, args.batch_size, args.lr)
    report = evaluate(model, tasks)
    print("\n===== v0 evaluation =====")
    print(json.dumps(report, indent=2))

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), RUNS_DIR / "mh_pinn_v0.pt")
    (RUNS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))

    # Demo telemetry payload for the information layer (DRAFT schema, demo key).
    payload = build_payload(
        machine_id="curing_press_01",
        head_outputs={k: v for k, v in report.items()},
        key=b"demo-key-not-a-secret-replace-with-shared-team-key",
    )
    (RUNS_DIR / "sample_telemetry.json").write_text(json.dumps(payload, indent=2))
    print(f"\nSaved checkpoint, metrics and sample signed telemetry to {RUNS_DIR}/")


if __name__ == "__main__":
    main()
