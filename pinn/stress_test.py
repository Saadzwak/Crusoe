"""Step-7 stress test: question the equations themselves against the data.

A citation proves an equation is real, not that it is the right model for THIS
system. Four instrumented checks:

A1. C-MAPSS knee analysis — the piecewise-linear RUL target with a FIXED cap
    at 125 (Heimes 2008) is a convention. Fit a two-segment (flat, then
    linear) model to each of the 100 training engines' composite health index
    and measure where degradation actually becomes observable per unit. Wide
    dispersion => the fixed cap mislabels early/late-knee engines.

A2. Multi-task gradient conflict — cosine similarity between per-task
    gradients on the SHARED BODY across batches. Identifies which task(s)
    actually pull against RUL (the diagnosed interference), so the remedy is
    chosen from measurement, not fashion.

A3. Vibration pseudo-label variants — is fundamental+2nd-harmonic envelope
    band energy the right physical proxy (81.5% agreement)? Test: harmonics
    1-3; local SNR against flanking guard bands; BPFI +/- f_r sidebands for
    inner-race faults (the IR signature per Randall & Antoni 2011). All
    scored against real CWRU ground truth.

A4. Sensitivity of every category-(c) constant — Kamal (m, n, k2/k1) with A
    re-calibrated per combination (the calibration anchors t99(195C)=12min by
    construction), roller length and load share in the Hertz/Basquin chain,
    Ea/R across the patent's 10-14kK range. Answers: which (c) choices
    actually matter, which are absorbed by calibration.

Usage:  python -m pinn.stress_test      Writes: runs/v2/stress_test.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import hilbert

from pinn.data import cwru
from pinn.model import MHPinn
from pinn.physics.bearing import CWRU_DRIVE_END_6205, fault_frequencies
from pinn.physics.cure_kinetics import CureParams, _t99_at, calibrated_prefactor, kamal_rate
from pinn.physics.fatigue import calibrate_sigma_f_prime, ims_hertz_stress

OUT = Path("runs/v2")


# ------------------------------------------------------------------------- A1
def cmapss_knee_analysis() -> dict:
    cols = (["unit", "cycle", "op1", "op2", "op3"]
            + [f"s{i}" for i in range(1, 22)])
    drop = ["op3", "s1", "s5", "s6", "s10", "s16", "s18", "s19"]
    feats = [c for c in cols if c.startswith("s") and c not in drop]
    df = pd.read_csv("data/cmapss/train_FD001.txt", sep=r"\s+", header=None,
                     names=cols)
    mu, sd = df[feats].mean(), df[feats].std() + 1e-8

    rul_at_knee = []
    for _unit, g in df.groupby("unit"):
        z = ((g[feats] - mu) / sd).to_numpy()
        # orient every sensor so end-of-life drift is positive, then average
        sign = np.sign(z[-15:].mean(axis=0) - z[:15].mean(axis=0))
        h = (z * sign).mean(axis=1)
        k = 5
        h = np.convolve(h, np.ones(k) / k, mode="valid")     # light smoothing
        n = len(h)
        best_sse, best_t = np.inf, None
        for t in range(8, n - 8):
            sse = float(((h[:t] - h[:t].mean()) ** 2).sum())
            x = np.arange(t, n)
            a, b = np.polyfit(x, h[t:], 1)
            sse += float(((h[t:] - (a * x + b)) ** 2).sum())
            if sse < best_sse:
                best_sse, best_t = sse, t
        rul_at_knee.append(len(g) - best_t)                   # cycles left at knee
    r = np.array(rul_at_knee, dtype=float)
    return {
        "median_RUL_at_knee": float(np.median(r)),
        "IQR": [float(np.percentile(r, 25)), float(np.percentile(r, 75))],
        "min_max": [float(r.min()), float(r.max())],
        "convention_cap": 125.0,
        "frac_units_knee_above_125": float((r > 125).mean()),
        "frac_units_knee_below_75": float((r < 75).mean()),
        "reading": "wide dispersion => the fixed 125 cap mislabels a large "
                   "share of windows (flat-sensor windows labeled as "
                   "degrading, or vice versa); per-unit knee targets are the "
                   "data-supported alternative",
    }


# ------------------------------------------------------------------------- A2
def gradient_conflict(n_batches: int = 6, batch: int = 64) -> dict:
    from pinn.train import build_pressure, build_rul, build_thermal, build_vibration
    from pinn.train import head_step, task_list

    args = argparse.Namespace(smoke=False, data_root="data")
    tasks = {"vibration": build_vibration(args), "thermal": build_thermal(args),
             "rul": build_rul(args), "pressure": build_pressure(args)}
    model = MHPinn()
    model.load_state_dict(torch.load("runs/v1/mh_pinn_v1.pt", weights_only=True))
    model.train()

    names = [n for n, _ in task_list(tasks)]
    srcs = dict(task_list(tasks))
    rng = np.random.default_rng(0)
    G: dict[str, list[np.ndarray]] = {n: [] for n in names}
    for _b in range(n_batches):
        for n in names:
            d = tasks[srcs[n]]
            bidx = rng.choice(d["train"], size=batch, replace=False)
            terms = head_step(model, n, d, bidx)
            model.zero_grad()
            terms["total"].backward()
            g = torch.cat([p.grad.flatten() for p in model.body.parameters()
                           if p.grad is not None])
            G[n].append(g.numpy().copy())
    cos = {}
    for i, a in enumerate(names):
        for b_ in names[i + 1:]:
            vals = [float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
                    for x, y in zip(G[a], G[b_])]
            cos[f"{a}|{b_}"] = round(float(np.mean(vals)), 4)
    rul_row = {k: v for k, v in cos.items() if "rul" in k}
    return {"mean_pairwise_cosine_shared_body": cos,
            "rul_conflicts_sorted": dict(sorted(rul_row.items(), key=lambda kv: kv[1]))}


# ------------------------------------------------------------------------- A3
def _bands(freqs, spec, f0, tol_frac=0.02, tol_min=3.0):
    tol = max(tol_min, tol_frac * f0)
    m = (freqs >= f0 - tol) & (freqs <= f0 + tol)
    return float(spec[m].sum()), float(m.sum())


def pseudolabel_variants() -> dict:
    X, y, _bands_cached, rpms, _meta = cwru.load_windows("data/cwru", fe_targets=False)
    fs = 12_000.0
    faulty = y > 0
    Xf, yf, rf = X[faulty], y[faulty], rpms[faulty]

    agree = {"current_h12": 0, "harm123": 0, "local_snr": 0, "ir_sidebands": 0}
    n = len(Xf)
    for w, cls, rpm in zip(Xf, yf, rf):
        env = np.abs(hilbert(w - w.mean()))
        env -= env.mean()
        spec = np.abs(np.fft.rfft(env)) ** 2
        freqs = np.fft.rfftfreq(len(w), 1.0 / fs)
        f = fault_frequencies(CWRU_DRIVE_END_6205, rpm / 60.0)
        fr = rpm / 60.0
        keys = ("BPFI", "BPFO", "BSF")           # order -> classes IR, OR, B

        def score(harmonics, snr=False, sidebands=False):
            s = []
            for kk in keys:
                e = 0.0
                for h in range(1, harmonics + 1):
                    eh, nb = _bands(freqs, spec, f[kk] * h)
                    if snr:
                        g1, n1 = _bands(freqs, spec, f[kk] * h - 6 * max(3, .02 * f[kk] * h))
                        g2, n2 = _bands(freqs, spec, f[kk] * h + 6 * max(3, .02 * f[kk] * h))
                        guard = (g1 + g2) / max(1, n1 + n2) * max(1, nb)
                        e += eh / (guard + 1e-12)
                    else:
                        e += eh
                if sidebands and kk == "BPFI":   # IR signature: f_r sidebands
                    for h in (1, 2):
                        for sb in (-fr, fr):
                            e += _bands(freqs, spec, f[kk] * h + sb)[0]
                s.append(e)
            return int(np.argmax(s)) + 1          # -> class index 1..3

        agree["current_h12"] += int(score(2) == cls)
        agree["harm123"] += int(score(3) == cls)
        agree["local_snr"] += int(score(2, snr=True) == cls)
        agree["ir_sidebands"] += int(score(2, sidebands=True) == cls)
    return {k: round(v / n, 4) for k, v in agree.items()} | {"n_faulty_windows": n}


# ------------------------------------------------------------------------- A4
def kamal_sensitivity() -> dict:
    rows = []
    # fixed reproducible leak scenario: 195C hold, saturation-temp deficit
    # ramping linearly to 15 K over the second half of a 720 s hold
    t = np.arange(720.0)
    deficit = np.where(t > 360, (t - 360) / 360 * 15.0, 0.0)
    temp = 195.0 - deficit
    for m in (0.5, 1.0, 2.0):
        for n_ in (1.0, 1.5, 2.0):
            for r in (1.0, 2.0, 4.0):
                p = CureParams(m=m, n=n_, k2_over_k1=r)
                A = calibrated_prefactor(p)
                a = 1e-6
                for T in temp:
                    a = min(1.0, a + float(kamal_rate(a, T + 273.15, A, p)))
                rows.append({"m": m, "n": n_, "k2/k1": r,
                             "t99_180C_min": round(_t99_at(A, 180.0, p) / 60, 1),
                             "alpha_end_leak_scenario": round(a, 4)})
    alpha_spread = [r["alpha_end_leak_scenario"] for r in rows]
    t99_spread = [r["t99_180C_min"] for r in rows]
    return {"grid": rows,
            "alpha_end_range_across_grid": [min(alpha_spread), max(alpha_spread)],
            "t99_180C_range_min": [min(t99_spread), max(t99_spread)]}


def hertz_sensitivity() -> dict:
    nf = 4.8e8
    rows = []
    for L in (0.25, 0.35, 0.45):
        for share in (0.33, 0.5, 1.0):
            h = ims_hertz_stress(radial_load_lbf=6000.0 * share, roller_length_in=L)
            rows.append({"roller_len_in": L, "load_share": share,
                         "sigma_H_GPa": round(h.sigma_h_pa / 1e9, 2),
                         "sigma_f_prime_GPa_at_b-0.0713":
                             round(calibrate_sigma_f_prime(nf, h.sigma_h_pa, -0.0713) / 1e9, 1)})
    return {"grid": rows,
            "reading": "sigma_H scales ~sqrt(share/L); the sigma_f' calibration "
                       "absorbs the scale exactly (same N_f anchor), so damage "
                       "D and hours-to-replacement are INSENSITIVE to these "
                       "two (c) choices — only absolute stress values shift"}


def ea_sensitivity() -> dict:
    rows = []
    for ea in (10_000.0, 12_000.0, 14_000.0):
        p = CureParams(ea_over_r=ea)
        A = calibrated_prefactor(p)
        t180, t210 = _t99_at(A, 180.0, p) / 60, _t99_at(A, 210.0, p) / 60
        rows.append({"Ea_over_R_K": ea, "t99_180C_min": round(t180, 1),
                     "t99_210C_min": round(t210, 1),
                     "ratio_180_over_210": round(t180 / t210, 2)})
    return {"grid": rows,
            "reading": "A is re-anchored at t99(195C)=12min for every Ea/R, so "
                       "only the temperature SENSITIVITY changes; the patent's "
                       "range spans the ratio shown — flag when citing exact "
                       "cold-end cycle times"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    print("== A1. C-MAPSS knee analysis ==", flush=True)
    res["A1_cmapss_knee"] = cmapss_knee_analysis()
    print(json.dumps(res["A1_cmapss_knee"], indent=2))
    print("\n== A3. vibration pseudo-label variants ==", flush=True)
    res["A3_pseudolabels"] = pseudolabel_variants()
    print(json.dumps(res["A3_pseudolabels"], indent=2))
    print("\n== A4. (c)-constant sensitivity ==", flush=True)
    res["A4_kamal"] = kamal_sensitivity()
    print("kamal alpha_end range:", res["A4_kamal"]["alpha_end_range_across_grid"],
          "t99(180) range:", res["A4_kamal"]["t99_180C_range_min"])
    res["A4_hertz"] = hertz_sensitivity()
    print(json.dumps(res["A4_hertz"]["grid"], indent=2))
    res["A4_ea"] = ea_sensitivity()
    print(json.dumps(res["A4_ea"]["grid"], indent=2))
    print("\n== A2. gradient conflict (loads all datasets) ==", flush=True)
    res["A2_gradient_conflict"] = gradient_conflict()
    print(json.dumps(res["A2_gradient_conflict"], indent=2))
    (OUT / "stress_test.json").write_text(json.dumps(res, indent=2))
    print(f"\nSaved {OUT / 'stress_test.json'}")


if __name__ == "__main__":
    main()
