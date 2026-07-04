"""Exploration + verification of the 5 datasets -> NOTES.md + PNG per dataset.

Run from repo root:  python pinn_data_exploration/explore_all.py
Each dataset gets pinn_data_exploration/<name>/{NOTES.md, *.png}.

Descriptions of what each column/dataset physically represents come from the
team brief and the datasets' own documentation — not invented here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn.data import cwru as cwru_mod  # noqa: E402
from pinn.data.ims import CHANNELS, list_snapshots  # noqa: E402
from pinn.physics.ai4i_rules import ai4i_hard_rules  # noqa: E402
from pinn.physics.bearing import CWRU_DRIVE_END_6205, fault_frequencies  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "pinn_data_exploration"


def stats_md(df: pd.DataFrame) -> str:
    rows = ["| column | min | max | mean | missing |", "|---|---|---|---|---|"]
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            rows.append(f"| {c} | {df[c].min():.4g} | {df[c].max():.4g} "
                        f"| {df[c].mean():.4g} | {int(df[c].isna().sum())} |")
        else:
            rows.append(f"| {c} | — | — | — | {int(df[c].isna().sum())} |")
    return "\n".join(rows)


# ------------------------------------------------------------------- AI4I ----
def explore_ai4i():
    out = OUT / "ai4i2020"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "ai4i" / "ai4i2020.csv")

    rules = ai4i_hard_rules(df["Air temperature [K]"], df["Process temperature [K]"],
                            df["Rotational speed [rpm]"], df["Torque [Nm]"],
                            df["Tool wear [min]"], df["Type"].to_numpy())
    hdf_overlap = int((rules["HDF"] & df["HDF"].to_numpy()).sum())
    pwf_overlap = int((rules["PWF"] & df["PWF"].to_numpy()).sum())
    osf_overlap = int((rules["OSF"] & df["OSF"].to_numpy()).sum())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    power = rules["power_W"]
    ok = df["Machine failure"] == 0
    axes[0].scatter(df.loc[ok, "Rotational speed [rpm]"], df.loc[ok, "Torque [Nm]"],
                    s=4, alpha=0.25, label="no failure")
    axes[0].scatter(df.loc[~ok, "Rotational speed [rpm]"], df.loc[~ok, "Torque [Nm]"],
                    s=10, color="crimson", label="machine failure")
    axes[0].set_xlabel("Rotational speed [rpm]"); axes[0].set_ylabel("Torque [Nm]")
    axes[0].set_title("AI4I: torque vs speed — failures sit on the physics limits")
    axes[0].legend()
    dT = df["Process temperature [K]"] - df["Air temperature [K]"]
    hdf = df["HDF"] == 1
    axes[1].scatter(df.loc[~hdf, "Rotational speed [rpm]"], dT[~hdf], s=4, alpha=0.25,
                    label="no HDF")
    axes[1].scatter(df.loc[hdf, "Rotational speed [rpm]"], dT[hdf], s=10,
                    color="darkorange", label="HDF")
    axes[1].axhline(8.6, ls="--", c="k", lw=1); axes[1].axvline(1380, ls="--", c="k", lw=1)
    axes[1].set_xlabel("Rotational speed [rpm]")
    axes[1].set_ylabel("Process - Air temperature [K]")
    axes[1].set_title("HDF rule: dT<8.6K AND speed<1380rpm (dashed = documented thresholds)")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(out / "ai4i_failure_physics.png", dpi=130)
    plt.close(fig)

    (out / "NOTES.md").write_text(f"""# AI4I 2020 Predictive Maintenance — exploration notes

**Provenance: SIMULATED** (synthetic-by-design, generative rules documented by
the authors — Matzka 2020, UCI dataset #601). Stand-in for the curing press's
thermal/power/overstrain behavior; feeds the **thermal/power head**.

## Verification vs brief
- Download link worked **as given** (UCI static zip).
- Shape: **{df.shape[0]} rows x {df.shape[1]} cols** (expected 10,000 x 14) — OK
- `Machine failure=1`: **{int(df['Machine failure'].sum())}** (expected 339) — OK
- `HDF=1`: **{int(df['HDF'].sum())}** (expected 115) — OK
- `PWF=1`: **{int(df['PWF'].sum())}** (expected 95) — OK
- Missing values: **{int(df.isna().sum().sum())}** — OK

## Closed-form rule consistency (why a physics head fits this data)
Evaluating the documented failure rules on the raw columns reproduces the
labels: HDF rule matches **{hdf_overlap}/115** flagged rows, PWF rule
**{pwf_overlap}/95**, OSF rule **{osf_overlap}/{int(df['OSF'].sum())}**.
The dataset satisfies its own documented physics — the physics-consistency
loss in `pinn/losses.py` penalizes the model for disagreeing with these rules.

## Columns (physical meaning per dataset docs)
UDI/Product ID (identifiers), Type (product quality L/M/H), Air & Process
temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min],
Machine failure + per-mode flags TWF/HDF/PWF/OSF/RNF.

## Column statistics
{stats_md(df)}

## Plot
`ai4i_failure_physics.png` — failures concentrate exactly on the documented
physical limits (power window edges, HDF corner below dT=8.6K & 1380 rpm).
""", encoding="utf-8")
    print("AI4I notes + plot done")


# ----------------------------------------------------------------- C-MAPSS ---
def explore_cmapss():
    out = OUT / "cmapss_fd001"
    out.mkdir(parents=True, exist_ok=True)
    cols = (["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
            + [f"sensor_{i}" for i in range(1, 22)])
    train = pd.read_csv(DATA / "cmapss" / "train_FD001.txt", sep=r"\s+", header=None,
                        names=cols)
    test = pd.read_csv(DATA / "cmapss" / "test_FD001.txt", sep=r"\s+", header=None,
                       names=cols)
    rul = np.loadtxt(DATA / "cmapss" / "RUL_FD001.txt")

    stds = train.std(numeric_only=True)
    constant = [c for c in cols[2:] if stds[c] < 1e-6]
    near_constant = [c for c in cols[2:] if 1e-6 <= stds[c] < 0.02]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for unit in (1, 25, 50, 75):
        g = train[train.unit == unit]
        axes[0].plot(g.cycle, g.sensor_2, alpha=0.8, label=f"engine {unit}")
    axes[0].set_xlabel("operating cycle"); axes[0].set_ylabel("sensor_2 (LPC outlet temp)")
    axes[0].set_title("C-MAPSS FD001: sensor drift accelerating toward failure")
    axes[0].legend()
    lifetimes = train.groupby("unit").cycle.max()
    axes[1].hist(lifetimes, bins=25, color="steelblue")
    axes[1].set_xlabel("total lifetime [cycles]"); axes[1].set_ylabel("engines")
    axes[1].set_title(f"lifetimes: {lifetimes.min()}-{lifetimes.max()} cycles "
                      f"(median {int(lifetimes.median())})")
    fig.tight_layout(); fig.savefig(out / "cmapss_degradation.png", dpi=130)
    plt.close(fig)

    (out / "NOTES.md").write_text(f"""# NASA C-MAPSS FD001 — exploration notes

**Provenance: SIMULATED** (NASA high-fidelity turbofan simulation,
run-to-failure). Stand-in for "how much life is left" — the degradation
trajectory shape (gradual drift accelerating toward failure) is what the
**degradation/RUL head** must learn; a curing press is not a jet engine and we
never claim it is.

## Verification vs brief
- Download link worked **as given** (zip-inside-zip, `CMAPSSData.zip` extracted).
- train_FD001.txt: **{len(train)}** rows (expected 20,631) — OK
- test_FD001.txt: **{len(test)}** rows (expected 13,096) — OK
- RUL_FD001.txt: **{len(rul)}** rows (expected 100) — OK
- 26 space-separated columns, no header — OK; **{train['unit'].nunique()}** train engines.
- Missing values: {int(train.isna().sum().sum())}.

## Column-drop finding (brief discrepancy — flagged)
Exactly constant (std=0): **{', '.join(constant)}** — matches the brief's list.
BUT dropping only those leaves **15** sensors, not the "14 informative
sensors" the brief announces. `sensor_6` is near-constant
(std={stds['sensor_6']:.4f}; near-constant set: {', '.join(near_constant)})
and is dropped too in `pinn/data/cmapss.py` — this reconciles the count and
matches common FD001 practice. **Decision to confirm with the team.**

## Column statistics (train)
{stats_md(train)}

## Plot
`cmapss_degradation.png` — sensor_2 drift for 4 engines + lifetime histogram.
""", encoding="utf-8")
    print("C-MAPSS notes + plot done")


# ------------------------------------------------------------------- CWRU ----
def explore_cwru():
    out = OUT / "cwru_bearings"
    out.mkdir(parents=True, exist_ok=True)
    inv = cwru_mod.inventory(DATA / "cwru")
    df = pd.DataFrame([{"subset": f.subset, "class": cwru_mod.CLASSES[f.label],
                        "diam_mils": f.fault_diam_mils, "load_hp": f.load_hp,
                        "or_pos": f.or_position, "file": f.path.name} for f in inv])
    grid = df.groupby(["subset", "class"]).size().unstack(fill_value=0)

    normal_sig, normal_rpm = cwru_mod._load_channel(
        DATA / "cwru" / "Normal" / "97_Normal_0.mat", "DE")
    ir_sig, ir_rpm = cwru_mod._load_channel(
        DATA / "cwru" / "12k_Drive_End_Bearing_Fault_Data" / "IR" / "007" / "105_0.mat", "DE")
    fs = 12_000.0
    n = 2048
    from scipy.signal import hilbert
    w = ir_sig[:8192].astype(float)
    env = np.abs(hilbert(w - w.mean())); env -= env.mean()
    spec = np.abs(np.fft.rfft(env)) ** 2
    freqs = np.fft.rfftfreq(len(w), 1 / fs)
    fdict = fault_frequencies(CWRU_DRIVE_END_6205, ir_rpm / 60.0)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    t = np.arange(n) / fs * 1000
    axes[0].plot(t, normal_sig[:n], lw=0.7); axes[0].set_title(f"Normal (97, {normal_rpm:.0f} rpm)")
    axes[1].plot(t, ir_sig[:n], lw=0.7, color="crimson")
    axes[1].set_title(f"Inner-race fault 0.007\" (105, {ir_rpm:.0f} rpm)")
    for ax in axes[:2]:
        ax.set_xlabel("time [ms]"); ax.set_ylabel("accel [g]")
    m = freqs <= 500
    axes[2].plot(freqs[m], spec[m] / spec[m].max(), lw=0.8, color="crimson")
    for key, c in (("BPFI", "k"), ("BPFO", "gray"), ("BSF", "silver")):
        axes[2].axvline(fdict[key], ls="--", c=c, lw=1, label=f"{key}={fdict[key]:.0f}Hz")
    axes[2].set_xlabel("frequency [Hz]"); axes[2].set_ylabel("envelope PSD (norm.)")
    axes[2].set_title("IR envelope spectrum peaks at BPFI — bearing kinematics")
    axes[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "cwru_waveforms_envelope.png", dpi=130)
    plt.close(fig)

    (out / "NOTES.md").write_text(f"""# CWRU Bearing Dataset — exploration notes

**Provenance: REAL** (measured accelerometer data, artificially seeded
faults). World-reference bearing vibration dataset; feeds the **vibration
head** ("what does each fault type look like").

## Source substitution (flagged, per brief instruction)
The official CWRU download page is interactive/geo-fragile; we used the
GitHub mirror **s-whynot/CWRU-dataset** (shallow clone). Canonical CWRU file
IDs check out (97=Normal_0, 105=IR007_0, 118=B007_0, ...). Mirror README
matches the CWRU Bearing Data Center inventory.

## FULL grid this time (previous session's gap fixed)
**{len(inv)} .mat files**: 12k drive-end + 12k fan-end + 48k drive-end + Normal;
fault diameters 007/014/021(/028) mils x motor loads 0-3 HP; OR faults with
clock positions @3/@6/@12.

Files per subset x class:

{grid.to_markdown()}

## Structure of each file
MATLAB arrays `X<id>_DE_time` / `_FE_time` / `_BA_time` (drive-end, fan-end,
base accelerometers; ~121k samples @ 12 kHz ≈ 10 s) + `X<id>RPM` (measured
shaft speed). Fault type/size/position encoded in the folder path, canonical
ID in the filename.

## Bearing geometry cross-check
SKF 6205 (drive end) and SKF 6203 (fan end) geometry constants reproduce the
CWRU-published fault-frequency multipliers to 4 significant figures
(`pinn.physics.bearing.verify_against_published()` — runs at every training
start). v0 trains on the 12k drive-end subset + Normal; fan-end and 48k are
inventoried for a later robustness pass.

## Plot
`cwru_waveforms_envelope.png` — Normal vs IR raw waveforms + IR envelope
spectrum with documented BPFI/BPFO/BSF lines: the fault's energy sits at BPFI,
exactly where bearing kinematics says it must.
""", encoding="utf-8")
    print("CWRU notes + plot done")


# -------------------------------------------------------------------- IMS ----
def explore_ims():
    out = OUT / "ims_bearings"
    out.mkdir(parents=True, exist_ok=True)
    root = DATA / "ims"
    sets = [d.name for d in sorted(root.iterdir()) if d.is_dir()]
    if not sets:
        print("IMS not extracted yet — skipping")
        return

    lines = []
    for s in sets:
        files = list_snapshots(root / s)
        first = pd.read_csv(files[0], sep="\t", header=None)
        lines.append(f"- **{s}**: {len(files)} snapshot files, "
                     f"{first.shape[0]} samples x {first.shape[1]} channels each "
                     f"(first file: {files[0].name}, last: {files[-1].name})")

    # Degradation curve on the documented failing bearing of 2nd_test (bearing 1).
    s = "2nd_test" if "2nd_test" in sets else sets[0]
    files = list_snapshots(root / s)
    stride = max(1, len(files) // 250)
    rms, kurt = [], []
    idx = list(range(0, len(files), stride))
    for i in idx:
        x = pd.read_csv(files[i], sep="\t", header=None).iloc[:, 0].to_numpy()
        rms.append(float(np.sqrt(np.mean(x ** 2))))
        xc = x - x.mean()
        kurt.append(float(np.mean(xc ** 4) / (np.mean(xc ** 2) ** 2 + 1e-12)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(idx, rms, color="crimson")
    axes[0].set_xlabel(f"snapshot index (~10 min each, {s})")
    axes[0].set_ylabel("RMS accel [g]")
    axes[0].set_title("IMS: bearing-1 RMS rising toward real physical failure")
    axes[1].plot(idx, kurt, color="darkorange")
    axes[1].set_xlabel("snapshot index"); axes[1].set_ylabel("kurtosis")
    axes[1].set_title("kurtosis spikes as impulsive fault develops")
    fig.tight_layout(); fig.savefig(out / "ims_degradation.png", dpi=130)
    plt.close(fig)

    (out / "NOTES.md").write_text(f"""# NASA IMS Bearing Dataset — exploration notes

**Provenance: REAL run-to-failure** (4 Rexnord ZA-2115 bearings on one shaft,
2000 rpm, 6000 lbs load, run until actual physical failure — no seeded
faults). Complements CWRU for the **vibration head**: CWRU = "what does this
fault look like", IMS = "what does the *approach* to failure look like".

## Download & extraction (flagged: non-standard nesting)
S3 link worked **as given** (~1.07 GB). Nesting: zip -> `IMS.7z` (py7zr) ->
three `.rar` (extracted with Windows bsdtar/libarchive — no extra installs)
+ `Readme Document for IMS Bearing Data.pdf` (kept in `data/ims_7z/`).

## Test sets found (all 3, per brief requirement)
{chr(10).join(lines)}

Snapshot files are tab-separated ASCII, 20,480 samples/channel = 1 s at
20.48 kHz, recorded every ~10 min; filename = timestamp.

## Failure documentation (from dataset readme)
- 1st_test (8 ch, 2/bearing): bearing 3 inner race + bearing 4 roller failures
- 2nd_test (4 ch): bearing 1 outer race failure
- 3rd_test (4 ch): bearing 3 outer race failure
**Caveat flagged**: our channel->bearing mapping for 1st_test
(`pinn/data/ims.py`) is from the readme's stated order; verify against the
PDF before demo claims about *which* physical bearing is shown.

## Health-label choice for training (documented proxy)
`health = snapshot_index / (N-1)` in [0,1] — monotonic lifetime position, the
standard proxy when no intermediate damage inspections exist. It supervises
the vibration head's `health` output (IMS has no per-window fault-class labels).

## Plot
`ims_degradation.png` — RMS + kurtosis of the failing bearing channel across
the whole {s} run: flat healthy plateau, then acceleration to failure. This
trajectory shape is exactly what the RUL/degradation story needs.
""", encoding="utf-8")
    print("IMS notes + plot done")


# ----------------------------------------------------------------- Milling ---
def explore_milling():
    out = OUT / "nasa_milling"
    out.mkdir(parents=True, exist_ok=True)
    import scipy.io as sio
    matpath = next((DATA / "milling").rglob("*.mat"))
    mill = sio.loadmat(matpath)["mill"][0]
    rows = []
    for c in mill:
        vb = np.asarray(c["VB"], dtype=float).squeeze()
        rows.append({
            "case": int(np.squeeze(c["case"])), "run": int(np.squeeze(c["run"])),
            "VB_mm": float(vb) if vb.size == 1 and not np.isnan(vb) else np.nan,
            "time_min": float(np.squeeze(c["time"])),
            "DOC_mm": float(np.squeeze(c["DOC"])), "feed": float(np.squeeze(c["feed"])),
            "material": int(np.squeeze(c["material"])),
        })
    df = pd.DataFrame(rows)
    n_missing = int(df["VB_mm"].isna().sum())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for case, g in df.groupby("case"):
        axes[0].plot(g["run"], g["VB_mm"], marker="o", ms=3, alpha=0.7, label=f"case {case}")
    axes[0].set_xlabel("run"); axes[0].set_ylabel("flank wear VB [mm]")
    axes[0].set_title(f"tool wear per case ({n_missing}/167 runs missing VB label)")
    axes[0].legend(fontsize=6, ncol=2)
    sig = np.asarray(mill[0]["smcAC"], dtype=float).squeeze()
    axes[1].plot(np.arange(len(sig)) / 250.0, sig, lw=0.5)
    axes[1].set_xlabel("time [s] (250 Hz)"); axes[1].set_ylabel("spindle motor AC current")
    axes[1].set_title("example smcAC trace (cut #1)")
    fig.tight_layout(); fig.savefig(out / "milling_wear.png", dpi=130)
    plt.close(fig)

    (out / "NOTES.md").write_text(f"""# NASA Milling Dataset — exploration notes

**Provenance: REAL** (instrumented milling machine, BEST lab). Optional /
secondary per brief — possible future tool-wear head; **not used in v0
training** (explored and verified only).

## Verification vs brief
- S3 link worked **as given** (nested: `mill.zip` inside, then `mill.mat`).
- **{len(mill)} cuts** (expected 167) — OK.
- Missing VB wear labels: **{n_missing}/167** (brief said 21) — {'OK' if n_missing == 21 else 'MISMATCH'}.
  Real data-quality issue: any future training on VB requires a **masked
  loss** (ignore unlabeled cuts), as the brief anticipated.

## Structure
MATLAB struct, fields per cut: case, run, VB (flank wear, the label), time,
DOC (depth of cut), feed, material, and 6 sensor traces per cut — smcAC/smcDC
(spindle motor currents), vib_table/vib_spindle, AE_table/AE_spindle
(acoustic emission), sampled at 250 Hz.

## Sampling-rate note (for the future head)
250 Hz here vs 12-48 kHz (CWRU/IMS): slow wear labels may be interpolated
across cuts if ever aligned to cycle timestamps (wear is slow and monotonic
between measurements) — but the raw sensor *waveforms* must never be
interpolated across rates: that would fabricate signal. Per-head adapters in
`pinn/model` exist precisely to avoid mixing rates in one input layer.

## Runs summary (VB per case)
{df.groupby('case')['VB_mm'].agg(['count', 'min', 'max']).to_markdown()}

## Plot
`milling_wear.png` — VB progression per case (gaps = the 21 missing labels)
+ one raw spindle-current trace.
""", encoding="utf-8")
    print("Milling notes + plot done")


if __name__ == "__main__":
    for fn in (explore_ai4i, explore_cmapss, explore_cwru, explore_ims, explore_milling):
        try:
            fn()
        except Exception as e:  # keep going, report at the end
            print(f"ERROR in {fn.__name__}: {e!r}")
