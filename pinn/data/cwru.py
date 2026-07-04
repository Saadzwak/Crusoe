"""CWRU bearing dataset loader (vibration head — REAL measured data).

Source layout (GitHub mirror s-whynot/CWRU-dataset, canonical CWRU file IDs):

    12k_Drive_End_Bearing_Fault_Data/<B|IR|OR>/<007|014|021|028>[/@3|@6|@12]/<id>_<load>.mat
    12k_Fan_End_Bearing_Fault_Data/...
    48k_Drive_End_Bearing_Fault_Data/...
    Normal/<id>_Normal_<load>.mat

Each .mat holds X<id>_DE_time / _FE_time / _BA_time accelerometer channels and
X<id>RPM (measured shaft speed). Fault diameters are in mils (0.007"...0.028"),
loads 0-3 HP, OR faults additionally have a clock position (@3/@6/@12 relative
to the load zone).

For each window we also precompute the *envelope-spectrum band energies* at
the bearing fault characteristic frequencies (BPFO/BPFI/BSF from
pinn.physics.bearing, using the file's own measured RPM). Envelope (Hilbert)
spectrum rather than raw FFT because localized bearing faults excite
high-frequency resonances amplitude-modulated at the fault frequency — the
fault signature lives in the envelope, not the raw spectrum (Randall & Antoni
2011). These band energies feed the physics-consistency loss.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import hilbert

from pinn.physics.bearing import CWRU_DRIVE_END_6205, fault_frequencies

CLASSES = ["Normal", "IR", "OR", "B"]  # head output order
# Nominal shaft speeds per motor load (HP 0..3), fallback if RPM key missing.
NOMINAL_RPM = {0: 1797.0, 1: 1772.0, 2: 1750.0, 3: 1730.0}


@dataclass
class CwruFile:
    path: Path
    subset: str          # "12k_DE" | "12k_FE" | "48k_DE" | "Normal"
    fs_hz: float
    label: int           # index into CLASSES
    fault_diam_mils: int | None
    load_hp: int
    or_position: str | None
    rpm: float


def _parse_file(path: Path, root: Path) -> CwruFile | None:
    rel = path.relative_to(root).as_posix()
    m_id = re.match(r"(\d+)_", path.name)
    if not m_id:
        return None
    load = int(re.search(r"_(\d)\.mat$", path.name).group(1))
    if rel.startswith("Normal/"):
        return CwruFile(path, "Normal", 12_000.0, CLASSES.index("Normal"), None, load, None, 0.0)
    m = re.match(r"(12k_Drive_End|12k_Fan_End|48k_Drive_End)[^/]*/(B|IR|OR)/(\d{3})(?:/@(\d+))?/", rel)
    if not m:
        return None
    subset_raw, fault, diam, orpos = m.groups()
    fs = 48_000.0 if subset_raw.startswith("48k") else 12_000.0
    subset = {"12k_Drive_End": "12k_DE", "12k_Fan_End": "12k_FE", "48k_Drive_End": "48k_DE"}[subset_raw]
    return CwruFile(path, subset, fs, CLASSES.index(fault), int(diam), load,
                    f"@{orpos}" if orpos else None, 0.0)


def inventory(root: str | Path = "data/cwru") -> list[CwruFile]:
    """Walk the mirror and parse every .mat file's metadata from its path."""
    root = Path(root)
    files = []
    for p in sorted(root.rglob("*.mat")):
        f = _parse_file(p, root)
        if f is not None:
            files.append(f)
    return files


def _load_channel(path: Path, channel: str = "DE") -> tuple[np.ndarray, float]:
    """Return (signal, rpm) for one file; rpm falls back to nominal-by-load."""
    mat = sio.loadmat(str(path))
    sig_key = next((k for k in mat if k.endswith(f"_{channel}_time")), None)
    if sig_key is None:  # some files only have one channel
        sig_key = next(k for k in mat if k.endswith("_time"))
    rpm_key = next((k for k in mat if k.endswith("RPM")), None)
    load = int(re.search(r"_(\d)\.mat$", path.name).group(1))
    rpm = float(mat[rpm_key].squeeze()) if rpm_key else NOMINAL_RPM[load]
    return mat[sig_key].squeeze().astype(np.float32), rpm


def envelope_band_energies(window: np.ndarray, fs_hz: float, rpm: float,
                           n_harmonics: int = 2) -> np.ndarray:
    """Energy of the envelope spectrum in bands around BPFO/BPFI/BSF.

    Band = f0*h +/- max(3 Hz, 2% of f0*h) for harmonics h=1..n_harmonics
    (2% tolerance covers normal bearing slip). Returns [E_BPFO, E_BPFI, E_BSF],
    each normalized by total envelope-spectrum energy so the feature is
    amplitude-invariant.
    """
    env = np.abs(hilbert(window - window.mean()))
    env -= env.mean()
    spec = np.abs(np.fft.rfft(env)) ** 2
    freqs = np.fft.rfftfreq(len(window), d=1.0 / fs_hz)
    total = spec.sum() + 1e-12

    shaft_hz = rpm / 60.0
    fdict = fault_frequencies(CWRU_DRIVE_END_6205, shaft_hz)
    energies = []
    for key in ("BPFO", "BPFI", "BSF"):
        e = 0.0
        for h in range(1, n_harmonics + 1):
            f0 = fdict[key] * h
            tol = max(3.0, 0.02 * f0)
            band = (freqs >= f0 - tol) & (freqs <= f0 + tol)
            e += spec[band].sum()
        energies.append(e / total)
    return np.asarray(energies, dtype=np.float32)


def load_windows(root: str | Path = "data/cwru", subset: str = "12k_DE",
                 window: int = 2048, hop: int = 2048, channel: str = "DE",
                 include_normal: bool = True, fe_targets: bool = False,
                 cache: bool = True):
    """Windowed dataset: (X [N,window], y [N], band_energies [N,3], rpm [N], meta).

    v0/v1 default trains on the 12k drive-end grid + Normal baseline; the 12k
    fan-end and 48k subsets are inventoried and verified but not yet used for
    training (kept for a later robustness pass — documented in NOTES).

    fe_targets=True additionally returns, for every drive-end window, the
    time-aligned FAN-END channel's envelope band energies [N,3] and RMS [N] —
    the withheld-sensor ground truth for the virtual-sensor demo (REAL
    measurements, never seen as model input). Files lacking an FE channel are
    skipped in that mode.

    cache=True memoizes the windowed arrays to an .npz beside the data (the
    .mat walk + Hilbert transforms cost ~5 min; diagnostics re-load often).
    NOTE: the per-window `meta` list is NOT cached — cache hits return
    meta=None (no current training/diagnostic path uses it).
    """
    root = Path(root)
    cache_key = f"{subset}_{window}_{hop}_{channel}_{int(include_normal)}_{int(fe_targets)}"
    cache_path = root / f"_windows_cache_{cache_key}.npz"
    if cache and cache_path.exists():
        z = np.load(cache_path)
        out = (z["X"], z["y"], z["bands"], z["rpms"], None)
        if fe_targets:
            return out + (z["fe_bands"], z["fe_rms"])
        return out

    files = [f for f in inventory(root)
             if f.subset == subset or (include_normal and f.subset == "Normal")]
    X, y, bands, rpms, meta = [], [], [], [], []
    fe_bands, fe_rms = [], []
    for f in files:
        sig, rpm = _load_channel(f.path, channel)
        fe_sig = None
        if fe_targets:
            mat_keys = sio.loadmat(str(f.path)).keys()
            if not any(k.endswith("_FE_time") for k in mat_keys):
                continue
            fe_sig, _ = _load_channel(f.path, "FE")
            n_usable = min(len(sig), len(fe_sig))
            sig = sig[:n_usable]
        for start in range(0, len(sig) - window + 1, hop):
            w = sig[start:start + window]
            X.append(w)
            y.append(f.label)
            bands.append(envelope_band_energies(w, f.fs_hz, rpm))
            rpms.append(rpm)
            meta.append(f)
            if fe_sig is not None:
                wf = fe_sig[start:start + window]
                fe_bands.append(envelope_band_energies(wf, f.fs_hz, rpm))
                fe_rms.append(float(np.sqrt(np.mean(wf.astype(np.float64) ** 2))))
    out = (np.stack(X), np.asarray(y, dtype=np.int64),
           np.stack(bands), np.asarray(rpms, dtype=np.float32), meta)
    if fe_targets:
        out = out + (np.stack(fe_bands),
                     np.asarray(fe_rms, dtype=np.float32))
    if cache:
        arrays = {"X": out[0], "y": out[1], "bands": out[2], "rpms": out[3]}
        if fe_targets:
            arrays.update(fe_bands=out[5], fe_rms=out[6])
        np.savez_compressed(cache_path, **arrays)
    return out
