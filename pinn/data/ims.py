"""NASA IMS bearing dataset loader (vibration head — REAL run-to-failure data).

Source: IMS/University of Cincinnati bearing prognostics data (Qiu et al.
2006). 4 Rexnord ZA-2115 bearings on one shaft at 2000 rpm under 6000 lbs
radial load, run until physical failure. Three test sets; each is a folder of
ASCII snapshot files (one file = 1 s of vibration at 20,480 Hz, recorded every
~10 min; filename is the timestamp `YYYY.MM.DD.HH.MM.SS`):

- 1st_test: 8 channels (2 accelerometers x 4 bearings). Failures: bearing 3
  (inner race), bearing 4 (roller).
- 2nd_test: 4 channels (1 per bearing). Failure: bearing 1 (outer race).
- 3rd_test: 4 channels. Failure: bearing 3 (outer race).

Role in the model: CWRU teaches *what* each fault class looks like; IMS
teaches *what the approach to failure looks like over time*. The head's
`health` output is supervised here with a lifetime-position proxy label:

    health_target(file_i) = i / (N_files - 1)   in [0, 1]

i.e. 0 = start of test (healthy), 1 = last snapshot before failure. This is a
proxy (the true internal damage state is unobservable), standard practice for
run-to-failure sets without intermediate inspections; it is *monotonic by
construction*, which matches the irreversible-damage assumption documented in
pinn.physics.rul.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FS_HZ = 20_480.0
SHAFT_RPM = 2000.0
# NOTE: the official archive extracts the *third* test into a folder named
# "4th_test" (known IMS packaging quirk) — key below matches the on-disk name.
CHANNELS = {"1st_test": 8, "2nd_test": 4, "4th_test": 4}
# Channel (0-based) that actually fails per set, per the dataset readme:
FAILING_CHANNEL = {"1st_test": 4, "2nd_test": 0, "4th_test": 2}  # b3(IR)->acc ch5? see note
# NOTE (flagged, not silently assumed): for 1st_test the readme maps bearing 3
# to channels 5-6 (1-based) and bearing 4 to 7-8. We use channel 5 (index 4)
# for bearing 3's inner-race failure. Verify against the PDF readme in
# data/ims_7z before any demo claim.


def list_snapshots(set_dir: str | Path) -> list[Path]:
    """Snapshot files sorted chronologically (filenames are timestamps).

    The third test ships as `4th_test/txt/<timestamps>` — descend into the
    `txt` subfolder when present.
    """
    d = Path(set_dir)
    if (d / "txt").is_dir():
        d = d / "txt"
    return sorted(p for p in d.iterdir() if p.is_file())


def load_snapshot(path: str | Path, n_channels: int) -> np.ndarray:
    """One ASCII snapshot -> array [20480, n_channels] (pandas: ~10x np.loadtxt)."""
    import pandas as pd

    arr = pd.read_csv(path, sep="\t", header=None).to_numpy(dtype=np.float32)
    if arr.ndim == 1 or arr.shape[1] != n_channels:
        arr = arr.reshape(-1, n_channels)
    return arr


def load_health_windows(root: str | Path = "data/ims", test_set: str = "2nd_test",
                        window: int = 2048, stride_files: int = 5,
                        windows_per_file: int = 2, seed: int = 3):
    """Windows + lifetime-position health targets from one run-to-failure set.

    stride_files subsamples the ~1000 snapshots (every ~10 min) to keep v0
    CPU-friendly; windows_per_file random windows are cut from each kept
    snapshot's failing-bearing channel.

    Returns (X [N, window], health [N] in 0..1, file_index [N]).
    """
    root = Path(root) / test_set
    files = list_snapshots(root)
    n_ch = CHANNELS[test_set]
    ch = FAILING_CHANNEL[test_set]
    rng = np.random.default_rng(seed)

    X, health, fidx = [], [], []
    kept = files[::stride_files]
    for i, f in enumerate(kept):
        sig = load_snapshot(f, n_ch)[:, ch]
        pos = (i * stride_files) / max(1, len(files) - 1)
        for _ in range(windows_per_file):
            s = int(rng.integers(0, len(sig) - window))
            X.append(sig[s:s + window])
            health.append(min(1.0, pos))
            fidx.append(i * stride_files)
    return (np.stack(X).astype(np.float32),
            np.asarray(health, dtype=np.float32),
            np.asarray(fidx))
