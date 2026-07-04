# NASA IMS Bearing Dataset — exploration notes

**Provenance: REAL run-to-failure** (4 Rexnord ZA-2115 bearings on one shaft,
2000 rpm, 6000 lbs load, run until actual physical failure — no seeded
faults). Complements CWRU for the **vibration head**: CWRU = "what does this
fault look like", IMS = "what does the *approach* to failure look like".

## Download & extraction (flagged: non-standard nesting)
S3 link worked **as given** (~1.07 GB). Nesting: zip -> `IMS.7z` (py7zr) ->
three `.rar` (extracted with Windows bsdtar/libarchive — no extra installs)
+ `Readme Document for IMS Bearing Data.pdf` (kept in `data/ims_7z/`).

## Test sets found (all 3, per brief requirement)
- **1st_test**: 2156 snapshot files, 20480 samples x 8 channels each (first file: 2003.10.22.12.06.24, last: 2003.11.25.23.39.56)
- **2nd_test**: 984 snapshot files, 20480 samples x 4 channels each (first file: 2004.02.12.10.32.39, last: 2004.02.19.06.22.39)
- **4th_test**: 6324 snapshot files, 20480 samples x 4 channels each (first file: 2004.03.04.09.27.46, last: 2004.04.18.02.42.55)

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
the whole 2nd_test run: flat healthy plateau, then acceleration to failure. This
trajectory shape is exactly what the RUL/degradation story needs.
