# NASA Milling Dataset — exploration notes

**Provenance: REAL** (instrumented milling machine, BEST lab). Optional /
secondary per brief — possible future tool-wear head; **not used in v0
training** (explored and verified only).

## Verification vs brief
- S3 link worked **as given** (nested: `mill.zip` inside, then `mill.mat`).
- **167 cuts** (expected 167) — OK.
- Missing VB wear labels: **21/167** (brief said 21) — OK.
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
|   case |   count |   min |   max |
|-------:|--------:|------:|------:|
|      1 |      13 |  0    |  0.5  |
|      2 |      13 |  0.08 |  0.55 |
|      3 |      14 |  0    |  0.55 |
|      4 |       7 |  0.08 |  0.49 |
|      5 |       6 |  0    |  0.74 |
|      6 |       1 |  0    |  0    |
|      7 |       7 |  0    |  0.46 |
|      8 |       5 |  0    |  0.62 |
|      9 |       9 |  0    |  0.81 |
|     10 |      10 |  0    |  0.7  |
|     11 |      20 |  0    |  0.76 |
|     12 |      12 |  0.05 |  0.65 |
|     13 |      13 |  0.1  |  1.53 |
|     14 |       7 |  0.09 |  1.14 |
|     15 |       6 |  0.15 |  0.7  |
|     16 |       3 |  0.24 |  0.62 |

## Plot
`milling_wear.png` — VB progression per case (gaps = the 21 missing labels)
+ one raw spindle-current trace.
