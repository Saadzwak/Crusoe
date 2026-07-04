# CWRU Bearing Dataset — exploration notes

**Provenance: REAL** (measured accelerometer data, artificially seeded
faults). World-reference bearing vibration dataset; feeds the **vibration
head** ("what does each fault type look like").

## Source substitution (flagged, per brief instruction)
The official CWRU download page is interactive/geo-fragile; we used the
GitHub mirror **s-whynot/CWRU-dataset** (shallow clone). Canonical CWRU file
IDs check out (97=Normal_0, 105=IR007_0, 118=B007_0, ...). Mirror README
matches the CWRU Bearing Data Center inventory.

## FULL grid this time (previous session's gap fixed)
**149 .mat files**: 12k drive-end + 12k fan-end + 48k drive-end + Normal;
fault diameters 007/014/021(/028) mils x motor loads 0-3 HP; OR faults with
clock positions @3/@6/@12.

Files per subset x class:

| subset   |   B |   IR |   Normal |   OR |
|:---------|----:|-----:|---------:|-----:|
| 12k_DE   |  16 |   16 |        0 |   20 |
| 12k_FE   |  12 |   12 |        0 |   21 |
| 48k_DE   |  12 |   12 |        0 |   24 |
| Normal   |   0 |    0 |        4 |    0 |

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
