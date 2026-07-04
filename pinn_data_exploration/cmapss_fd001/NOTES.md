# NASA C-MAPSS FD001 — exploration notes

**Provenance: SIMULATED** (NASA high-fidelity turbofan simulation,
run-to-failure). Stand-in for "how much life is left" — the degradation
trajectory shape (gradual drift accelerating toward failure) is what the
**degradation/RUL head** must learn; a curing press is not a jet engine and we
never claim it is.

## Verification vs brief
- Download link worked **as given** (zip-inside-zip, `CMAPSSData.zip` extracted).
- train_FD001.txt: **20631** rows (expected 20,631) — OK
- test_FD001.txt: **13096** rows (expected 13,096) — OK
- RUL_FD001.txt: **100** rows (expected 100) — OK
- 26 space-separated columns, no header — OK; **100** train engines.
- Missing values: 0.

## Column-drop finding (brief discrepancy — flagged)
Exactly constant (std=0): **op_setting_3, sensor_1, sensor_5, sensor_10, sensor_16, sensor_18, sensor_19** — matches the brief's list.
BUT dropping only those leaves **15** sensors, not the "14 informative
sensors" the brief announces. `sensor_6` is near-constant
(std=0.0014; near-constant set: op_setting_1, op_setting_2, sensor_6)
and is dropped too in `pinn/data/cmapss.py` — this reconciles the count and
matches common FD001 practice. **Decision to confirm with the team.**

## Column statistics (train)
| column | min | max | mean | missing |
|---|---|---|---|---|
| unit | 1 | 100 | 51.51 | 0 |
| cycle | 1 | 362 | 108.8 | 0 |
| op_setting_1 | -0.0087 | 0.0087 | -8.87e-06 | 0 |
| op_setting_2 | -0.0006 | 0.0006 | 2.351e-06 | 0 |
| op_setting_3 | 100 | 100 | 100 | 0 |
| sensor_1 | 518.7 | 518.7 | 518.7 | 0 |
| sensor_2 | 641.2 | 644.5 | 642.7 | 0 |
| sensor_3 | 1571 | 1617 | 1591 | 0 |
| sensor_4 | 1382 | 1441 | 1409 | 0 |
| sensor_5 | 14.62 | 14.62 | 14.62 | 0 |
| sensor_6 | 21.6 | 21.61 | 21.61 | 0 |
| sensor_7 | 549.9 | 556.1 | 553.4 | 0 |
| sensor_8 | 2388 | 2389 | 2388 | 0 |
| sensor_9 | 9022 | 9245 | 9065 | 0 |
| sensor_10 | 1.3 | 1.3 | 1.3 | 0 |
| sensor_11 | 46.85 | 48.53 | 47.54 | 0 |
| sensor_12 | 518.7 | 523.4 | 521.4 | 0 |
| sensor_13 | 2388 | 2389 | 2388 | 0 |
| sensor_14 | 8100 | 8294 | 8144 | 0 |
| sensor_15 | 8.325 | 8.585 | 8.442 | 0 |
| sensor_16 | 0.03 | 0.03 | 0.03 | 0 |
| sensor_17 | 388 | 400 | 393.2 | 0 |
| sensor_18 | 2388 | 2388 | 2388 | 0 |
| sensor_19 | 100 | 100 | 100 | 0 |
| sensor_20 | 38.14 | 39.43 | 38.82 | 0 |
| sensor_21 | 22.89 | 23.62 | 23.29 | 0 |

## Plot
`cmapss_degradation.png` — sensor_2 drift for 4 engines + lifetime histogram.
