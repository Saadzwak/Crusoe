# AI4I 2020 Predictive Maintenance — exploration notes

**Provenance: SIMULATED** (synthetic-by-design, generative rules documented by
the authors — Matzka 2020, UCI dataset #601). Stand-in for the curing press's
thermal/power/overstrain behavior; feeds the **thermal/power head**.

## Verification vs brief
- Download link worked **as given** (UCI static zip).
- Shape: **10000 rows x 14 cols** (expected 10,000 x 14) — OK
- `Machine failure=1`: **339** (expected 339) — OK
- `HDF=1`: **115** (expected 115) — OK
- `PWF=1`: **95** (expected 95) — OK
- Missing values: **0** — OK

## Closed-form rule consistency (why a physics head fits this data)
Evaluating the documented failure rules on the raw columns reproduces the
labels: HDF rule matches **115/115** flagged rows, PWF rule
**95/95**, OSF rule **98/98**.
The dataset satisfies its own documented physics — the physics-consistency
loss in `pinn/losses.py` penalizes the model for disagreeing with these rules.

## Columns (physical meaning per dataset docs)
UDI/Product ID (identifiers), Type (product quality L/M/H), Air & Process
temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min],
Machine failure + per-mode flags TWF/HDF/PWF/OSF/RNF.

## Column statistics
| column | min | max | mean | missing |
|---|---|---|---|---|
| UDI | 1 | 1e+04 | 5000 | 0 |
| Product ID | — | — | — | 0 |
| Type | — | — | — | 0 |
| Air temperature [K] | 295.3 | 304.5 | 300 | 0 |
| Process temperature [K] | 305.7 | 313.8 | 310 | 0 |
| Rotational speed [rpm] | 1168 | 2886 | 1539 | 0 |
| Torque [Nm] | 3.8 | 76.6 | 39.99 | 0 |
| Tool wear [min] | 0 | 253 | 108 | 0 |
| Machine failure | 0 | 1 | 0.0339 | 0 |
| TWF | 0 | 1 | 0.0046 | 0 |
| HDF | 0 | 1 | 0.0115 | 0 |
| PWF | 0 | 1 | 0.0095 | 0 |
| OSF | 0 | 1 | 0.0098 | 0 |
| RNF | 0 | 1 | 0.0019 | 0 |

## Plot
`ai4i_failure_physics.png` — failures concentrate exactly on the documented
physical limits (power window edges, HDF corner below dT=8.6K & 1380 rpm).
