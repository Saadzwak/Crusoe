"""SYNTHETIC curing-press pressure cycles (pressure head).

================================ SYNTHETIC ====================================
Every sample produced here is GENERATED, not measured. No public dataset of
tire-curing-press pressure traces exists; mentors accept synthetic generation
anchored to real published operating parameters as normal practice. Keep the
"synthetic" label attached to this head in code, docs, plots and demo output.
===============================================================================

Generation = Euler integration of the first-order cycle ODE in
pinn.physics.curing_press (ramp -> hold -> release) + Gaussian sensor noise,
with cycle-to-cycle variation of setpoint/durations inside the documented
envelope (16-19 bar steam, 180-210 degC, 10-15 min).

Anomaly scenario (what the head must catch): mid-hold seal/bladder leak —
an extra -k_leak * P term switched on at a random time during the hold phase,
producing slow decay where the trace should be flat.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pinn.physics.curing_press import CuringCycleParams, nominal_pressure_ode, phase_masks

PROVENANCE = "SYNTHETIC"  # imported by telemetry/report so the label can't drift


def generate_cycle(rng: np.random.Generator, anomalous: bool = False,
                   base: CuringCycleParams | None = None):
    """Simulate one cycle; returns dict with t, pressure, temperature, masks, label."""
    base = base or CuringCycleParams()
    # Cure-matched cycle duration (category (c) choice mirroring real practice:
    # cure time is set per recipe so a NOMINAL cycle reaches full cure — the
    # documented 10-15 min at ordinary temps, up to ~30 min at the cold end).
    # Without this, random (temp, duration) pairs drown the leak->under-cure
    # signal in setpoint variation, which no real plant would exhibit.
    from pinn.physics.cure_kinetics import CureParams, _t99_at, calibrated_prefactor

    temp_set = float(rng.uniform(185.0, 210.0))
    t99 = _t99_at(calibrated_prefactor(), round(temp_set, 1), CureParams())
    t_ramp, t_release = 90.0, 60.0
    hold_margin = float(rng.uniform(1.05, 1.20))         # engineering cure margin
    cycle_s = float(np.clip(t_ramp + hold_margin * t99 + t_release, 600.0, 1800.0))
    params = replace(
        base,
        cycle_s=cycle_s,                                  # 10-30 min, cure-matched
        p_set_bar=float(rng.uniform(16.0, 19.0)),        # steam pressure envelope
        temp_set_C=temp_set,
        tau_ramp_s=float(rng.uniform(20.0, 35.0)),
        tau_release_s=float(rng.uniform(8.0, 16.0)),
    )
    n = int(params.cycle_s / params.dt_s)
    t = np.arange(n, dtype=np.float32) * params.dt_s

    leak_rate, t_leak = 0.0, None
    if anomalous:
        # Leak strong enough to be visible but not a step change: 0.05-0.5%/s.
        # Upper end raised from 3e-3 after the first v1 runs: the sampled leaks
        # rarely pushed alpha_end below the 0.90 under-cure threshold, leaving
        # the detection tail with ~1-3 validation examples — unmeasurable.
        leak_rate = float(rng.uniform(5e-4, 5e-3))
        hold = phase_masks(t, params)["hold"]
        hold_times = t[hold]
        t_leak = float(rng.uniform(hold_times[0] + 30.0, hold_times[-1] - 60.0))

    p = np.zeros(n, dtype=np.float64)
    temp = np.full(n, 25.0)  # start at ambient
    for i in range(1, n):
        dpdt = nominal_pressure_ode(p[i - 1:i], t[i - 1:i], params,
                                    leak_rate=leak_rate, t_leak_s=t_leak)[0]
        p[i] = p[i - 1] + params.dt_s * dpdt
        # temperature: same first-order form toward setpoint, releases with mold opening
        target = params.temp_set_C if t[i] < params.cycle_s - params.t_release_s else 25.0
        tau_t = params.tau_temp_s if target > temp[i - 1] else 3.0 * params.tau_temp_s
        temp[i] = temp[i - 1] + params.dt_s * (target - temp[i - 1]) / tau_t

    # Leak -> saturation-temperature coupling (steam tables, ~3.2 K/bar; see
    # pinn.physics.cure_kinetics): a steam-pressure deficit during hold lowers
    # the effective cure temperature, which slows the Kamal-Sourour cure.
    from pinn.physics.cure_kinetics import integrate_cure, saturation_temp_drop

    hold_mask = phase_masks(t, params)["hold"]
    p_deficit = np.where(hold_mask, np.maximum(0.0, params.p_set_bar - p), 0.0)
    temp_effective = temp - saturation_temp_drop(p_deficit)
    alpha = integrate_cure(temp_effective, params.dt_s)

    p_noisy = p + rng.normal(0.0, params.noise_bar, size=n)
    temp_noisy = temp_effective + rng.normal(0.0, 0.4, size=n)
    return {
        "t": t,
        "pressure_bar": p_noisy.astype(np.float32),
        "pressure_clean": p.astype(np.float32),
        "temp_C": temp_noisy.astype(np.float32),
        "alpha": alpha.astype(np.float32),          # degree of cure along the cycle
        "alpha_end": float(alpha[-1]),              # the unmeasurable target quantity
        "masks": phase_masks(t, params),
        "label": int(anomalous),
        "params": params,
        "leak": {"rate": leak_rate, "t_start_s": t_leak},
        "provenance": PROVENANCE,
    }


def generate_dataset(n_cycles: int = 400, anomaly_fraction: float = 0.3,
                     seq_len: int = 256, seed: int = 42):
    """Fixed-length dataset for training: resample each cycle to `seq_len` steps.

    Resampling note (documented interpolation): cycles have variable duration
    (600-900 s at 1 Hz); we linearly interpolate each trace onto a common
    seq_len grid. Acceptable here because the pressure/temperature traces are
    smooth first-order responses — unlike vibration waveforms, nothing above
    the new Nyquist carries fault information (the leak signature is a slow
    trend). Timestamps are kept so the ODE residual uses the TRUE dt per cycle.
    """
    rng = np.random.default_rng(seed)
    X, T, labels, dts, alpha_ends = [], [], [], [], []
    mask_hold, mask_ramp, mask_rel, cycles = [], [], [], []
    for _ in range(n_cycles):
        anom = bool(rng.random() < anomaly_fraction)
        c = generate_cycle(rng, anomalous=anom)
        n = len(c["t"])
        grid = np.linspace(0, n - 1, seq_len)
        src = np.arange(n)
        X.append(np.stack([np.interp(grid, src, c["pressure_bar"]),
                           np.interp(grid, src, c["temp_C"])], axis=1))
        T.append(np.interp(grid, src, c["t"]))
        for name, buf in (("ramp", mask_ramp), ("hold", mask_hold), ("release", mask_rel)):
            buf.append(np.interp(grid, src, c["masks"][name].astype(float)) > 0.5)
        labels.append(c["label"])
        alpha_ends.append(c["alpha_end"])
        dts.append(float(c["t"][-1] / (seq_len - 1)))
        cycles.append(c)
    return {
        "X": np.stack(X).astype(np.float32),          # [N, seq_len, 2] P,T
        "t": np.stack(T).astype(np.float32),          # [N, seq_len] true seconds
        "y": np.asarray(labels, dtype=np.float32),    # anomaly label
        "alpha_end": np.asarray(alpha_ends, dtype=np.float32),  # final degree of cure
        "dt": np.asarray(dts, dtype=np.float32),      # per-cycle grid step [s]
        "mask_ramp": np.stack(mask_ramp), "mask_hold": np.stack(mask_hold),
        "mask_release": np.stack(mask_rel),
        "cycles": cycles,                              # full-resolution originals
        "provenance": PROVENANCE,
    }
