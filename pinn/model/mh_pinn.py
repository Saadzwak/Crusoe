"""Multi-Head PINN: per-head input adapters + shared recurrent body + heads.

Heterogeneous-input design (required — the four data sources differ wildly in
sampling rate and channel count, from 1 tabular row per sample to 12 kHz
waveforms):

    raw input --[per-head ADAPTER]--> common token sequence [B, K, d_model]
              --[shared PI-LSTM BODY]--> hidden states [B, K, hidden]
              --[per-head HEAD]--> physical outputs

Adapters:
- vibration:  2048-sample window framed into 32 frames of 64 samples, each
              frame linearly projected (a learned short-time transform; keeps
              the LSTM at 32 steps instead of 2048).
- thermal:    one tabular row -> a single token (seq len 1).
- rul:        30-cycle window of 14 sensors -> 30 tokens (linear per cycle).
- pressure:   256-step (P,T) trace framed into 32 frames of 8 steps.

The BODY is a shared LSTM: "PI-LSTM-style" means physics enters through the
loss residuals (pinn.losses), not through exotic cells — the recurrent core
is what the heads share so cross-phenomenon structure (slow degradation vs
fast transients) can transfer.
"""

from __future__ import annotations

import torch
from torch import nn

HEADS = ("vibration", "thermal", "rul", "pressure",
         "fatigue", "thermal_recon", "fe_recon")

# Heads that consume the same input modality share the adapter — the fatigue
# and fan-end-reconstruction heads read accelerometer windows exactly like the
# vibration head; masked-input temperature reconstruction reads AI4I rows like
# the thermal head. Sharing keeps the "one shared physical representation"
# property of the MH-PINN instead of forking per task.
ADAPTER_ALIASES = {"fatigue": "vibration", "fe_recon": "vibration",
                   "thermal_recon": "thermal"}


class FrameAdapter(nn.Module):
    """Frame a long waveform/trace into K tokens and project to d_model."""

    def __init__(self, frame: int, in_channels: int, d_model: int):
        super().__init__()
        self.frame = frame
        self.in_channels = in_channels
        self.proj = nn.Linear(frame * in_channels, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L] (single channel) or [B, L, C]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        b, length, c = x.shape
        k = length // self.frame
        x = x[:, : k * self.frame].reshape(b, k, self.frame * c)
        return self.proj(x)


class VibrationHead(nn.Module):
    """4-class fault logits (Normal/IR/OR/B) + degradation health score in [0,1]."""

    def __init__(self, hidden: int):
        super().__init__()
        self.cls = nn.Linear(hidden, 4)
        self.health = nn.Linear(hidden, 1)

    def forward(self, h_seq: torch.Tensor) -> dict:
        pooled = h_seq.mean(dim=1)
        return {"fault_logits": self.cls(pooled),
                "health": torch.sigmoid(self.health(pooled)).squeeze(-1)}


class ThermalHead(nn.Module):
    """Failure-probability logits: [machine_failure, HDF, PWF]."""

    def __init__(self, hidden: int):
        super().__init__()
        self.out = nn.Linear(hidden, 3)

    def forward(self, h_seq: torch.Tensor) -> dict:
        return {"failure_logits": self.out(h_seq[:, -1])}


class RulHead(nn.Module):
    """Per-timestep RUL (cycles), softplus-clamped to stay physically >= 0."""

    def __init__(self, hidden: int):
        super().__init__()
        self.out = nn.Linear(hidden, 1)
        # Start predictions mid-range (softplus(60) ~= 60 cycles) instead of ~0.7,
        # so few-epoch v0 runs land inside the physical 0..125 range quickly.
        nn.init.constant_(self.out.bias, 60.0)

    def forward(self, h_seq: torch.Tensor) -> dict:
        return {"rul_seq": nn.functional.softplus(self.out(h_seq)).squeeze(-1)}


class RulHeadMonotone(nn.Module):
    """RUL non-increasing BY CONSTRUCTION (Step-6 diagnostic variant).

    Parameterization: RUL_t = RUL_end + sum_{s>t} delta_s with delta_s =
    softplus(.) >= 0. Irreversible damage becomes an architectural guarantee
    instead of a soft penalty; the physics loss then only has to push
    delta -> 1 cycle/cycle below the cap. Built to answer: is the -0.61 slope
    a lambda-weighting problem or a representational one?
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.end = nn.Linear(hidden, 1)
        self.delta = nn.Linear(hidden, 1)
        nn.init.constant_(self.end.bias, 60.0)   # same mid-range rationale
        # softplus(0.541) ~= 1.0: start at the physical -1 cycle/cycle slope.
        nn.init.constant_(self.delta.bias, 0.541)

    def forward(self, h_seq: torch.Tensor) -> dict:
        rul_end = nn.functional.softplus(self.end(h_seq[:, -1])).squeeze(-1)  # [B]
        deltas = nn.functional.softplus(self.delta(h_seq)).squeeze(-1)        # [B,T]
        tail = deltas.flip(1).cumsum(dim=1).flip(1) - deltas                  # sum_{s>t}
        return {"rul_seq": rul_end.unsqueeze(1) + tail}


class PressureHead(nn.Module):
    """v2 (Step-7) — physics-INTEGRATED nominal reconstruction.

    v1's finite-difference ODE residual was measured "vacuously satisfied"
    (shared-body gradient ratio 4e-4, runs/v1/diagnosis.json): an average
    canonical cycle obeys the ODE *shape* without tracking any individual
    cycle. v2 gives the ODE more structure instead of a bolt-on: the head
    ESTIMATES each cycle's physical parameters (P_set, tau_ramp, tau_release)
    from the observed trace, and the nominal reconstruction is produced by
    INTEGRATING the curing-cycle ODE with those parameters. The ODE is
    piecewise linear, so each token step has the closed-form solution
        ramp:    P+ = P_set + (P - P_set) * exp(-dt/tau_ramp)
        hold:    P+ = P
        release: P+ = P * exp(-dt/tau_release)
    — exact, unconditionally stable (a plain Euler step would blow up at
    dt_token ~ 56 s > tau ~ 25 s), and differentiable end-to-end.

    Anomaly and alpha_end read from residual statistics of
    (observed - integrated nominal), kept from the v1 degeneracy fix; the
    residual is now per-cycle-meaningful because the reconstruction is.
    alpha_end estimates the FINAL DEGREE OF CURE — the quantity no production
    sensor measures (Kamal-Sourour kinetics, SYNTHETIC supervision).
    """

    TAU_RAMP_FLOOR = 3.0     # [s] keep taus off zero for exp stability
    TAU_RELEASE_FLOOR = 2.0

    def __init__(self, hidden: int):
        super().__init__()
        self.params = nn.Linear(hidden, 3)   # -> P_set, tau_ramp, tau_release
        n_resid_stats = 4
        self.anom = nn.Linear(hidden + n_resid_stats, 1)
        self.alpha = nn.Linear(hidden + n_resid_stats, 1)
        with torch.no_grad():
            # Start at the documented envelope midpoints: ~17.5 bar,
            # tau_ramp ~ 27 s, tau_release ~ 12 s (softplus(x) ~= x here).
            self.params.bias.copy_(torch.tensor([17.5, 24.0, 10.0]))
        # Nominal cycles cure to ~0.99: start alpha near there (sigmoid(3)=0.95).
        nn.init.constant_(self.alpha.bias, 3.0)

    def forward(self, h_seq: torch.Tensor, p_obs_tok: torch.Tensor,
                dt_tok: torch.Tensor, ramp_tok: torch.Tensor,
                hold_tok: torch.Tensor, rel_tok: torch.Tensor) -> dict:
        pooled_h = h_seq.mean(dim=1)
        pr = nn.functional.softplus(self.params(pooled_h))
        p_set = pr[:, 0]
        tau_r = pr[:, 1] + self.TAU_RAMP_FLOOR
        tau_rel = pr[:, 2] + self.TAU_RELEASE_FLOOR

        e_ramp = torch.exp(-dt_tok / tau_r)          # per-cycle constants
        e_rel = torch.exp(-dt_tok / tau_rel)
        p = torch.zeros_like(p_set)
        recon_steps = []
        for k in range(p_obs_tok.shape[1]):
            p_ramp = p_set + (p - p_set) * e_ramp
            p_next = torch.where(ramp_tok[:, k], p_ramp,
                                 torch.where(rel_tok[:, k], p * e_rel, p))
            p = p_next
            recon_steps.append(p)
        recon = torch.stack(recon_steps, dim=1)      # [B, K]

        resid = p_obs_tok - recon
        kk = resid.shape[1]
        late = resid[:, kk // 3:]                    # leak lives mid-hold onward
        stats = torch.stack([resid.mean(dim=1),
                             resid.min(dim=1).values,
                             resid.abs().max(dim=1).values,
                             late.mean(dim=1)], dim=1)
        pooled = torch.cat([pooled_h, stats], dim=1)
        return {"pressure_recon": recon,
                "cycle_params": torch.stack([p_set, tau_r, tau_rel], dim=1),
                "anomaly_logit": self.anom(pooled).squeeze(-1),
                "alpha_end": torch.sigmoid(self.alpha(pooled)).squeeze(-1)}


class FatigueHead(nn.Module):
    """Miner damage fraction D in [0, 1] from a vibration window.

    D is supervised by lifetime position on the IMS run-to-failure data
    (exact under constant amplitude: D = t / T_fail) and interpreted at
    inference through Basquin/Hertz (pinn.physics.fatigue) to produce
    hours-to-replacement — the "replace before it fails" quantity.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.out = nn.Linear(hidden, 1)

    def forward(self, h_seq: torch.Tensor) -> dict:
        return {"damage": torch.sigmoid(self.out(h_seq.mean(dim=1))).squeeze(-1)}


class ThermalReconHead(nn.Module):
    """VIRTUAL SENSOR: reconstruct the (withheld) process temperature [K].

    Input rows have the process-temperature column MASKED; the head must
    recover it from air temperature + speed + torque + wear, regularized by
    the AI4I HDF rule evaluated on the reconstruction.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.out = nn.Linear(hidden, 1)
        nn.init.constant_(self.out.bias, 310.0)  # AI4I process temp ~305-314 K

    def forward(self, h_seq: torch.Tensor) -> dict:
        return {"process_temp_K": self.out(h_seq[:, -1]).squeeze(-1)}


class FeReconHead(nn.Module):
    """VIRTUAL SENSOR: reconstruct the (withheld) fan-end accelerometer's
    physics features — envelope band energies at {BPFO,BPFI,BSF} + RMS —
    from the drive-end window alone (shared shaft => shared fault kinematics)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.bands = nn.Linear(hidden, 3)
        self.rms = nn.Linear(hidden, 1)

    def forward(self, h_seq: torch.Tensor) -> dict:
        pooled = h_seq.mean(dim=1)
        return {"fe_bands": torch.sigmoid(self.bands(pooled)),
                "fe_rms": nn.functional.softplus(self.rms(pooled)).squeeze(-1)}


class MHPinn(nn.Module):
    # hidden 96 -> 128 (Step-7, A2): the measured multi-task "interference"
    # is capacity dilution, NOT gradient conflict (all pairwise task-gradient
    # cosines ~ 0 on the shared body, runs/v2/stress_test.json) — so the
    # remedy is representation budget, not gradient surgery.
    def __init__(self, d_model: int = 64, hidden: int = 128, num_layers: int = 1,
                 vib_window: int = 2048, pressure_seq: int = 256,
                 rul_monotone: bool = False):
        super().__init__()
        self.adapters = nn.ModuleDict({
            "vibration": FrameAdapter(frame=64, in_channels=1, d_model=d_model),
            "thermal": nn.Linear(8, d_model),
            "rul": nn.Linear(14, d_model),
            "pressure": FrameAdapter(frame=8, in_channels=2, d_model=d_model),
        })
        self.body = nn.LSTM(d_model, hidden, num_layers=num_layers, batch_first=True)
        self.heads = nn.ModuleDict({
            "vibration": VibrationHead(hidden),
            "thermal": ThermalHead(hidden),
            "rul": RulHeadMonotone(hidden) if rul_monotone else RulHead(hidden),
            "pressure": PressureHead(hidden),
            "fatigue": FatigueHead(hidden),
            "thermal_recon": ThermalReconHead(hidden),
            "fe_recon": FeReconHead(hidden),
        })
        self.pressure_frame = 8

    def forward(self, head: str, x: torch.Tensor, extras: dict | None = None) -> dict:
        assert head in HEADS, f"unknown head {head!r}"
        tokens = self.adapters[ADAPTER_ALIASES.get(head, head)](x)
        if tokens.dim() == 2:                            # single tabular token
            tokens = tokens.unsqueeze(1)
        h_seq, _ = self.body(tokens)
        if head == "pressure":
            # Tokenize observed pressure, phase masks and time step to the
            # recon grid; the head integrates the cycle ODE on that grid.
            assert extras is not None, "pressure head needs extras: dt + phase masks"
            f = self.pressure_frame
            b, k = x.shape[0], x.shape[1] // f
            p_obs_tok = x[:, : k * f, 0].reshape(b, k, f).mean(dim=2)

            def tok(m: torch.Tensor) -> torch.Tensor:
                return m[:, : k * f].reshape(b, k, f).float().mean(dim=2) > 0.5

            return self.heads[head](h_seq, p_obs_tok, extras["dt"] * f,
                                    tok(extras["mask_ramp"]), tok(extras["mask_hold"]),
                                    tok(extras["mask_release"]))
        return self.heads[head](h_seq)
