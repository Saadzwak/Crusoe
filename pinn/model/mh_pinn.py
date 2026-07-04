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
    """Nominal-pressure reconstruction + anomaly logit + final degree of cure.

    The reconstruction learns what a HEALTHY cycle looks like (trained on
    nominal cycles + constrained by the curing-cycle ODE). Anomaly and
    alpha_end are then read from the PHYSICS RESIDUAL — statistics of
    (observed - healthy reconstruction) — concatenated with the pooled hidden
    state. Rationale (learned in the first two v1 runs, where a plain
    pooled-hidden logit collapsed to a degenerate majority-class predictor in
    both directions): deviation from the ODE-consistent healthy trace IS the
    leak signature; giving the classifier the residual directly is the
    PINN-native detector, not a weighting trick. alpha_end estimates the FINAL
    DEGREE OF CURE — the quantity no production sensor measures
    (Kamal-Sourour kinetics, SYNTHETIC supervision).
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.recon = nn.Linear(hidden, 1)
        n_resid_stats = 4
        self.anom = nn.Linear(hidden + n_resid_stats, 1)
        self.alpha = nn.Linear(hidden + n_resid_stats, 1)
        # Start reconstructions near the documented steam envelope (~17 bar)
        # rather than ~0.7 bar — same few-epoch convergence rationale as RulHead.
        nn.init.constant_(self.recon.bias, 17.0)
        # Nominal cycles cure to ~0.99: start alpha near there (sigmoid(3)=0.95).
        nn.init.constant_(self.alpha.bias, 3.0)

    def forward(self, h_seq: torch.Tensor, p_obs_tok: torch.Tensor) -> dict:
        recon = nn.functional.softplus(self.recon(h_seq)).squeeze(-1)   # [B, K]
        resid = p_obs_tok - recon
        k = resid.shape[1]
        late = resid[:, k // 3:]          # leak lives mid-hold onward
        stats = torch.stack([resid.mean(dim=1),
                             resid.min(dim=1).values,
                             resid.abs().max(dim=1).values,
                             late.mean(dim=1)], dim=1)
        pooled = torch.cat([h_seq.mean(dim=1), stats], dim=1)
        return {"pressure_recon": recon,
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
    def __init__(self, d_model: int = 64, hidden: int = 96, num_layers: int = 1,
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

    def forward(self, head: str, x: torch.Tensor) -> dict:
        assert head in HEADS, f"unknown head {head!r}"
        tokens = self.adapters[ADAPTER_ALIASES.get(head, head)](x)
        if tokens.dim() == 2:                            # single tabular token
            tokens = tokens.unsqueeze(1)
        h_seq, _ = self.body(tokens)
        if head == "pressure":
            # Token-averaged observed pressure, matching the head's recon grid,
            # so the head can compute the physics residual (obs - healthy).
            f = self.pressure_frame
            k = x.shape[1] // f
            p_obs_tok = x[:, : k * f, 0].reshape(x.shape[0], k, f).mean(dim=2)
            return self.heads[head](h_seq, p_obs_tok)
        return self.heads[head](h_seq)
