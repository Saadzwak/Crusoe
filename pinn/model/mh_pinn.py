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

HEADS = ("vibration", "thermal", "rul", "pressure")


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


class PressureHead(nn.Module):
    """Per-token nominal-pressure reconstruction + cycle-level anomaly logit.

    The reconstruction learns what a HEALTHY cycle looks like (trained on
    nominal cycles + constrained by the curing-cycle ODE); the anomaly logit
    flags departures such as the injected mid-hold seal leak.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.recon = nn.Linear(hidden, 1)
        self.anom = nn.Linear(hidden, 1)
        # Start reconstructions near the documented steam envelope (~17 bar)
        # rather than ~0.7 bar — same few-epoch convergence rationale as RulHead.
        nn.init.constant_(self.recon.bias, 17.0)

    def forward(self, h_seq: torch.Tensor) -> dict:
        return {"pressure_recon": nn.functional.softplus(self.recon(h_seq)).squeeze(-1),
                "anomaly_logit": self.anom(h_seq.mean(dim=1)).squeeze(-1)}


class MHPinn(nn.Module):
    def __init__(self, d_model: int = 64, hidden: int = 96, num_layers: int = 1,
                 vib_window: int = 2048, pressure_seq: int = 256):
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
            "rul": RulHead(hidden),
            "pressure": PressureHead(hidden),
        })
        self.pressure_frame = 8

    def forward(self, head: str, x: torch.Tensor) -> dict:
        assert head in HEADS, f"unknown head {head!r}"
        tokens = self.adapters[head](x)
        if tokens.dim() == 2:                            # single tabular token
            tokens = tokens.unsqueeze(1)
        h_seq, _ = self.body(tokens)
        return self.heads[head](h_seq)
