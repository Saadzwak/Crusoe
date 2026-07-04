"""PINN stand-in: replays public benchmark data as HMAC-signed PinnReadings.

Three Roanne machines (PIPELINE_CONTRACT.md §narrative):
  RC-07  "Curing"       C3M electric press — the bottleneck. Follows NASA
                        C-MAPSS FD001 **unit 1** (192 cycles, run-to-failure).
  CL-03  "Calendering"  Calender line — AI4I 2020 rows, one scheduled
                        heat-dissipation (HDF) spike around epoch 8.
  MX-02  "Mixing"       Banbury internal mixer — AI4I 2020 healthy rows only.

C-MAPSS → C3M signal mapping (per files (1)/DATA_PACK_README.md table;
sensor columns are unit,cycle,op1..3,s1..s21 — s2=col idx 6, s4=8, s11=15, s15=19):

  s2   LPC outlet temp   [641.0, 644.5] °R  → mould_temp_C      [150, 190] °C
  s4   LPT outlet temp   [1398, 1428]   °R  → coil_power_kW     [ 60,  92] kW
  s11  HPC static press. [47.0, 48.5]  psia → vibration_rms_mm_s[1.5, 7.5] mm/s
  s15  bypass ratio      [8.38, 8.55]    —  → pressure_bar      [ 14,  22] bar

All four drift upward as unit 1 approaches failure, which reads as a curing
press running hot, drawing more coil power, vibrating harder — exactly the
dossier's demo scenario [site_dossier p.5]. Linear map, clamped to range.

Demo arc (epoch = loop tick, ~12-15 epochs total):
  cycle(epoch) = min(40 + 12*epoch, 191)  →  health = 1 - cycle/192,
  rul = 192 - cycle. Epochs 0-5 CLEAR, 6-9 drifting (notes trip Tier-2 HIGH
  in mock mode), >=10 CRITICAL (rul 32 → 1). AI4I rows are drawn at fixed
  prime strides from the non-failure pool (deterministic, seed-free — pure
  function of epoch); CL-03 substitutes the first HDF failure row (UDI 3237,
  thermal margin 8.6 K, 1342 rpm) at epoch 8 only.

Mock-LLM note discipline: MockClient tier2 keys on substrings
  CRITICAL: "critical", "failure", "imminent" (+ "0.9", "rul_cycles': 1")
  HIGH:     "high", "drift", "rising", "anomal", "+3"
Healthy notes contain NONE of these; anomalous notes deliberately do.
`_safe_round` also nudges any numeric whose repr would contain "0.9" so a
healthy reading dumped raw into a prompt can't trip the mock's CRITICAL rule.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .hmac_auth import sign_payload
from .schemas import PinnReading, PinnState, RiskLabel, SignedReading

_DATA_DIR = Path(__file__).resolve().parents[2] / "pinn" / "data"

# ---------------------------------------------------------------- mock words
CRITICAL_WORDS = ("critical", "failure", "imminent")
HIGH_WORDS = ("high", "drift", "rising", "anomal", "+3")


def note_risk(note: str) -> RiskLabel:
    """Mirror of MockClient's tier2 keyword rules — used by the degraded
    'triage-echo' loop in main.py and by the service test."""
    low = note.lower()
    if any(w in low for w in CRITICAL_WORDS):
        return RiskLabel.CRITICAL
    if any(w in low for w in HIGH_WORDS):
        return RiskLabel.HIGH
    return RiskLabel.CLEAR


def _safe_round(v: float, nd: int = 2) -> float:
    """Round, then nudge away from any repr containing '0.9' (mock footgun)."""
    r = round(v, nd)
    for _ in range(8):
        if "0.9" not in repr(r):
            return r
        r = round(r + 0.02, nd)
    return r


def _lin(x: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    x = min(max(x, lo), hi)
    return out_lo + (x - lo) * (out_hi - out_lo) / (hi - lo)


class TelemetrySource:
    """Deterministic replay — next_batch(epoch) is a pure function of epoch."""

    UNIT = 1  # C-MAPSS FD001 unit followed by RC-07

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        d = Path(data_dir) if data_dir else _DATA_DIR
        self._fd001 = self._load_fd001(d / "train_FD001.txt", self.UNIT)
        self._max_cycle = len(self._fd001)  # unit 1 → 192
        self._ai4i_healthy, self._ai4i_hdf = self._load_ai4i(d / "ai4i2020.csv")

    # ------------------------------------------------------------- loaders
    @staticmethod
    def _load_fd001(path: Path, unit: int) -> list[list[float]]:
        """Only the chosen unit's rows: [cycle, s2, s4, s11, s15]."""
        rows: list[list[float]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if not parts or int(parts[0]) != unit:
                    if rows:  # units are contiguous — stop after ours
                        break
                    continue
                rows.append([float(parts[1]), float(parts[6]), float(parts[8]),
                             float(parts[15]), float(parts[19])])
        if not rows:
            raise FileNotFoundError(f"no rows for unit {unit} in {path}")
        return rows

    @staticmethod
    def _load_ai4i(path: Path) -> tuple[list[dict], dict]:
        """(healthy rows, first HDF failure row) as small dicts."""
        healthy: list[dict] = []
        hdf: Optional[dict] = None
        with open(path, "r", encoding="utf-8-sig") as f:
            header = f.readline()  # skip
            for line in f:
                c = line.rstrip("\n").split(",")
                if len(c) < 14:
                    continue
                row = {
                    "air_K": float(c[3]), "proc_K": float(c[4]),
                    "rpm": float(c[5]), "torque": float(c[6]), "wear": float(c[7]),
                }
                if c[8] == "0":
                    healthy.append(row)
                elif c[10] == "1" and hdf is None:  # first HDF row (UDI 3237)
                    hdf = row
        if not healthy or hdf is None:
            raise ValueError(f"unexpected ai4i2020.csv contents at {path}")
        return healthy, hdf

    # ------------------------------------------------------------- curing
    def _curing_reading(self, epoch: int) -> PinnReading:
        cycle = min(40 + 12 * max(epoch, 0), self._max_cycle - 1)
        _, s2, s4, s11, s15 = self._fd001[cycle - 1]
        health = round(1.0 - cycle / self._max_cycle, 3)
        rul = float(self._max_cycle - cycle)
        residual = _safe_round(0.015 + 0.8 * (1.0 - health) ** 4, 4)

        signals = {
            "mould_temp_C": _safe_round(_lin(s2, 641.0, 644.5, 150.0, 190.0)),
            "coil_power_kW": _safe_round(_lin(s4, 1398.0, 1428.0, 60.0, 92.0)),
            "vibration_rms_mm_s": _safe_round(_lin(s11, 47.0, 48.5, 1.5, 7.5)),
            "pressure_bar": _safe_round(_lin(s15, 8.38, 8.55, 14.0, 22.0)),
        }

        if epoch <= 5:
            note = ("Nominal curing cycle on press RC-07: mould temperature, "
                    "coil power and press vibration all inside the envelope.")
            modes: dict[str, float] = {}
        elif epoch <= 9:
            note = ("Press RC-07 trend watch: vibration rising cycle over cycle "
                    "and a thermal hotspot is suspected on the mould shoulder.")
            modes = {"bearing_wearout": 0.35}
        else:
            note = (f"CRITICAL drift on press RC-07: vibration and mould temperature "
                    f"climbing toward the failure envelope, RUL down to {int(rul)} "
                    f"cycles — imminent bearing wear-out suspected.")
            modes = {"bearing_wearout": 0.65, "thermal_runaway": 0.25}

        return PinnReading(
            machine_id="RC-07", department="Curing", epoch=epoch,
            signals=signals,
            pinn=PinnState(health_index=max(0.0, min(health, 0.89)),
                           rul_cycles=rul, residual=residual,
                           failure_mode_probs=modes),
            note=note,
        )

    # ------------------------------------------------------------- ai4i
    def _ai4i_signals(self, row: dict, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_temp_C": _safe_round(row["proc_K"] - 273.15),
            "thermal_margin_K": _safe_round(row["proc_K"] - row["air_K"]),
            "drive_speed_rpm": _safe_round(row["rpm"], 0),
            "drive_torque_Nm": _safe_round(row["torque"], 1),
            "tool_wear_min": _safe_round(row["wear"], 0),
        }

    def _calendering_reading(self, epoch: int) -> PinnReading:
        n = len(self._ai4i_healthy)
        if epoch == 8:  # scheduled HDF spike (UDI 3237: margin 8.6 K, 1342 rpm)
            row = self._ai4i_hdf
            note = ("Heat-dissipation anomaly on calender CL-03 nip drive: "
                    "thermal margin collapsing and torque load rising above envelope.")
            pinn = PinnState(health_index=0.55, rul_cycles=None, residual=0.34,
                             failure_mode_probs={"HDF": 0.72})
        else:
            row = self._ai4i_healthy[(epoch * 131 + 17) % n]
            note = ("Calender CL-03 steady: nip temperatures, web tension and "
                    "roll torque in range.")
            if epoch == 9:
                note = ("Calender CL-03 back in range after the thermal-margin "
                        "spike; nip drive load settled.")
            pinn = PinnState(health_index=_safe_round(0.84 + (epoch % 3) * 0.01),
                             rul_cycles=None, residual=0.02,
                             failure_mode_probs={})
        return PinnReading(
            machine_id="CL-03", department="Calendering", epoch=epoch,
            signals=self._ai4i_signals(row, "roll"), pinn=pinn, note=note,
        )

    def _mixing_reading(self, epoch: int) -> PinnReading:
        n = len(self._ai4i_healthy)
        row = self._ai4i_healthy[(epoch * 197 + 55) % n]
        note = ("Banbury mixer MX-02 nominal: rotor torque and chamber "
                "temperature steady, dust extraction in range.")
        pinn = PinnState(health_index=_safe_round(0.86 + (epoch % 2) * 0.01),
                         rul_cycles=None, residual=0.015,
                         failure_mode_probs={})
        return PinnReading(
            machine_id="MX-02", department="Mixing", epoch=epoch,
            signals=self._ai4i_signals(row, "chamber"), pinn=pinn, note=note,
        )

    # ------------------------------------------------------------- public
    def next_batch(self, epoch: int) -> list[SignedReading]:
        """One signed reading per machine for this epoch."""
        out: list[SignedReading] = []
        for reading in (self._curing_reading(epoch),
                        self._calendering_reading(epoch),
                        self._mixing_reading(epoch)):
            payload = reading.model_dump()
            out.append(SignedReading(payload=payload,
                                     signature=sign_payload(payload)))
        return out
