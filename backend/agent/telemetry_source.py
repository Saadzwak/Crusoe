"""PINN stand-in: replays public benchmark data as HMAC-signed PinnReadings.

Three Roanne machines (PIPELINE_CONTRACT.md §narrative):
  RC-07  "Curing"       C3M electric press — the bottleneck. Follows NASA
                        C-MAPSS FD001 **unit 1** (192 cycles, run-to-failure).
  CL-03  "Calendering"  Calender line — AI4I 2020 rows, one scheduled
                        heat-dissipation (HDF) spike around epoch 8.
  MX-02  "Mixing"       Banbury internal mixer — AI4I 2020 healthy rows only.

D4-telemetry-v1 — RC-07 signals rescaled to the VERIFIED curing limits
(backend.agent.limits, provenance in limits.PROVENANCE). The C-MAPSS unit-1
sensors still provide deterministic realism, but as small wobble terms on top
of a monotonic degradation arc `frac = (cycle-40)/151` (0 at epoch 0, 1 at
end of life), so the arc is GUARANTEED to sweep the verified bands in order:

  mould_temp_C      = 193 + 3·w(s2)  + 18.0·frac^2.6   → ~193-200 nominal,
                      crosses the 210 °C normal ceiling at end of life (~214).
  coil_power_kW     = C-MAPSS s4 linear map [60, 92] kW  (unchanged stand-in).
  vibration_rms_mm_s= 0.85 + 0.2·w(s11) + 7.3·frac^2.2 → ISO zones swept in
                      order as health decays: A(≤1.12) epochs 0-2 →
                      B(≤2.8) 3-6 → C(≤7.1) 7-11 → D(>7.1) 12+.
  pressure_bar      = 17.6 − 0.4·w(s15) − 2.9·frac^1.8 → ~17.5 nominal in the
                      16-19 bar steam band, dips under 16 late (~14.5 at end).
  cycle_min         = 12.4 + 0.3·w(s2) + 18.8·frac^3.2 → 12.5 nominal, stays
                      ≤15 early, extends late, overruns 30 (→~31) in the last
                      epochs (cycle ≥ ~188).
  (w(sN) = the C-MAPSS sensor mapped linearly to [0,1] over its unit-1 range.)

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


# ==== INTEGRATION (demo 2026-07-05) — real-PINN hero + honest replays ========
# CP-07 (the UI hero card) <- RC-07 fed by the TRAINED MH-PINN
# (pinn/models/mh_pinn_v2.pt) instead of the linear stand-in; three extra hall
# presses replay DISTINCT real FD001 units (readings only — no PINN claim, no
# advisory pipeline; the loop skips the LLM path via REPLAY_ONLY_IDS).
REPLAY_PRESS_UNITS = {"CP-01": 24, "CP-03": 76, "CP-10": 2}
REPLAY_ONLY_IDS = frozenset(REPLAY_PRESS_UNITS)


class TelemetrySource:
    """Deterministic replay — next_batch(epoch) is a pure function of epoch."""

    UNIT = 1  # C-MAPSS FD001 unit followed by RC-07

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        d = Path(data_dir) if data_dir else _DATA_DIR
        self._fd001 = self._load_fd001(d / "train_FD001.txt", self.UNIT)
        self._max_cycle = len(self._fd001)  # unit 1 → 192
        self._ai4i_healthy, self._ai4i_hdf = self._load_ai4i(d / "ai4i2020.csv")
        # ---- integration: trained-PINN runtime for RC-07 (graceful fallback)
        self._pinn_runtime = None
        try:
            import sys
            _root = Path(__file__).resolve().parents[2]
            if str(_root) not in sys.path:
                sys.path.insert(0, str(_root))
            from pinn.inference import PinnRuntime
            self._pinn_runtime = PinnRuntime(repo_root=_root, unit=self.UNIT)
            print("[telemetry] RC-07 physical state: TRAINED MH-PINN "
                  "(pinn/models/mh_pinn_v2.pt, RUL head)")
        except Exception as e:  # noqa: BLE001 — no torch/checkpoint: stand-in
            print(f"[telemetry] PinnRuntime unavailable ({e!r}) — RC-07 keeps "
                  "the linear stand-in mapping")
        # ---- integration: distinct-real-unit replays for the hall presses
        self._replays: dict[str, list[list[float]]] = {}
        for _mid, _unit in REPLAY_PRESS_UNITS.items():
            try:
                self._replays[_mid] = self._load_fd001(d / "train_FD001.txt", _unit)
            except Exception as e:  # noqa: BLE001
                print(f"[telemetry] replay {_mid} (FD001 unit {_unit}) off: {e!r}")
        # ---- integration: live scenario state (a real factory dashboard must
        # keep moving). "heartbeat" = RC-07 healthy, gently ticking; "fault" =
        # the degradation arc runs from the epoch the operator injected it.
        self.arc_mode = "heartbeat"     # "heartbeat" | "fault"
        self.fault_kind = "bearing"     # "bearing" | "thermal"
        self._fault_epoch0 = 0
        self._replay_epoch0 = 0         # hall replays restart from their healthy start

    def set_scenario(self, mode: str, kind: str = "bearing",
                     epoch: int = 0) -> None:
        """Switch RC-07 between healthy heartbeat and a fault arc (demo control)."""
        self.arc_mode = "fault" if mode == "fault" else "heartbeat"
        if self.arc_mode == "fault":
            self.fault_kind = kind if kind in ("thermal", "bearing", "hdf") \
                else "bearing"
            self._fault_epoch0 = int(epoch)
        else:
            # Reset to normal: rewind the hall replays to the healthy start of
            # their real units so every press reads green again (still a REAL
            # trajectory — just from its early, healthy portion).
            self._replay_epoch0 = int(epoch)

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
        # D4-telemetry-v1: arc rescaled to the verified limits (see module
        # docstring for the exact mapping).
        # Integration: two modes. HEARTBEAT holds a healthy early cycle that
        # oscillates gently (the dashboard stays alive at rest); FAULT runs the
        # degradation arc from the epoch the operator injected the fault.
        if self.arc_mode == "fault":
            # Demo pacing: the operator just pressed the fault button — the
            # targeted signal must move within a tick or two and cross its
            # limit in ~3 ticks (6 s at the 2 s loop), not half a minute.
            # 30 cycles/epoch still walks the SAME real degradation
            # trajectory, just faster (~5 epochs to end of life).
            eff = max(0, epoch - self._fault_epoch0)
            cycle = min(40 + 30 * eff, self._max_cycle - 1)
            heartbeat = False
        else:
            phase = epoch % 16
            cycle = 42 + (phase if phase < 8 else 16 - phase)   # 42..50..42, healthy
            heartbeat = True
        _, s2, s4, s11, s15 = self._fd001[cycle - 1]
        health = round(1.0 - cycle / self._max_cycle, 3)
        rul = float(self._max_cycle - cycle)
        residual = _safe_round(0.015 + 0.8 * (1.0 - health) ** 4, 4)
        # ---- integration: the TRAINED MH-PINN drives RC-07's physical state
        # (health/RUL/residual/modes); the linear values above stay as the
        # documented fallback when torch or the checkpoint are unavailable.
        model_modes: Optional[dict] = None
        if getattr(self, "_pinn_runtime", None) is not None:
            try:
                _p = self._pinn_runtime.infer_at_cycle(cycle)
                health = _p["health_index"]
                rul = float(_p["rul_cycles"])
                residual = _p["residual"]
                model_modes = _p["failure_mode_probs"]
            except Exception as e:  # noqa: BLE001 — never kill the feed
                print(f"[telemetry] PINN inference failed at cycle {cycle}: {e!r}")

        # degradation fraction 0..1 over the demo arc; 0 while healthy.
        frac = 0.0 if heartbeat else \
            max(0.0, min(1.0, (cycle - 40.0) / float(self._max_cycle - 1 - 40)))
        # thermal fault drives mould temp harder; bearing fault drives vibration.
        # Factors sized so the TARGETED signal crosses its verified limit at
        # eff≈2-3 (4-6 s) while the other signals follow later — the drawer
        # tile goes red on the same machine the button named, fast.
        f_temp = frac * (1.6 if self.fault_kind == "thermal" else 0.7)
        f_vib = frac * (1.45 if self.fault_kind == "bearing" else 0.6)
        # C-MAPSS sensor wobble terms, each mapped to [0, 1] over unit-1 range
        w2 = _lin(s2, 641.0, 644.5, 0.0, 1.0)
        w11 = _lin(s11, 47.0, 48.5, 0.0, 1.0)
        w15 = _lin(s15, 8.38, 8.55, 0.0, 1.0)
        # HEARTBEAT: gentle in-envelope sensor noise so the dashboard visibly
        # lives at rest (display-level only; PINN health/RUL above are real).
        import math
        hb = (1.0 if heartbeat else 0.0)
        n_temp = hb * 0.9 * math.sin(epoch * 0.9)
        n_vib = hb * (0.35 + 0.28 * math.sin(epoch * 1.7))   # ~0.1..0.9 mm/s live wobble
        n_pow = hb * 1.4 * math.sin(epoch * 0.6 + 1.0)

        signals = {
            "mould_temp_C": _safe_round(193.0 + 3.0 * w2 + 22.0 * f_temp ** 2.4 + n_temp),
            "coil_power_kW": _safe_round(_lin(s4, 1398.0, 1428.0, 60.0, 92.0) + n_pow),
            "vibration_rms_mm_s": _safe_round(0.85 + 0.2 * w11 + 8.2 * f_vib ** 2.1 + n_vib),
            "pressure_bar": _safe_round(17.6 - 0.4 * w15 - 2.9 * frac ** 1.8),
            "cycle_min": _safe_round(12.4 + 0.3 * w2 + 18.8 * frac ** 3.2, 1),
        }

        eff_note = 0 if heartbeat else (epoch - self._fault_epoch0)
        if heartbeat or eff_note <= 0:
            note = ("Nominal curing cycle on press RC-07: mould temperature, "
                    "coil power and press vibration all inside the envelope.")
            modes: dict[str, float] = {}
        elif eff_note <= 2:
            driver = ("mould temperature climbing on the shoulder"
                      if self.fault_kind == "thermal"
                      else "vibration rising cycle over cycle")
            note = (f"Press RC-07 trend watch: {driver}; the physics twin sees "
                    f"remaining life falling.")
            modes = {("thermal_runaway" if self.fault_kind == "thermal"
                      else "bearing_wearout"): 0.35}
        else:
            driver = ("mould temperature" if self.fault_kind == "thermal"
                      else "vibration")
            note = (f"CRITICAL drift on press RC-07: {driver} climbing toward the "
                    f"failure envelope, remaining life down to {int(rul)} cycles "
                    f"— imminent {self.fault_kind} failure suspected.")
            modes = ({"thermal_runaway": 0.65, "bearing_wearout": 0.2}
                     if self.fault_kind == "thermal"
                     else {"bearing_wearout": 0.65, "thermal_runaway": 0.25})

        if model_modes is not None:      # integration: the model's view wins
            modes = model_modes or modes
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
        # Integration: the HDF spike is its own scenario (fault_kind "hdf"),
        # NOT a side effect of a CP-07 fault — a thermal/bearing injection on
        # the press must not make the calender pop a red card mid-demo (the
        # operator pressed a CP-07 button; every alert should stay CP-07).
        eff = (epoch - self._fault_epoch0) \
            if (self.arc_mode == "fault" and self.fault_kind == "hdf") else -1
        if eff == 4:  # scheduled HDF spike (UDI 3237: margin 8.6 K, 1342 rpm)
            row = self._ai4i_hdf
            note = ("Heat-dissipation anomaly on calender CL-03 nip drive: "
                    "thermal margin collapsing and torque load rising above envelope.")
            pinn = PinnState(health_index=0.55, rul_cycles=None, residual=0.34,
                             failure_mode_probs={"HDF": 0.72})
        else:
            row = self._ai4i_healthy[(epoch * 131 + 17) % n]
            note = ("Calender CL-03 steady: nip temperatures, web tension and "
                    "roll torque in range.")
            if eff == 5:
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

    # ---------------------------------------------------- integration: replays
    def _replay_reading(self, mid: str, epoch: int) -> PinnReading:
        """Readings-only replay of one DISTINCT real FD001 unit (demo hall).

        Same signal-mapping family as RC-07, slower advance, no drama arc,
        neutral note wording (mock-triage discipline). The pinn block is the
        trajectory position — labeled a replay, never PINN inference.
        """
        rows = self._replays[mid]
        n = len(rows)
        # slow advance (1 cycle/epoch) from the unit's healthy start; reset
        # rewinds _replay_epoch0 so the hall goes green again on demand.
        cycle = min(5 + max(0, epoch - self._replay_epoch0), n - 1)
        _, s2, s4, s11, s15 = rows[cycle - 1]
        w2 = _lin(s2, 641.0, 644.5, 0.0, 1.0)
        signals = {
            "mould_temp_C": _safe_round(193.0 + 3.0 * w2),
            "coil_power_kW": _safe_round(_lin(s4, 1398.0, 1428.0, 60.0, 92.0)),
            "vibration_rms_mm_s": _safe_round(
                0.9 + 0.25 * _lin(s11, 47.0, 48.5, 0.0, 1.0), 2),
            "pressure_bar": _safe_round(17.5 - 0.4 * _lin(s15, 8.38, 8.55, 0.0, 1.0)),
            "cycle_min": _safe_round(12.3 + 0.4 * w2, 1),
        }
        health = round(1.0 - cycle / n, 3)
        return PinnReading(
            machine_id=mid, department="Curing", epoch=epoch,
            signals=signals,
            pinn=PinnState(health_index=max(0.0, min(health, 0.89)),
                           rul_cycles=float(n - cycle), residual=0.0,
                           failure_mode_probs={}),
            note=(f"REPLAY of real C-MAPSS FD001 unit "
                  f"{REPLAY_PRESS_UNITS[mid]} — readings replay only, no PINN "
                  f"inference, no advisory pipeline (demo hall press)."),
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
        # ---- integration: honest hall replays (distinct REAL FD001 units,
        # readings only — the loop keeps them out of the LLM pipeline).
        for _mid in self._replays:
            payload = self._replay_reading(_mid, epoch).model_dump()
            out.append(SignedReading(payload=payload,
                                     signature=sign_payload(payload)))
        return out
