"""Operator diagnosis toolbox (Builder C) — C3-tools-v1.

Plain-python tool implementations behind the tool-calling operator agent
(`operator_agent.py`). Every tool returns a JSON-serializable dict that
ALWAYS carries: {"tool": name, "params": {...}, "at": iso_timestamp,
"ok": bool, ...data}. The agent builds its auditable "Data Provenance"
trail from these envelopes — tools never raise, they return ok:false.

Spec (team): get_sensor_history, get_machine_spec, run_diagnostic,
verify_custody_chain, analyze_drift + two explicit stubs
(get_pinn_reconstruction, get_camera_frame — "physical subsystem
unintegrated"). Store/knowledge are duck-typed (PIPELINE_CONTRACT.md).
"""
from __future__ import annotations

import inspect
from datetime import datetime, timezone
from typing import Any, Optional

from .hmac_auth import sign_payload

# Builder A's causal matrix — guarded import so a mid-edit triage.py can
# never take the chat agent down (matrix context simply degrades to {}).
try:  # pragma: no cover - trivial
    from .triage import CAUSAL_MATRIX
except Exception:  # noqa: BLE001
    CAUSAL_MATRIX = {}

DRIFT_PCT = 5.0        # |% shift recent vs baseline| >= this → "drifting"
TREND_PCT = 3.0        # per-signal first→last trend threshold in history
_WINDOW = 5            # baseline = first 5 readings, recent = last 5


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _envelope(tool: str, params: dict[str, Any]) -> dict[str, Any]:
    return {"tool": tool, "params": params, "at": _now_iso()}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ===================================================================== specs
# Static design boundaries for the three Roanne machines (narrative constants
# from PIPELINE_CONTRACT.md + pinn/data mapping; AI4I failure-mode logic for
# the CL-03/MX-02 limits, C-MAPSS→C3M mapping ranges for RC-07).
MACHINE_SPECS: dict[str, dict[str, Any]] = {
    "RC-07": {
        "machine_id": "RC-07",
        "department": "Curing",
        "type": "C3M electric curing press",
        "role": ("Plant bottleneck — cures UHP tyres at ~200/h (~5,000/day "
                 "site total). An unplanned stop scraps in-process green "
                 "tyres and starves final inspection."),
        "human_risk": ("Burns from hot mould surfaces and high-pressure "
                       "curing media — hot high-pressure presses are one of "
                       "the two high-severity human-risk stages on the line."),
        "signals": {
            "mould_temp_C": {"unit": "degC", "operating": [150.0, 185.0], "trip": 195.0},
            "coil_power_kW": {"unit": "kW", "operating": [60.0, 92.0], "trip": 95.0},
            "vibration_rms_mm_s": {"unit": "mm/s", "operating": [1.5, 4.5],
                                   "alert": 4.5, "trip": 7.0},
            "pressure_bar": {"unit": "bar", "operating": [14.0, 22.0], "trip": 24.0},
        },
        "causal_tags": ["BEARING", "HDF"],
    },
    "CL-03": {
        "machine_id": "CL-03",
        "department": "Calendering",
        "type": "4-roll calender line",
        "role": ("Rubberises steel/textile plies that feed tyre building; "
                 "sits directly upstream of the curing bottleneck."),
        "human_risk": ("Nip-point entrapment between counter-rotating rolls "
                       "— the other high-severity human-risk stage."),
        "signals": {
            "roll_temp_C": {"unit": "degC", "operating": [30.0, 45.0], "trip": 50.0},
            "thermal_margin_K": {"unit": "K", "operating": [8.0, 15.0], "alert_low": 9.0,
                                 "note": ("AI4I HDF logic: margin under ~8 K with nip "
                                          "drive under ~1380 rpm = heat-dissipation "
                                          "failure band")},
            "drive_speed_rpm": {"unit": "rpm", "operating": [1200.0, 2900.0]},
            "drive_torque_Nm": {"unit": "Nm", "operating": [10.0, 70.0], "trip": 76.0},
            "tool_wear_min": {"unit": "min", "operating": [0.0, 220.0], "trip": 250.0},
        },
        "causal_tags": ["HDF", "OSF"],
    },
    "MX-02": {
        "machine_id": "MX-02",
        "department": "Mixing",
        "type": "Banbury internal mixer",
        "role": ("Head of the chain — compounds rubber with carbon black; a "
                 "mixer stall starves calendering and building downstream."),
        "human_risk": ("Carbon-black dust exposure and rotor entanglement "
                       "during charging and cleaning."),
        "signals": {
            "chamber_temp_C": {"unit": "degC", "operating": [30.0, 45.0], "trip": 55.0},
            "thermal_margin_K": {"unit": "K", "operating": [8.0, 15.0]},
            "drive_speed_rpm": {"unit": "rpm", "operating": [1200.0, 2900.0]},
            "drive_torque_Nm": {"unit": "Nm", "operating": [10.0, 70.0], "trip": 76.0,
                                "note": ("AI4I PWF logic: torque x speed outside the "
                                         "3.5-9 kW power window = power-failure band")},
            "tool_wear_min": {"unit": "min", "operating": [0.0, 220.0], "trip": 250.0,
                              "note": ("AI4I OSF logic: wear x torque beyond ~11,000 "
                                       "minNm = overstrain band")},
        },
        "causal_tags": ["PWF", "OSF", "TWF"],
    },
}

_MODE_TO_TAG = (
    ("bear", "BEARING"), ("vib", "BEARING"),
    ("therm", "HDF"), ("heat", "HDF"), ("hdf", "HDF"),
    ("power", "PWF"), ("pwf", "PWF"),
    ("strain", "OSF"), ("osf", "OSF"),
    ("wear", "TWF"), ("twf", "TWF"),
)


def _mode_tag(mode_name: str) -> str:
    low = (mode_name or "").lower()
    for key, tag in _MODE_TO_TAG:
        if key in low:
            return tag
    return mode_name.upper() if mode_name.upper() in CAUSAL_MATRIX else "GENERIC"


def _check_signal(name: str, value: float, sig_spec: dict) -> Optional[dict]:
    """Worst violation of one signal vs its spec, or None if inside limits."""
    trip = sig_spec.get("trip")
    if trip is not None and value >= trip:
        return {"signal": name, "value": value, "limit": trip,
                "kind": "at/beyond TRIP limit"}
    op = sig_spec.get("operating")
    if op and len(op) == 2:
        lo, hi = op
        if value > hi:
            return {"signal": name, "value": value, "limit": hi,
                    "kind": "above operating ceiling"}
        if value < lo:
            return {"signal": name, "value": value, "limit": lo,
                    "kind": "under operating floor"}
    alert = sig_spec.get("alert")
    if alert is not None and value > alert:
        return {"signal": name, "value": value, "limit": alert,
                "kind": "above alert threshold"}
    alert_low = sig_spec.get("alert_low")
    if alert_low is not None and value < alert_low:
        return {"signal": name, "value": value, "limit": alert_low,
                "kind": "below alert floor"}
    return None


def _outside_operating(value: float, sig_spec: dict) -> bool:
    op = sig_spec.get("operating")
    if not op or len(op) != 2:
        return False
    return value < op[0] or value > op[1]


# =================================================================== toolbox
class OperatorToolbox:
    """Real local implementations of the operator agent's tools.

    `store` follows the StateStore duck-type (get_sensor_history,
    get_advisories, get_overrides, department_snapshot,
    get_readings_with_signatures); `knowledge` is optional and only used by
    the agent layer, kept here so both paths share one construction site.
    """

    TOOL_NAMES = ("get_sensor_history", "get_machine_spec", "run_diagnostic",
                  "verify_custody_chain", "analyze_drift",
                  "get_pinn_reconstruction", "get_camera_frame")

    def __init__(self, store: Any, knowledge: Any = None) -> None:
        self.store = store
        self.knowledge = knowledge

    # -------------------------------------------------------------- dispatch
    def call(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute tool `name` with loosely-typed params. Never raises."""
        if name not in self.TOOL_NAMES:
            return {**_envelope(name, params), "ok": False,
                    "error": f"unknown tool '{name}'"}
        fn = getattr(self, name)
        try:
            sig = inspect.signature(fn)
            kwargs = {k: v for k, v in (params or {}).items() if k in sig.parameters}
            return fn(**kwargs)
        except Exception as e:  # noqa: BLE001 — tool errors are data, not crashes
            return {**_envelope(name, params), "ok": False,
                    "error": f"tool execution error: {e!r}"}

    # --------------------------------------------------------------- history
    def get_sensor_history(self, machine_id: str = "RC-07", limit: int = 20) -> dict:
        try:
            limit = max(2, min(int(limit), 100))
        except (TypeError, ValueError):
            limit = 20
        machine_id = str(machine_id).upper().strip()
        res = _envelope("get_sensor_history", {"machine_id": machine_id, "limit": limit})
        rows = self.store.get_sensor_history(machine_id, limit=limit)
        if not rows:
            return {**res, "ok": False,
                    "error": f"no telemetry stored for {machine_id} — "
                             "run the loop or check the machine id"}
        summary: dict[str, dict] = {}
        for name in rows[-1].get("signals", {}):
            series = [r["signals"][name] for r in rows if name in r.get("signals", {})]
            if not series:
                continue
            first, last = series[0], series[-1]
            pct = ((last - first) / abs(first) * 100.0) if first else 0.0
            trend = ("rising" if pct >= TREND_PCT
                     else "falling" if pct <= -TREND_PCT else "stable")
            summary[name] = {
                "first": round(first, 3), "last": round(last, 3),
                "min": round(min(series), 3), "max": round(max(series), 3),
                "change_pct": round(pct, 1), "trend": trend,
            }
        pinn_first = rows[0].get("pinn", {}) or {}
        pinn_last = rows[-1].get("pinn", {}) or {}
        tail = [
            {
                "epoch": r.get("epoch"),
                "signals": r.get("signals", {}),
                "health_index": (r.get("pinn", {}) or {}).get("health_index"),
                "rul_cycles": (r.get("pinn", {}) or {}).get("rul_cycles"),
                "residual": (r.get("pinn", {}) or {}).get("residual"),
                "note": str(r.get("note", ""))[:110],
            }
            for r in rows[-5:]
        ]
        return {
            **res, "ok": True,
            "machine_id": machine_id,
            "department": rows[-1].get("department", "?"),
            "count": len(rows),
            "epoch_range": [rows[0].get("epoch"), rows[-1].get("epoch")],
            "signal_summary": summary,
            "pinn_summary": {
                "health_index": {"first": pinn_first.get("health_index"),
                                 "last": pinn_last.get("health_index")},
                "rul_cycles": {"first": pinn_first.get("rul_cycles"),
                               "last": pinn_last.get("rul_cycles")},
                "residual": {"first": pinn_first.get("residual"),
                             "last": pinn_last.get("residual")},
            },
            "recent_tail": tail,
        }

    # ------------------------------------------------------------------ spec
    def get_machine_spec(self, machine_id: str = "RC-07") -> dict:
        machine_id = str(machine_id).upper().strip()
        res = _envelope("get_machine_spec", {"machine_id": machine_id})
        spec = MACHINE_SPECS.get(machine_id)
        if spec is None:
            return {**res, "ok": False,
                    "error": f"unknown machine '{machine_id}' — known machines: "
                             f"{', '.join(sorted(MACHINE_SPECS))}"}
        causal = {tag: CAUSAL_MATRIX[tag] for tag in spec.get("causal_tags", [])
                  if tag in CAUSAL_MATRIX}
        return {**res, "ok": True, **spec,
                "causal_matrix": causal or
                {"note": "causal matrix unavailable (triage module not loaded)"}}

    # ------------------------------------------------------------ diagnostic
    def run_diagnostic(self, machine_id: str = "RC-07") -> dict:
        machine_id = str(machine_id).upper().strip()
        res = _envelope("run_diagnostic", {"machine_id": machine_id})
        rows = self.store.get_sensor_history(machine_id, limit=3)
        if not rows:
            return {**res, "ok": False,
                    "error": f"no telemetry stored for {machine_id} — cannot diagnose"}
        last = rows[-1]
        spec = MACHINE_SPECS.get(machine_id, {})
        sig_specs = spec.get("signals", {})

        # Background epoch-triage state (kept SEPARATE from realtime — spec rule 2)
        risk, tick_epoch = "UNKNOWN", None
        try:
            for _dept, v in (self.store.department_snapshot() or {}).items():
                if v.get("machine_id") == machine_id:
                    risk, tick_epoch = v.get("risk", "UNKNOWN"), v.get("epoch")
        except Exception:  # noqa: BLE001
            pass
        machine_advs = [a for a in self.store.get_advisories(limit=50)
                        if a.machine_id == machine_id]
        adv_list = [
            {"id": a.id, "severity": a.severity.value, "status": a.status.value,
             "title": a.title,
             "jury_passed": (a.jury.passed if a.jury else None)}
            for a in machine_advs[:3]
        ]
        adv_ids = {a.id for a in machine_advs}
        override_list = [
            {"advisory_id": o.advisory_id, "decision": o.decision,
             "reason": (o.reason or "")[:120]}
            for o in self.store.get_overrides(limit=20) if o.advisory_id in adv_ids
        ]

        # Realtime check: latest signals vs design limits
        out_of_limits = []
        for name, value in (last.get("signals") or {}).items():
            if name in sig_specs:
                hit = _check_signal(name, float(value), sig_specs[name])
                if hit:
                    out_of_limits.append(hit)
        pinn = last.get("pinn", {}) or {}
        modes = pinn.get("failure_mode_probs") or {}
        dominant = max(modes, key=modes.get) if modes else None
        causal = CAUSAL_MATRIX.get(_mode_tag(dominant)) if dominant else None

        health = pinn.get("health_index")
        what = (f"triage risk {risk}; "
                + (", ".join(f"{h['signal']}={h['value']} {h['kind']} ({h['limit']})"
                             for h in out_of_limits)
                   if out_of_limits else "all monitored signals inside spec limits"))
        why = (f"PINN health {health * 100:.0f}%" if isinstance(health, (int, float))
               else "PINN health n/a")
        why += (f", residual {pinn.get('residual')}, RUL {pinn.get('rul_cycles')} cycles"
                + (f"; dominant mode {dominant} p={modes[dominant]:.2f}" if dominant else ""))
        how = (causal.get("remedy") if causal else
               "keep the machine under dense polling and compare against its spec "
               "(get_machine_spec) before any intervention")
        verdict = {
            "WHERE": f"{machine_id} ({last.get('department', '?')}), latest epoch {last.get('epoch')}",
            "WHAT": what,
            "WHY": why + (f"; probable cause: {causal['probable_cause']}" if causal else ""),
            "HOW": f"proposed (operator decides): {how}",
        }
        return {
            **res, "ok": True,
            "machine_id": machine_id,
            "risk": risk,
            "realtime": {
                "epoch": last.get("epoch"),
                "signals": last.get("signals", {}),
                "pinn": pinn,
                "out_of_limits": out_of_limits,
            },
            "background": {
                "last_tick_risk": risk,
                "last_tick_epoch": tick_epoch,
                "advisories": adv_list,
                "operator_overrides": override_list,
            },
            "verdict": verdict,
        }

    # --------------------------------------------------------------- custody
    def verify_custody_chain(self, machine_id: str = "RC-07",
                             epoch: Optional[int] = None) -> dict:
        machine_id = str(machine_id).upper().strip()
        params: dict[str, Any] = {"machine_id": machine_id}
        if epoch is not None:
            params["epoch"] = epoch
        res = _envelope("verify_custody_chain", params)
        getter = getattr(self.store, "get_readings_with_signatures", None)
        if getter is None:
            return {**res, "ok": False,
                    "error": "store lacks custody records "
                             "(get_readings_with_signatures unavailable)"}
        rows = getter(machine_id, limit=100)
        if epoch is not None:
            try:
                rows = [r for r in rows if int(r.get("epoch", -1)) == int(epoch)]
            except (TypeError, ValueError):
                pass
        if not rows:
            return {**res, "ok": False,
                    "error": f"no stored readings for {machine_id}"
                             + (f" at epoch {epoch}" if epoch is not None else "")}
        signed = [r for r in rows if r.get("signature")]
        unsigned = len(rows) - len(signed)
        if not signed:
            return {**res, "ok": False,
                    "error": f"no custody records — {len(rows)} reading(s) for "
                             f"{machine_id} were stored without signatures"}
        intact, broken = 0, []
        for r in signed:
            payload = r.get("payload")
            ep = r.get("epoch")
            if payload is None:
                broken.append({"epoch": ep, "link": f"payload_at_rest epoch {ep}",
                               "detail": "raw payload missing — cannot re-verify HMAC"})
                continue
            if sign_payload(payload) == r["signature"]:
                intact += 1
            else:
                broken.append({"epoch": ep, "link": f"ingestion_hmac epoch {ep}",
                               "detail": "stored payload no longer matches its "
                                         "ingestion HMAC — data altered after capture"})
        ok = not broken
        if broken:
            verdict = (f"custody chain BROKEN at {broken[0]['link']}"
                       + (f" (+{len(broken) - 1} more link(s))" if len(broken) > 1 else "")
                       + f" — {intact}/{len(signed)} links intact")
        else:
            verdict = (f"custody chain INTACT — {intact}/{len(signed)} stored readings "
                       "still match their ingestion HMAC")
            if unsigned:
                verdict += f" ({unsigned} pre-custody reading(s) without signature skipped)"
        return {**res, "ok": ok, "machine_id": machine_id,
                "checked": len(signed), "intact": intact, "unsigned": unsigned,
                "broken_links": broken, "verdict": verdict}

    # ----------------------------------------------------------------- drift
    def analyze_drift(self, machine_id: str = "RC-07",
                      metric: Optional[str] = None) -> dict:
        machine_id = str(machine_id).upper().strip()
        params: dict[str, Any] = {"machine_id": machine_id}
        if metric:
            params["metric"] = metric
        res = _envelope("analyze_drift", params)
        rows = self.store.get_sensor_history(machine_id, limit=40)
        if len(rows) < 2 * _WINDOW:
            return {**res, "ok": False,
                    "error": f"insufficient history for {machine_id} "
                             f"({len(rows)} readings, need >= {2 * _WINDOW})"}
        base_rows, recent_rows = rows[:_WINDOW], rows[-_WINDOW:]
        names = list(rows[-1].get("signals", {}).keys())
        if metric:
            if metric not in names:
                return {**res, "ok": False,
                        "error": f"unknown metric '{metric}' for {machine_id}; "
                                 f"available: {', '.join(names)}"}
            names = [metric]
        sig_specs = MACHINE_SPECS.get(machine_id, {}).get("signals", {})
        per_signal: dict[str, dict] = {}
        drifting, chronic = [], []
        for name in names:
            base = _mean([r["signals"][name] for r in base_rows
                          if name in r.get("signals", {})])
            recent = _mean([r["signals"][name] for r in recent_rows
                            if name in r.get("signals", {})])
            shift = ((recent - base) / abs(base) * 100.0) if base else 0.0
            if abs(shift) >= DRIFT_PCT:
                status = "drifting"
                drifting.append(name)
            elif name in sig_specs and _outside_operating(base, sig_specs[name]) \
                    and _outside_operating(recent, sig_specs[name]):
                status = "chronic"  # bad in BOTH windows but flat — not drift
                chronic.append(name)
            else:
                status = "stable"
            per_signal[name] = {"baseline_mean": round(base, 3),
                                "recent_mean": round(recent, 3),
                                "shift_pct": round(shift, 1), "status": status}
        h_base = _mean([(r.get("pinn", {}) or {}).get("health_index") or 0.0
                        for r in base_rows])
        h_recent = _mean([(r.get("pinn", {}) or {}).get("health_index") or 0.0
                          for r in recent_rows])
        if drifting:
            verdict = (f"{machine_id} is DRIFTING on {', '.join(drifting)} "
                       f"(recent {_WINDOW}-reading window vs baseline)")
        elif chronic:
            verdict = (f"{machine_id} shows CHRONIC out-of-spec on "
                       f"{', '.join(chronic)} — flat but outside limits (not drift)")
        else:
            verdict = f"{machine_id} is stable vs its baseline window"
        return {
            **res, "ok": True, "machine_id": machine_id,
            "baseline_epochs": [base_rows[0].get("epoch"), base_rows[-1].get("epoch")],
            "recent_epochs": [recent_rows[0].get("epoch"), recent_rows[-1].get("epoch")],
            "signals": per_signal,
            "health_index": {"baseline_mean": round(h_base, 3),
                             "recent_mean": round(h_recent, 3)},
            "drifting": drifting, "chronic": chronic,
            "verdict": verdict,
            "note": ("drift = recent window moved vs baseline; a signal bad in BOTH "
                     "windows but flat is chronic, not drifting"),
        }

    # ----------------------------------------------------------------- stubs
    def get_pinn_reconstruction(self, machine_id: str = "RC-07") -> dict:
        res = _envelope("get_pinn_reconstruction",
                        {"machine_id": str(machine_id).upper().strip()})
        return {**res, "ok": False, "stub": True,
                "error": "STUB — physical subsystem not integrated in this layer yet",
                "note": ("full-field PINN reconstruction is reserved for the physical "
                         "layer; only scalar PinnState telemetry reaches this "
                         "information layer today")}

    def get_camera_frame(self, camera_id: str = "CAM-RC-07") -> dict:
        res = _envelope("get_camera_frame", {"camera_id": str(camera_id).strip()})
        return {**res, "ok": False, "stub": True,
                "error": "STUB — physical subsystem not integrated in this layer yet",
                "note": ("shop-floor camera feeds are not wired into this layer; "
                         "no frame can be served or analyzed yet")}


__all__ = ["OperatorToolbox", "MACHINE_SPECS", "DRIFT_PCT"]
