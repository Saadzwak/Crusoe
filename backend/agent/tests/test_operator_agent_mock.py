"""Tool-calling operator agent test (Builder C) — plain script, mock mode.

Run from repo root:  python3 backend/agent/tests/test_operator_agent_mock.py
Exit code 0 = green. Asserts:
  1. toolbox.get_sensor_history returns real numbers (summary + raw tail)
  2. run_diagnostic names out-of-limit signals late in the RC-07 arc
  3. verify_custody_chain: ok on clean records; detects a deliberately
     tampered stored payload and names the broken link ("ingestion_hmac epoch N")
  4. analyze_drift flags RC-07 drifting (and separates chronic from drift)
  5. stubs answer "physical subsystem not integrated"
  6. agent (mock path) answer carries real digits + "Data Provenance"
     + >= 2 tools in citations/provenance
  7. stream_events yields tool_call before token before turn
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
from pathlib import Path

os.environ["MOCK_LLM"] = "1"  # must precede any backend.agent import

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from backend.agent.operator_agent import ToolCallingOperator  # noqa: E402
from backend.agent.operator_tools import OperatorToolbox  # noqa: E402
from backend.agent.knowledge import Knowledge  # noqa: E402
from backend.agent.schemas import PinnReading, RiskLabel, TickResult, TriageResult  # noqa: E402
from backend.agent.state_store import StateStore  # noqa: E402
from backend.agent.telemetry_source import TelemetrySource, note_risk  # noqa: E402

EPOCHS = 12  # RC-07 arc: CLEAR early → HIGH drift mid → CRITICAL by 10+
PASS = 0


def ok(label: str) -> None:
    global PASS
    PASS += 1
    print(f"  ok {PASS:02d} — {label}")


def main() -> None:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    store = StateStore(db_path=tmp.name)
    ts = TelemetrySource()
    knowledge = Knowledge()

    # ------------------------------------------------ populate WITH signatures
    for epoch in range(EPOCHS):
        for signed in ts.next_batch(epoch):
            store.save_reading(signed.payload, signature=signed.signature)
            reading = PinnReading.model_validate(signed.payload)
            risk = note_risk(reading.note)
            store.save_tick(TickResult(
                machine_id=reading.machine_id, department=reading.department,
                epoch=epoch, hmac_verified=True,
                triage=TriageResult(tier_reached=1 if risk == RiskLabel.CLEAR else 2,
                                    risk=risk, reason="test echo"),
            ))
    ok(f"populated {EPOCHS} epochs x 3 machines with at-rest signatures")

    toolbox = OperatorToolbox(store, knowledge)

    # ------------------------------------------------------- 1 sensor history
    hist = toolbox.get_sensor_history("RC-07", limit=20)
    assert hist["ok"] and hist["tool"] == "get_sensor_history" and hist["at"]
    assert hist["count"] == EPOCHS
    vib = hist["signal_summary"]["vibration_rms_mm_s"]
    assert isinstance(vib["min"], float) and isinstance(vib["max"], float)
    assert vib["max"] > vib["min"] > 0, f"suspicious vibration range: {vib}"
    assert vib["trend"] == "rising", f"RC-07 vibration should trend rising: {vib}"
    assert len(hist["recent_tail"]) == 5
    assert hist["recent_tail"][-1]["signals"]["vibration_rms_mm_s"] > 4.5
    ok(f"get_sensor_history: {hist['count']} readings, vibration "
       f"{vib['min']}→{vib['max']} ({vib['trend']} {vib['change_pct']:+.1f}%)")

    # --------------------------------------------------------- 2 diagnostic
    diag = toolbox.run_diagnostic("RC-07")
    assert diag["ok"]
    hits = diag["realtime"]["out_of_limits"]
    assert hits, "late-arc RC-07 should have out-of-limit signals"
    named = {h["signal"] for h in hits}
    assert "vibration_rms_mm_s" in named, f"vibration not named: {named}"
    assert diag["risk"] == "CRITICAL", f"background risk should be CRITICAL: {diag['risk']}"
    v = diag["verdict"]
    assert all(k in v for k in ("WHERE", "WHAT", "WHY", "HOW"))
    assert "vibration_rms_mm_s" in v["WHAT"]
    assert re.search(r"\d", v["WHY"]), "verdict WHY carries no numbers"
    ok(f"run_diagnostic names excursions: {sorted(named)}; risk={diag['risk']}")

    unknown = toolbox.run_diagnostic("ZZ-99")
    assert not unknown["ok"] and "no telemetry" in unknown["error"]
    spec_unknown = toolbox.get_machine_spec("ZZ-99")
    assert not spec_unknown["ok"] and "unknown machine" in spec_unknown["error"]
    ok("unknown machine → explicit ok:false (diagnostic + spec)")

    spec = toolbox.get_machine_spec("RC-07")
    assert spec["ok"] and spec["signals"]["vibration_rms_mm_s"]["trip"] == 7.0
    assert spec["causal_matrix"], "causal matrix context missing"
    assert "BEARING" in spec["causal_matrix"]
    ok("get_machine_spec: limits + causal matrix context for RC-07")

    # ------------------------------------------------------------ 3 custody
    cust = toolbox.verify_custody_chain("RC-07")
    assert cust["ok"] and cust["checked"] == EPOCHS and cust["intact"] == EPOCHS
    assert not cust["broken_links"] and "INTACT" in cust["verdict"]
    ok(f"custody chain clean: {cust['verdict']}")

    # tamper one stored payload (flip a value at rest), expect a named break
    TAMPER_EPOCH = 3
    with store._lock:  # test-only surgical tamper of the at-rest payload
        row = store._conn.execute(
            "SELECT id, payload_json FROM readings WHERE machine_id='RC-07' AND epoch=?",
            (TAMPER_EPOCH,)).fetchone()
        payload = json.loads(row["payload_json"])
        payload["signals"]["vibration_rms_mm_s"] += 1.0  # the flipped byte
        store._conn.execute("UPDATE readings SET payload_json=? WHERE id=?",
                            (json.dumps(payload), row["id"]))
        store._conn.commit()
    cust2 = toolbox.verify_custody_chain("RC-07")
    assert not cust2["ok"], "tampered chain still verifies!"
    links = [b["link"] for b in cust2["broken_links"]]
    assert f"ingestion_hmac epoch {TAMPER_EPOCH}" in links, f"broken link not named: {links}"
    assert cust2["intact"] == EPOCHS - 1
    ok(f"tampered payload detected + named: {cust2['broken_links'][0]['link']}")

    cust3 = toolbox.verify_custody_chain("RC-07", epoch=TAMPER_EPOCH)
    assert not cust3["ok"] and cust3["checked"] == 1
    cust4 = toolbox.verify_custody_chain("RC-07", epoch=0)
    assert cust4["ok"] and cust4["checked"] == 1
    ok("epoch-scoped custody check isolates the broken link")

    # -------------------------------------------------------------- 4 drift
    drift = toolbox.analyze_drift("RC-07")
    assert drift["ok"]
    assert "vibration_rms_mm_s" in drift["drifting"], \
        f"RC-07 vibration should drift: {drift['signals']['vibration_rms_mm_s']}"
    assert "DRIFTING" in drift["verdict"]
    assert "chronic" in drift["note"], "chronic-vs-drift distinction missing"
    dsig = drift["signals"]["vibration_rms_mm_s"]
    assert dsig["recent_mean"] > dsig["baseline_mean"]
    bad_metric = toolbox.analyze_drift("RC-07", metric="warp_core_temp")
    assert not bad_metric["ok"] and "unknown metric" in bad_metric["error"]
    ok(f"analyze_drift flags RC-07: {drift['verdict']} "
       f"({dsig['baseline_mean']}→{dsig['recent_mean']}, {dsig['shift_pct']:+.1f}%)")

    # -------------------------------------------------------------- 5 stubs
    for stub, kwargs in (("get_pinn_reconstruction", {"machine_id": "RC-07"}),
                         ("get_camera_frame", {"camera_id": "CAM-RC-07"})):
        r = toolbox.call(stub, kwargs)
        assert not r["ok"] and r.get("stub") is True
        assert "not integrated" in r["error"], f"{stub} stub wording: {r['error']}"
    ok("both stubs answer 'physical subsystem not integrated' (ok:false)")

    # ------------------------------------------------------- 6 agent (mock)
    agent = ToolCallingOperator(store, knowledge)
    turn, provenance = asyncio.run(
        agent.answer("Why is RC-07 flagged and can I trust the data? "
                     "What would a stop cost?"))
    assert turn.role == "agent" and turn.content
    assert re.search(r"\d", turn.content), "answer carries no real numbers"
    assert "Data Provenance:" in turn.content
    prov_tools = [p["tool"] for p in provenance]
    assert len(set(prov_tools)) >= 2, f"expected >=2 tools, got {prov_tools}"
    assert set(prov_tools) & {"run_diagnostic", "get_sensor_history", "analyze_drift"}
    assert "verify_custody_chain" in prov_tools, "trust question should trigger custody"
    for p in provenance:
        assert {"tool", "params", "at", "ok", "summary"} <= set(p)
    assert len(turn.citations) >= 2
    assert any("site_dossier" in c for c in turn.citations), \
        f"cost question should cite the dossier: {turn.citations}"
    # provenance lines are appended programmatically — one per executed tool
    prov_section = turn.content.split("Data Provenance:")[1]
    assert prov_section.count("- ") == len(provenance)
    ok(f"mock agent answer: {len(provenance)} tools, citations={turn.citations[:4]}…")

    # -------------------------------------------------------- 7 stream order
    async def collect():
        evs = []
        async for ev in agent.stream_events("Is MX-02 drifting?"):
            evs.append(ev)
        return evs

    evs = asyncio.run(collect())
    kinds = [e["type"] for e in evs]
    assert "tool_call" in kinds and "token" in kinds and "turn" in kinds
    first_call, first_token = kinds.index("tool_call"), kinds.index("token")
    assert first_call < first_token < kinds.index("turn"), f"bad order: {kinds[:8]}"
    calls = [e for e in evs if e["type"] == "tool_call"]
    results = [e for e in evs if e["type"] == "tool_result"]
    assert len(calls) == len(results) >= 3
    assert all(e["params"]["machine_id"] == "MX-02" for e in calls
               if "machine_id" in e.get("params", {}))
    final = [e for e in evs if e["type"] == "turn"][0]["turn"]
    assert final["provenance"] and "".join(
        e["text"] for e in evs if e["type"] == "token") == final["content"]
    assert kinds[-1] == "turn"
    ok(f"stream_events order: tool_call({len(calls)}) → tokens → turn; "
       "streamed text == final content")

    store.close()
    os.unlink(tmp.name)
    print(f"\nALL GREEN — {PASS} checks passed (tool-calling operator, mock mode)")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nFAIL: {e}", file=sys.stderr)
        sys.exit(1)
