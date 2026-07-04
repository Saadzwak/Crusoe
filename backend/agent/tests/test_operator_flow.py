"""Operator-app backend test — plain runnable script, mock mode, temp DB.

Run from repo root:  MOCK_LLM=1 python3 backend/agent/tests/test_operator_flow.py
Exit code 0 = green. Covers docs/superpowers/specs/2026-07-04-operator-app-design.md:
  1. HIGH advisory → intervention 'notified' + notify routed to the ASSIGNED operator
  2. second advisory while active → linked silently (no duplicate toast)
  3. take_charge → acknowledged + linked advisory auto-accepted + broadcast notify
  4. VLM watcher: activity_detected then repair_verified, ordered timestamps
  5. repair_done guarded to the operator who took charge
  6. notification cooldown suppresses an immediate re-alert after closure
  7. routing: MX-02 alerts go to lina, never eric
  8. factory_state: all machines with assigned operator, advisory + intervention attached
  9. quick chat: fast-lane turn with programmatic citations + provenance
 10. interventions round-trip the store (upsert + active lookup)
 11. Teams card builder returns a well-formed MessageCard (no network)
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MOCK_LLM", "1")            # before any backend.agent import
os.environ["VLM_ACTIVITY_DELAY_S"] = "0.05"       # fast-forward the simulated VLM
os.environ["VLM_VERIFY_DELAY_S"] = "0.05"
os.environ["NOTIFY_COOLDOWN_S"] = "5.0"           # long enough to observe suppression

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from backend.agent.crusoe_client import MockClient  # noqa: E402
from backend.agent.operator_flow import (ASSIGNMENTS, OPERATORS,  # noqa: E402
                                         InterventionManager, assigned_operator,
                                         build_factory_state, quick_answer,
                                         teams_card)
from backend.agent.schemas import (Advisory, AdvisoryStatus,  # noqa: E402
                                   InterventionStatus, RiskLabel)
from backend.agent.state_store import StateStore  # noqa: E402

PASS = 0


def ok(label: str) -> None:
    global PASS
    PASS += 1
    print(f"  ok {PASS:02d} — {label}")


def mk_advisory(machine_id: str = "RC-07", severity: RiskLabel = RiskLabel.HIGH,
                department: str = "Curing", epoch: int = 6) -> Advisory:
    return Advisory(
        machine_id=machine_id, department=department, epoch=epoch,
        severity=severity, title=f"Test advisory on {machine_id}",
        message="Vibration trending toward the failure envelope.",
        justification="vibration_rms 6.9 mm/s vs 2.1 baseline [sensor_history]",
        recommended_action="Schedule bearing inspection in next window.",
    )


async def scenario(store: StateStore, events: list) -> None:
    def publish(kind: str, data: dict) -> None:
        events.append((kind, data))

    mgr = InterventionManager(store, publish)

    def notifies(kind: str = "alert"):
        return [d for k, d in events if k == "notify" and d.get("kind") == kind]

    # ------------------------------------------------ 1. notify + routing
    adv = mk_advisory("RC-07", RiskLabel.HIGH)
    store.save_advisory(adv)
    iv = mgr.on_advisory(adv)
    assert iv is not None and iv.status == InterventionStatus.NOTIFIED
    assert iv.machine_id == "RC-07" and iv.advisory_id == adv.id
    assert iv.notified_at is not None
    alerts = notifies("alert")
    assert len(alerts) == 1, f"expected 1 alert notify, got {len(alerts)}"
    assert alerts[0]["to_operator"]["id"] == ASSIGNMENTS["RC-07"]
    assert alerts[0]["machine_id"] == "RC-07" and alerts[0]["advisory_id"] == adv.id
    assert alerts[0]["severity"] == "HIGH" and alerts[0]["title"]
    assert any(k == "intervention" for k, _ in events)
    ok("HIGH advisory → intervention notified + alert routed to assigned operator")

    # WATCH/CLEAR advisories never notify
    watch_adv = mk_advisory("RC-07", RiskLabel.WATCH)
    assert mgr.on_advisory(watch_adv) is iv or \
        mgr.on_advisory(watch_adv).id == iv.id  # linked to same active intervention
    assert len(notifies("alert")) == 1
    ok("WATCH advisory does not toast (severity gate)")

    # ------------------------------------------------ 2. dedup while active
    adv2 = mk_advisory("RC-07", RiskLabel.CRITICAL, epoch=7)
    store.save_advisory(adv2)
    iv2 = mgr.on_advisory(adv2)
    assert iv2.id == iv.id, "second advisory must link to the SAME intervention"
    assert iv2.advisory_id == adv2.id, "intervention should track the latest advisory"
    assert len(notifies("alert")) == 1, "no duplicate toast while intervention active"
    ok("second advisory links silently to the active intervention")

    # ------------------------------------------------ 3. take charge
    with_events_before = len(events)
    iv3 = await mgr.take_charge("RC-07", "eric")
    assert iv3.id == iv.id and iv3.status == InterventionStatus.ACKNOWLEDGED
    assert iv3.operator_id == "eric" and iv3.operator_name == OPERATORS["eric"]["name"]
    assert iv3.acknowledged_at is not None and iv3.acknowledged_at >= iv3.notified_at
    broadcasts = notifies("broadcast")
    assert broadcasts and broadcasts[-1]["machine_id"] == "RC-07"
    assert broadcasts[-1]["to_operator"] is None
    got = store.get_advisory(adv2.id)
    assert got.status == AdvisoryStatus.ACCEPTED, "linked advisory must auto-accept"
    assert any(k == "override" for k, _ in events[with_events_before:]), \
        "console must hear the auto-accept as an override event"
    ok("take_charge → acknowledged + broadcast + linked advisory auto-accepted")

    # unknown operator refused
    try:
        await mgr.take_charge("RC-07", "mallory")
        raise AssertionError("unknown operator must be refused")
    except ValueError:
        ok("unknown operator refused")

    # ------------------------------------------------ 4. VLM activity
    await asyncio.sleep(0.25)
    iv4 = mgr.active("RC-07")
    assert iv4 is not None and iv4.vlm_activity_at is not None, "VLM never saw activity"
    assert iv4.vlm_activity_at >= iv4.acknowledged_at
    acts = [o for o in iv4.vlm_observations if o.kind == "activity_detected"]
    assert acts and 0.0 < acts[0].confidence <= 1.0 and acts[0].detail
    ok("VLM watcher logged activity_detected after take_charge")

    # ------------------------------------------------ 5. repair_done guard
    try:
        await mgr.repair_done("RC-07", "lina")
        raise AssertionError("repair_done by another operator must be refused")
    except ValueError:
        ok("repair_done rejected for an operator who did not take charge")

    iv5 = await mgr.repair_done("RC-07", "eric")
    assert iv5.status == InterventionStatus.CLAIMED_DONE
    assert iv5.claimed_done_at is not None and iv5.claimed_done_at >= iv5.vlm_activity_at

    await asyncio.sleep(0.25)
    iv6 = store.get_intervention(iv5.id)
    assert iv6.status == InterventionStatus.VERIFIED, f"status={iv6.status}"
    assert iv6.vlm_verified_at is not None and iv6.vlm_verified_at >= iv6.claimed_done_at
    vers = [o for o in iv6.vlm_observations if o.kind == "repair_verified"]
    assert vers and vers[0].confidence > 0
    assert mgr.active("RC-07") is None, "verified intervention must no longer be active"
    assert any(d.get("machine_id") == "RC-07" and "verified" in (d.get("body") or "").lower()
               for d in notifies("broadcast")), "verification broadcast missing"
    ok("claimed_done → VLM repair_verified closes the intervention (ordered timestamps)")

    # ------------------------------------------------ 6. cooldown after close
    adv3 = mk_advisory("RC-07", RiskLabel.HIGH, epoch=9)
    store.save_advisory(adv3)
    iv7 = mgr.on_advisory(adv3)
    assert iv7 is not None and iv7.id != iv.id, "closed machine gets a NEW intervention"
    assert len(notifies("alert")) == 1, "re-alert within cooldown must stay silent"
    ok("notification cooldown suppresses immediate re-alert after closure")

    # ------------------------------------------------ 7. routing to lina
    mx = mk_advisory("MX-02", RiskLabel.CRITICAL, department="Mixing", epoch=9)
    store.save_advisory(mx)
    mgr.on_advisory(mx)
    mx_alerts = [d for d in notifies("alert") if d["machine_id"] == "MX-02"]
    assert mx_alerts and mx_alerts[0]["to_operator"]["id"] == "lina"
    assert assigned_operator("MX-02")["id"] == "lina"
    assert assigned_operator("RC-07")["id"] == "eric"
    assert assigned_operator("UNKNOWN-99")["id"] in OPERATORS  # graceful fallback
    ok("MX-02 alert routed to lina (assignment map)")

    # ------------------------------------------------ 8. factory state
    store.save_reading({"machine_id": "RC-07", "department": "Curing", "epoch": 9,
                        "timestamp": time.time(),
                        "signals": {"mould_temp_C": 199.0, "vibration_rms_mm_s": 3.2},
                        "pinn": {"health_index": 0.72, "residual": 0.4,
                                 "rul_cycles": 40.0, "failure_mode_probs": {}},
                        "note": "test reading"})
    state = build_factory_state(store, mgr, mode="mock", epoch=9)
    assert state["mode"] == "mock" and state["epoch"] == 9
    machines = {m["machine_id"]: m for m in state["machines"]}
    assert {"RC-07", "CL-03", "MX-02"} <= set(machines)
    for m in machines.values():
        assert m["assigned_operator"]["id"] in OPERATORS
        assert "department" in m and "risk" in m
    rc = machines["RC-07"]
    assert rc["last_reading"] and rc["last_reading"]["signals"]["mould_temp_C"] == 199.0
    assert rc["active_advisory"] and rc["active_advisory"]["id"] == adv3.id
    assert rc["intervention"] and rc["intervention"]["id"] == iv7.id
    ok("factory_state carries readings + advisory + intervention per machine")

    # ------------------------------------------------ 9. quick chat
    turn, provenance = await quick_answer(
        "Why the alarm on RC-07?", "RC-07", store, MockClient(), mgr)
    assert turn["role"] == "agent" and turn["content"].strip()
    assert turn["citations"], "quick chat must carry programmatic citations"
    assert any("RC-07" in c for c in turn["citations"])
    assert provenance and all("source" in p and "detail" in p for p in provenance)
    srcs = " ".join(p["source"] for p in provenance)
    assert "sensor_history" in srcs and "limits" in srcs
    ok(f"quick chat cites its context: {turn['citations']}")

    # ------------------------------------------------ 10. store round-trip
    got_iv = store.get_intervention(iv7.id)
    assert got_iv is not None and got_iv.machine_id == "RC-07"
    got_iv.status = InterventionStatus.VERIFIED
    store.save_intervention(got_iv)                      # upsert
    assert store.get_intervention(iv7.id).status == InterventionStatus.VERIFIED
    assert store.get_active_intervention("RC-07") is None
    assert len(store.get_interventions(limit=10)) >= 2
    ok("interventions upsert + active lookup round-trip the store")

    # ------------------------------------------------ 11. Teams card (pure)
    card = teams_card(alerts[0])
    assert card.get("@type") == "MessageCard" and card.get("themeColor")
    assert "RC-07" in str(card)
    ok("teams_card builds a well-formed MessageCard")


def main() -> None:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    store = StateStore(db_path=tmp.name)
    events: list = []
    try:
        asyncio.run(scenario(store, events))
    finally:
        store.close()
        os.unlink(tmp.name)
    print(f"\nALL GREEN — {PASS} checks passed (operator flow)")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nFAIL: {e}", file=sys.stderr)
        sys.exit(1)
