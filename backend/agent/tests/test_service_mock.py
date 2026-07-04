"""Service-layer test (Builder B) — plain runnable script, mock mode, no pipeline needed.

Run from repo root:  python3 backend/agent/tests/test_service_mock.py
Exit code 0 = green. Asserts:
  1. HMAC verifies for every generated reading (12 epochs × 3 machines)
  2. Curing (RC-07) goes anomalous by late epochs; healthy notes carry no trigger words
  3. knowledge.lookup("cost of downtime curing") → ≥1 chunk cited [site_dossier p.N]
  4. StateStore round-trips an Advisory and an OperatorOverride
  5. sensor history is oldest→newest with signals+pinn; department snapshot is sane
  6. OperatorAgent answers with citations + handles the override path
Uses backend.agent.pipeline if Builder A has landed it; otherwise stores echo ticks.
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MOCK_LLM", "1")  # before any backend.agent import

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from backend.agent.crusoe_client import MockClient  # noqa: E402
from backend.agent.hmac_auth import verify_payload  # noqa: E402
from backend.agent.knowledge import Knowledge  # noqa: E402
from backend.agent.operator import OperatorAgent  # noqa: E402
from backend.agent.schemas import (Advisory, AdvisoryStatus, PinnReading,  # noqa: E402
                                   RiskLabel, TickResult, TriageResult)
from backend.agent.state_store import StateStore  # noqa: E402
from backend.agent.telemetry_source import (CRITICAL_WORDS, HIGH_WORDS,  # noqa: E402
                                            TelemetrySource, note_risk)

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
    client = MockClient()

    # optional: Builder A's pipeline
    pipeline = None
    try:
        from backend.agent.pipeline import AdvisoryPipeline  # type: ignore
        pipeline = AdvisoryPipeline(client, store, knowledge=knowledge)
        print("  (pipeline.py found — running readings through AdvisoryPipeline)")
    except Exception:
        print("  (pipeline.py not available — storing echo ticks directly)")

    # ---------------------------------------------------------------- 1+2 loop
    curing_notes: dict[int, str] = {}
    for epoch in range(12):
        batch = ts.next_batch(epoch)
        assert len(batch) == 3, f"expected 3 readings/epoch, got {len(batch)}"
        for signed in batch:
            assert verify_payload(signed.payload, signed.signature), \
                f"HMAC failed for {signed.payload.get('machine_id')} @ epoch {epoch}"
            tampered = dict(signed.payload, epoch=999)
            assert not verify_payload(tampered, signed.signature), "tampered payload verified!"

            reading = PinnReading.model_validate(signed.payload)
            if reading.machine_id == "RC-07":
                curing_notes[epoch] = reading.note
            store.save_reading(signed.payload)

            if pipeline is not None:
                tick = asyncio.run(pipeline.process_reading(signed))
            else:
                risk = note_risk(reading.note)
                tick = TickResult(
                    machine_id=reading.machine_id, department=reading.department,
                    epoch=epoch, hmac_verified=True,
                    triage=TriageResult(tier_reached=1 if risk == RiskLabel.CLEAR else 2,
                                        risk=risk, reason="test echo"),
                )
                store.save_tick(tick)
    ok("HMAC verifies for all 36 readings (and rejects a tampered payload)")

    # determinism: same epoch → same signals
    b_a = ts.next_batch(4)[0].payload["signals"]
    b_b = ts.next_batch(4)[0].payload["signals"]
    assert b_a == b_b, "telemetry is not deterministic"
    ok("telemetry deterministic per epoch")

    # healthy early notes: no mock trigger words
    triggers = tuple(CRITICAL_WORDS) + tuple(HIGH_WORDS)
    for e in range(0, 5):
        low = curing_notes[e].lower()
        assert not any(w in low for w in triggers), \
            f"healthy epoch {e} note contains a trigger word: {curing_notes[e]!r}"
    ok("early Curing notes are trigger-free (mock tier2 stays CLEAR)")

    # late epochs: Curing anomalous → CRITICAL trigger words present
    late = curing_notes[11].lower()
    assert any(w in late for w in CRITICAL_WORDS), f"late note not critical: {late!r}"
    assert note_risk(curing_notes[11]) == RiskLabel.CRITICAL
    assert note_risk(curing_notes[7]) == RiskLabel.HIGH, "mid-run Curing should drift HIGH"
    ok("Curing arc: CLEAR early → HIGH drift mid → CRITICAL by epoch 10+")

    # Calendering spike at epoch 8, healthy around it; Mixing always nominal
    cl8 = PinnReading.model_validate(ts.next_batch(8)[1].payload)
    assert cl8.machine_id == "CL-03" and note_risk(cl8.note) == RiskLabel.HIGH
    assert cl8.pinn.failure_mode_probs.get("HDF", 0) > 0.5, "HDF prob missing on spike"
    cl7 = PinnReading.model_validate(ts.next_batch(7)[1].payload)
    assert note_risk(cl7.note) == RiskLabel.CLEAR
    for e in range(12):
        mx = PinnReading.model_validate(ts.next_batch(e)[2].payload)
        assert note_risk(mx.note) == RiskLabel.CLEAR, f"Mixing not nominal at {e}"
    ok("Calendering HDF spike at epoch 8 only; Mixing nominal throughout")

    # ---------------------------------------------------------------- 3 knowledge
    hits = knowledge.lookup("cost of downtime curing", k=3)
    assert hits and all(len(h) == 2 for h in hits)
    assert any(re.search(r"\[site_dossier p\.\d+\]", cit) for _txt, cit in hits), \
        f"no site_dossier citation in {[c for _t, c in hits]}"
    ok(f"knowledge.lookup cites the dossier: {hits[0][1]}")

    dc = knowledge.downtime_context()
    assert any("€" in t or "loss" in t.lower() for t, _c in dc), "downtime context empty"
    ok(f"downtime_context returns economics snippet: {dc[0][1]}")

    gap = knowledge.lookup("zzz quantum flux capacitor", k=3)
    assert gap and gap[0][1] == "[knowledge gap]" and "no dossier coverage" in gap[0][0]
    ok("zero-score lookup returns explicit gap marker (gbrain pattern)")

    # ---------------------------------------------------------------- 4 store round-trip
    adv = Advisory(
        machine_id="RC-07", department="Curing", epoch=11, severity=RiskLabel.CRITICAL,
        title="Bearing degradation trend on curing press",
        message="Vibration and mould temperature trending toward the failure envelope.",
        justification="vibration_rms 6.9 mm/s vs 2.1 baseline [sensor_history RC-07]; "
                      "downtime ≈€20–30k/h [site_dossier p.5]",
        recommended_action="Schedule bearing swap in next planned window (~48 h).",
        eur_impact="≈€80–120k direct for an unplanned 4-h stop [site_dossier p.5]",
        ttf_estimate="~13 cycles",
    )
    store.save_advisory(adv)
    got = store.get_advisory(adv.id)
    assert got is not None and got.title == adv.title and got.status == AdvisoryStatus.PENDING
    adv2 = got.model_copy(update={"severity": RiskLabel.HIGH})
    store.save_advisory(adv2)  # upsert
    assert store.get_advisory(adv.id).severity == RiskLabel.HIGH
    assert len(store.get_advisories(status="pending")) >= 1
    ok("Advisory round-trip + upsert by id")

    operator = OperatorAgent()
    updated = operator.handle_override(adv.id, "overridden",
                                       "press already swapped on night shift", store)
    assert updated is not None and updated.status == AdvisoryStatus.OVERRIDDEN
    assert updated.override_reason.startswith("press already")
    ovs = store.get_overrides()
    assert ovs and ovs[0].advisory_id == adv.id and ovs[0].decision == "overridden"
    assert operator.handle_override("nope-404", "accepted", "", store) is None
    ok("override handler updates status + persists OperatorOverride")

    # ---------------------------------------------------------------- 5 history/snapshot
    hist = store.get_sensor_history("RC-07", limit=40)
    assert len(hist) == 12, f"expected 12 RC-07 readings, got {len(hist)}"
    epochs = [h["epoch"] for h in hist]
    assert epochs == sorted(epochs), "history not oldest→newest"
    assert "signals" in hist[-1] and "pinn" in hist[-1]
    assert hist[-1]["pinn"]["health_index"] < hist[0]["pinn"]["health_index"], \
        "Curing health did not degrade"
    ok("sensor history oldest→newest with signals+pinn; health degrades")

    snap = store.department_snapshot()
    assert set(snap) == {"Curing", "Calendering", "Mixing"}, f"snapshot depts: {set(snap)}"
    assert snap["Curing"]["machine_id"] == "RC-07"
    if pipeline is None:  # echo ticks mirror the notes exactly
        assert snap["Curing"]["risk"] == "CRITICAL"
        assert snap["Mixing"]["risk"] == "CLEAR"
    ok(f"department snapshot: { {d: v['risk'] for d, v in snap.items()} }")

    # ---------------------------------------------------------------- 6 operator Q&A
    turn = asyncio.run(operator.answer("Why is RC-07 flagged and what would a stop cost?",
                                       store, knowledge, client))
    assert turn.role == "agent" and turn.content
    assert turn.citations, f"no citations parsed from: {turn.content!r}"
    ok(f"operator answer carries citations: {turn.citations}")

    async def collect_stream():
        toks, final = [], None
        async for ev in operator.stream_answer("status of calendering?", store,
                                               knowledge, client):
            if ev["type"] == "token":
                toks.append(ev["text"])
            elif ev["type"] == "turn":
                final = ev["turn"]
        return toks, final

    toks, final = asyncio.run(collect_stream())
    assert toks and final is not None
    assert "".join(toks) == final["content"]
    ok("streamed answer == final turn content (SSE-friendly variant)")

    store.log_event("test", {"ok": True})
    assert store.get_events(kind="test")
    store.close()
    os.unlink(tmp.name)
    print(f"\nALL GREEN — {PASS} checks passed"
          + (" (with Builder A pipeline)" if pipeline else " (echo mode, no pipeline)"))


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nFAIL: {e}", file=sys.stderr)
        sys.exit(1)
