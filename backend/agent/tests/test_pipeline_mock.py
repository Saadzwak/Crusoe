"""Plain-script test of the AdvisoryPipeline, fully offline (mock mode).

Run from anywhere:  python3 backend/agent/tests/test_pipeline_mock.py
Exits non-zero on the first inconsistency; prints a readable transcript.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

os.environ["MOCK_LLM"] = "1"  # must precede any backend.agent import
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from backend.agent.crusoe_client import get_client                     # noqa: E402
from backend.agent.hmac_auth import sign_payload                       # noqa: E402
from backend.agent.pipeline import AdvisoryPipeline                    # noqa: E402
from backend.agent.schemas import (AdvisoryStatus, PinnReading,        # noqa: E402
                                   PinnState, RiskLabel, SignedReading)

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    tag = "ok " if cond else "FAIL"
    print(f"  [{tag}] {msg}")
    if not cond:
        FAILURES.append(msg)


# ------------------------------------------------- fake store (duck-typed)
class FakeStore:
    """In-memory stand-in for Builder B's SQLite StateStore (same methods)."""

    def __init__(self):
        self.ticks, self.advisories, self.overrides, self.events = [], {}, [], []

    def save_tick(self, tick):
        self.ticks.append(tick)

    def save_advisory(self, adv):
        self.advisories[adv.id] = adv

    def save_override(self, ov):
        self.overrides.append(ov)

    def get_advisories(self, status=None, limit=50):
        out = [a for a in self.advisories.values() if status is None or a.status == status]
        return out[:limit]

    def get_advisory(self, advisory_id):
        return self.advisories.get(advisory_id)

    def get_sensor_history(self, machine_id, limit=40):
        return [{"epoch": t.epoch, "machine_id": t.machine_id}
                for t in self.ticks if t.machine_id == machine_id][-limit:]

    def get_overrides(self, limit=20):
        return self.overrides[-limit:]

    def department_snapshot(self):
        snap = {}
        for t in self.ticks:
            if t.triage:
                snap[t.department] = {"risk": t.triage.risk.value, "epoch": t.epoch}
        return snap

    def log_event(self, kind, data):
        self.events.append((kind, data))


def signed(reading: PinnReading) -> SignedReading:
    payload = reading.model_dump()
    return SignedReading(payload=payload, signature=sign_payload(payload))


HEALTHY = PinnReading(
    machine_id="MX-02", department="Mixing", epoch=7,
    signals={"motor_current_a": 41.2, "vibration_rms": 1.1, "temp_c": 78.0},
    pinn=PinnState(health_index=0.97, rul_cycles=180, residual=0.03),
    note="",
)

# Keywords ("critical", "failure", "imminent", residual 0.9x) drive the mock
# tier-2 classifier to CRITICAL — see crusoe_client.MockClient.
CRITICAL = PinnReading(
    machine_id="RC-07", department="Curing", epoch=42,
    signals={"mould_temp_c": 178.4, "press_vibration_rms": 9.6, "cycle_drift_s": 14.2},
    pinn=PinnState(health_index=0.08, rul_cycles=3, residual=0.93,
                   failure_mode_probs={"HDF": 0.85}),
    note="Critical thermal signature on press RC-07: heat dissipation failure imminent.",
)


async def main() -> None:
    store = FakeStore()
    pipe = AdvisoryPipeline(get_client(), store)

    print("\n=== 1. healthy reading (MX-02, Mixing) — expect Tier-1 CLEAR, no advisory")
    tick = await pipe.process_reading(signed(HEALTHY))
    check(tick.hmac_verified, "HMAC verified")
    check(tick.triage is not None and tick.triage.tier_reached == 1, "stopped at tier 1")
    check(tick.triage is not None and tick.triage.risk == RiskLabel.CLEAR, "risk CLEAR")
    check(tick.advisory is None, "no advisory produced")
    check("tier1" in (tick.triage.latency_ms if tick.triage else {}), "tier-1 latency recorded")

    print("\n=== 2. critical curing reading (RC-07) — expect advisory + debate + jury")
    tick = await pipe.process_reading(signed(CRITICAL))
    check(tick.hmac_verified, "HMAC verified")
    check(tick.triage is not None and tick.triage.tier_reached >= 2, "escalated past tier 1")
    check(tick.triage is not None and tick.triage.risk == RiskLabel.CRITICAL, "risk CRITICAL")
    adv = tick.advisory
    check(adv is not None, "advisory produced")
    if adv is not None:
        check(adv.severity == RiskLabel.CRITICAL, "advisory severity from triage (CRITICAL)")
        check(adv.debate is not None and len(adv.debate.turns) > 0, "debate transcript non-empty")
        check(adv.jury is not None, "jury score attached")
        check(any(c.isdigit() for c in adv.message), "operator message cites values (digits)")
        check(adv.status == AdvisoryStatus.PENDING, "status pending (human decides)")
        check("tag" in (tick.triage.causal_context or {}), "causal context attached")
        check(adv.id in store.advisories, "advisory persisted to store")

        print("\n  --- advisory card -------------------------------------------")
        print(f"  [{adv.severity.value}] {adv.title}")
        print(f"  message : {adv.message}")
        print(f"  action  : {adv.recommended_action}")
        print(f"  eur     : {adv.eur_impact}")
        print(f"  safety  : {adv.safety_impact}   ttf: {adv.ttf_estimate}")
        print("  --- debate transcript ---------------------------------------")
        for turn in adv.debate.turns:
            print(f"  R{turn.round} {turn.role:>8}: {turn.content[:110]}")
        print(f"  verdict: {adv.debate.verdict} (rounds={adv.debate.rounds_used})")
        print(f"  final proposal: {adv.debate.final_proposal[:110]}")
        if adv.jury:
            j = adv.jury
            print("  --- jury ----------------------------------------------------")
            print(f"  grounding={j.grounding} safety={j.safety} actionability={j.actionability} "
                  f"human_in_charge={j.human_in_charge} overall={j.overall} passed={j.passed}")
            print(f"  critique: {j.critique}")

    print("\n=== 3. tampered HMAC — expect rejection before any processing")
    good = signed(CRITICAL)
    tampered = SignedReading(payload={**good.payload, "epoch": 999}, signature=good.signature)
    tick = await pipe.process_reading(tampered)
    check(not tick.hmac_verified, "hmac_verified is False")
    check(bool(tick.rejected_reason), f"rejected_reason set ({tick.rejected_reason!r})")
    check(tick.triage is None and tick.advisory is None, "no triage, no advisory")
    check(any(k == "hmac_rejected" for k, _ in store.events), "hmac_rejected event logged")

    print("\n=== 4. boss summary — reads the shared store only")
    summary = await pipe.boss_summary()
    check(bool(summary.text.strip()), "summary text non-empty")
    check(summary.open_advisories >= 1, f"open advisories counted ({summary.open_advisories})")
    check(summary.generated_by == "mock", f"generated_by={summary.generated_by!r}")
    check("Curing" in summary.department_status and "Mixing" in summary.department_status,
          f"department status covers both departments ({summary.department_status})")
    print(f"  boss: {summary.text[:160]}")

    stages = {k for k, _ in store.events}
    print(f"\n  flight recorder events: {sorted(stages)}")
    check({"triage", "advisory", "debate", "jury", "hmac_rejected"} <= stages,
          "log_event fired per stage")

    if FAILURES:
        print(f"\nRESULT: {len(FAILURES)} FAILURE(S)")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("\nRESULT: ALL CHECKS PASSED (mock mode)")


if __name__ == "__main__":
    asyncio.run(main())
