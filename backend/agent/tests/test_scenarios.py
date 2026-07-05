"""Component-coverage harness: EXPECTED vs ACHIEVED per architecture
component, driven by the ten verified-limit scenarios. # D4-harness-v1

Run from repo root:  MOCK_LLM=1 python3 backend/agent/tests/test_scenarios.py

Builds a fresh temp-db StateStore + the real AdvisoryPipeline + the mock
client, signs every payload properly, runs S1..S10 in order, then prints a
SCORECARD (scenario | component(s) | expected | achieved | PASS/DEVIATION)
plus an 11-component coverage line, and dumps the machine-readable scorecard
to /tmp/scenario_scorecard.json. Exit code != 0 on any DEVIATION.
Also exports the deterministic synthetic dataset via scenarios.export_csv().
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["MOCK_LLM"] = "1"  # must precede any backend.agent import
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from backend.agent import debate as debate_mod  # noqa: E402
from backend.agent import line_llm  # noqa: E402
from backend.agent.crusoe_client import MockClient, get_client  # noqa: E402
from backend.agent.hmac_auth import sign_payload  # noqa: E402
from backend.agent.operator import OperatorAgent  # noqa: E402
from backend.agent.operator_agent import ToolCallingOperator  # noqa: E402
from backend.agent.operator_tools import OperatorToolbox  # noqa: E402
from backend.agent.pipeline import AdvisoryPipeline  # noqa: E402
from backend.agent.prompts import format_overrides  # noqa: E402
from backend.agent.scenarios import (S9_QUESTION, S10_OVERRIDE_REASON,  # noqa: E402
                                     SCENARIOS, export_csv)
from backend.agent.schemas import (AdvisoryStatus, PinnReading, RiskLabel,  # noqa: E402
                                   SignedReading, TriageResult)
from backend.agent.state_store import StateStore  # noqa: E402
from backend.agent.triage import pick_causal  # noqa: E402

SC = {s["id"]: s for s in SCENARIOS}
ROWS: list[dict] = []          # scorecard rows
COVER = {k: False for k in (
    "tier1", "tier2", "tier3/line-llm", "debate", "jury", "hmac", "store",
    "drift-tool", "operator-agent+tools", "override-loop", "boss")}


class Checker:
    """Collects sub-check failures for one scenario."""

    def __init__(self) -> None:
        self.fails: list[str] = []

    def expect(self, cond: bool, msg: str) -> bool:
        if not cond:
            self.fails.append(msg)
        return bool(cond)


def row(sid: str, components: str, expected: str, achieved: str,
        c: Checker) -> None:
    ROWS.append({"scenario": sid, "components": components,
                 "expected": expected, "achieved": achieved,
                 "status": "PASS" if not c.fails else "DEVIATION",
                 "fails": list(c.fails)})


class RecordingClient(MockClient):
    """MockClient that records (hint, last-user-message) for prompt asserts."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def complete(self, role, messages, **kw):
        self.calls.append((kw.get("hint", ""), self._last_user(messages)))
        return await super().complete(role, messages, **kw)

    async def complete_json(self, role, messages, schema, **kw):
        self.calls.append((kw.get("hint", ""), self._last_user(messages)))
        return await super().complete_json(role, messages, schema, **kw)


async def process(pipe: AdvisoryPipeline, store: StateStore, rd_kwargs: dict,
                  tamper: bool = False):
    """Sign properly (main.py pattern); optionally corrupt the signature."""
    reading = PinnReading.model_validate(rd_kwargs)
    payload = reading.model_dump()
    sig = sign_payload(payload)
    if tamper:
        sig = sig[:-6] + ("000000" if not sig.endswith("000000") else "ffffff")
    else:
        store.save_reading(payload, signature=sig)  # service-loop behavior
    return await pipe.process_reading(SignedReading(payload=payload, signature=sig))


async def main() -> None:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    store = StateStore(db_path=tmp.name)
    client = get_client()
    pipe = AdvisoryPipeline(client, store)
    toolbox = OperatorToolbox(store)
    ticks: dict[str, list] = {}

    # Scenarios are processed strictly IN ORDER; each check runs before the
    # next scenario is ingested (run_diagnostic reads the LATEST reading).

    # S1 nominal-steam-cure — tier1 CLEAR, no advisory, spec must not
    # false-alarm a healthy 196 degC steam cure (the old trip-195 bug class).
    ticks["S1"] = [await process(pipe, store, rd) for rd in SC["S1"]["readings"]]
    c = Checker()
    t = ticks["S1"][0]
    c.expect(t.hmac_verified, "hmac not verified")
    c.expect(t.triage is not None and t.triage.tier_reached == 1,
             f"tier_reached={t.triage.tier_reached if t.triage else None} != 1")
    c.expect(t.triage is not None and t.triage.risk == RiskLabel.CLEAR,
             f"risk={t.triage.risk.value if t.triage else None} != CLEAR")
    c.expect(t.advisory is None, "advisory produced on nominal cure")
    diag = toolbox.run_diagnostic("RC-07")
    hits = (diag.get("realtime") or {}).get("out_of_limits") or []
    c.expect(diag.get("ok") and not hits,
             f"nominal 196 degC/17.5 bar cure false-alarms the spec: {hits}")
    if not c.fails:
        COVER["tier1"] = True
    row("S1", "tier1, spec/limits", "tier1 CLEAR, no advisory, 0 spec hits",
        f"tier={t.triage.tier_reached if t.triage else '?'} "
        f"risk={t.triage.risk.value if t.triage else '?'} "
        f"adv={'yes' if t.advisory else 'no'} spec_hits={len(hits)}", c)

    # S2 zoneB-longterm — CLEAR or WATCH max, no advisory.
    ticks["S2"] = [await process(pipe, store, rd) for rd in SC["S2"]["readings"]]
    c = Checker()
    t = ticks["S2"][0]
    c.expect(t.triage is not None and
             t.triage.risk in (RiskLabel.CLEAR, RiskLabel.WATCH),
             f"risk={t.triage.risk.value if t.triage else None} above WATCH")
    c.expect(t.advisory is None, "advisory produced for zone-B vibration")
    row("S2", "tier1", "<=WATCH, no advisory (zone B acceptable long-term)",
        f"risk={t.triage.risk.value if t.triage else '?'} "
        f"adv={'yes' if t.advisory else 'no'}", c)

    # S3 zoneC-surveillance — tier2 reached, >=HIGH, advisory + debate + jury.
    ticks["S3"] = [await process(pipe, store, rd) for rd in SC["S3"]["readings"]]
    c = Checker()
    t = ticks["S3"][0]
    c.expect(t.triage is not None and t.triage.tier_reached == 2,
             "tier 2 not reached")
    c.expect(t.triage is not None and
             t.triage.risk in (RiskLabel.HIGH, RiskLabel.CRITICAL),
             f"risk={t.triage.risk.value if t.triage else None} < HIGH")
    adv = t.advisory
    c.expect(adv is not None, "no advisory")
    if adv is not None:
        c.expect(adv.debate is not None and len(adv.debate.turns) > 0,
                 "debate missing/empty")
        c.expect(adv.jury is not None, "jury missing")
    if not c.fails:
        COVER["tier2"] = True
        COVER["tier3/line-llm"] = True
    row("S3", "tier1->tier2, tier3/line-llm, debate, jury",
        "tier2, >=HIGH, advisory WITH debate AND jury",
        f"tier={t.triage.tier_reached if t.triage else '?'} "
        f"risk={t.triage.risk.value if t.triage else '?'} "
        f"adv={'yes' if adv else 'no'} "
        f"debate={'yes' if adv and adv.debate else 'no'} "
        f"jury={'yes' if adv and adv.jury else 'no'}", c)

    # S4 zoneD-overheat-critical — CRITICAL; tier1 reasons name limit+value;
    # jury passed; message cites digits.
    ticks["S4"] = [await process(pipe, store, rd) for rd in SC["S4"]["readings"]]
    c = Checker()
    t = ticks["S4"][0]
    reason = t.triage.reason if t.triage else ""
    c.expect(t.triage is not None and t.triage.risk == RiskLabel.CRITICAL,
             f"risk={t.triage.risk.value if t.triage else None} != CRITICAL")
    c.expect("210" in reason and "214" in reason,
             f"tier1 reason does not name mould_temp value+limit: {reason!r}")
    c.expect("zone D" in reason, f"tier1 reason lacks ISO zone D: {reason!r}")
    c.expect("16" in reason, f"tier1 reason lacks pressure band: {reason!r}")
    adv = t.advisory
    c.expect(adv is not None, "no advisory")
    if adv is not None:
        c.expect(adv.debate is not None, "debate missing")
        c.expect(adv.jury is not None and adv.jury.passed,
                 f"jury not passed: {adv.jury}")
        c.expect(any(ch.isdigit() for ch in adv.message),
                 "message cites no digits")
    if not c.fails:
        COVER["debate"] = True
        COVER["jury"] = True
    row("S4", "tier1 limit-naming, tier2, tier3, debate, jury",
        "CRITICAL; reasons cite 214>210, zone D, pressure<16; jury passed",
        f"risk={t.triage.risk.value if t.triage else '?'} "
        f"jury_passed={adv.jury.passed if adv and adv.jury else None} "
        f"reason={reason[:70]}...", c)

    # S5 pressure-loss-undercure — >=HIGH advisory; causal/justification
    # mentions pressure.
    ticks["S5"] = [await process(pipe, store, rd) for rd in SC["S5"]["readings"]]
    c = Checker()
    t = ticks["S5"][0]
    c.expect(t.triage is not None and
             t.triage.risk in (RiskLabel.HIGH, RiskLabel.CRITICAL),
             f"risk={t.triage.risk.value if t.triage else None} < HIGH")
    adv = t.advisory
    c.expect(adv is not None, "no advisory")
    causal_txt = " ".join((t.triage.causal_context or {}).values()) if t.triage else ""
    just = (adv.justification if adv else "")
    c.expect("pressure" in (causal_txt + " " + just).lower(),
             "neither causal context nor justification mentions pressure")
    row("S5", "tier2, tier3/line-llm, causal matrix",
        ">=HIGH advisory; causal/justification mentions pressure",
        f"risk={t.triage.risk.value if t.triage else '?'} "
        f"adv={'yes' if adv else 'no'} "
        f"pressure_mentioned={'pressure' in (causal_txt + ' ' + just).lower()}", c)

    # S6 cycle-overrun — tier1 flags; WATCH or HIGH (not CLEAR, not
    # CRITICAL); advisory only if HIGH.
    ticks["S6"] = [await process(pipe, store, rd) for rd in SC["S6"]["readings"]]
    c = Checker()
    t = ticks["S6"][0]
    r = t.triage.risk if t.triage else None
    c.expect(r is not None and r != RiskLabel.CLEAR,
             "cycle 31 min (>30 extended max) triaged CLEAR")
    c.expect(r in (RiskLabel.WATCH, RiskLabel.HIGH),
             f"risk={r.value if r else None} not in WATCH/HIGH")
    c.expect("cycle" in (t.triage.reason.lower() if t.triage else ""),
             f"tier1 reason does not flag the cycle: {t.triage.reason if t.triage else ''!r}")
    if r == RiskLabel.HIGH:
        c.expect(t.advisory is not None, "HIGH but no advisory")
    else:
        c.expect(t.advisory is None, "advisory despite non-HIGH risk")
    row("S6", "tier1 cycle check", "flags 31>30 overrun; WATCH/HIGH not CLEAR;"
        " advisory iff HIGH",
        f"risk={r.value if r else '?'} adv={'yes' if t.advisory else 'no'} "
        f"reason={t.triage.reason[:60] if t.triage else ''}...", c)

    # ---------------------------------------------------------- S7 tampered
    c = Checker()
    n_adv_before = len(store.get_advisories(limit=200))
    t = await process(pipe, store, SC["S7"]["readings"][0], tamper=True)
    c.expect(not t.hmac_verified, "tampered signature verified!")
    c.expect(bool(t.rejected_reason), "no rejected_reason")
    c.expect(t.triage is None and t.advisory is None,
             "triage/advisory ran on a rejected reading")
    c.expect(bool(store.get_events(kind="hmac_rejected")),
             "hmac_rejected event not logged")
    c.expect(len(store.get_advisories(limit=200)) == n_adv_before,
             "advisory count changed on rejected reading")
    if not c.fails:
        COVER["hmac"] = True
    row("S7", "hmac, store(events)",
        "rejected before triage; hmac_rejected logged; no advisory",
        f"verified={t.hmac_verified} triage={'yes' if t.triage else 'no'} "
        f"event={'yes' if store.get_events(kind='hmac_rejected') else 'no'}", c)

    # --------------------------------------------------------- S8 slow drift
    c = Checker()
    s8_ticks = [await process(pipe, store, rd) for rd in SC["S8"]["readings"]]
    ticks["S8"] = s8_ticks
    bad = [(t.epoch, t.triage.risk.value) for t in s8_ticks
           if t.triage is None or t.triage.risk not in (RiskLabel.CLEAR,
                                                        RiskLabel.WATCH)]
    c.expect(not bad, f"S8 ticks above WATCH: {bad}")
    c.expect(all(t.advisory is None for t in s8_ticks),
             "advisory produced during benign slow drift")
    drift = toolbox.analyze_drift("RC-07")
    c.expect(bool(drift.get("ok")), f"analyze_drift failed: {drift.get('error')}")
    c.expect("vibration_rms_mm_s" in (drift.get("drifting") or []),
             f"vibration not reported drifting: {drift.get('drifting')}")
    c.expect("DRIFTING" in str(drift.get("verdict", "")),
             f"verdict lacks DRIFTING: {drift.get('verdict')}")
    if not c.fails:
        COVER["drift-tool"] = True
    row("S8", "tier1/tier2 (10 ticks), drift-tool",
        "each tick <=WATCH, no advisory; analyze_drift flags vibration",
        f"risks={sorted({t.triage.risk.value for t in s8_ticks if t.triage})} "
        f"advisories={sum(1 for t in s8_ticks if t.advisory)} "
        f"drifting={drift.get('drifting')}", c)

    # ----------------------------------------------------- S9 operator agent
    c = Checker()
    agent = ToolCallingOperator(store)
    turn, provenance = await agent.answer(S9_QUESTION)
    content = turn.content
    digit_markers = [m for m in ("214", "3.2", "80") if m in content]
    c.expect(len(digit_markers) >= 2,
             f"<2 stored-data digits in answer (found {digit_markers})")
    # @integration (2026-07-05): the operator-facing sources line is now the
    # compact "_Checked: …_" footer (full machine trail stays on
    # turn.provenance) — accept either wording, same guarantee.
    c.expect(("Data Provenance" in content) or ("Checked:" in content),
             "no sources line (Data Provenance/Checked)")
    tools_used = {p["tool"] for p in provenance}
    c.expect(len(tools_used) >= 2, f"<2 tools in provenance: {tools_used}")
    if not c.fails:
        COVER["operator-agent+tools"] = True
    row("S9", "operator-agent+tools",
        ">=2 real digits, sources line, >=2 tools",
        f"digits={digit_markers} tools={sorted(tools_used)}", c)

    # -------------------------------------------------- S10 override feedback
    c = Checker()
    rec = None
    s3_adv = ticks["S3"][0].advisory
    c.expect(s3_adv is not None, "S3 advisory unavailable for override")
    if s3_adv is not None:
        op = OperatorAgent()
        updated = op.handle_override(s3_adv.id, "overridden",
                                     S10_OVERRIDE_REASON, store)
        c.expect(updated is not None and
                 updated.status == AdvisoryStatus.OVERRIDDEN,
                 "override did not update advisory status")
        ovs = store.get_overrides(limit=20)
        c.expect(any(o.reason == S10_OVERRIDE_REASON for o in ovs),
                 f"override reason not in store: {[o.reason for o in ovs]}")
        c.expect(S10_OVERRIDE_REASON in format_overrides(ovs),
                 "format_overrides drops the pushback reason")
        # run_debate accepts overrides (contract feed-forward)
        c.expect("overrides" in inspect.signature(debate_mod.run_debate).parameters,
                 "run_debate has no overrides parameter")
        # fresh debate on an S3-like reading must carry the pushback text
        reading = PinnReading.model_validate(SC["S10"]["readings"][0])
        triage = TriageResult(tier_reached=2, risk=RiskLabel.HIGH,
                              reason="s10 replay", causal_context=pick_causal(reading))
        draft = line_llm.fallback_draft(reading, triage)
        rec = RecordingClient()
        deb = await debate_mod.run_debate(rec, draft, reading, ovs)
        skeptic_prompts = [p for h, p in rec.calls if h == "skeptic"]
        c.expect(bool(skeptic_prompts), "no skeptic prompt captured")
        c.expect(any(S10_OVERRIDE_REASON in p for p in skeptic_prompts),
                 "skeptic context lacks the override pushback text")
        c.expect(deb is not None and bool(deb.turns), "fresh debate produced no turns")
    if not c.fails:
        COVER["override-loop"] = True
        COVER["store"] = True
    row("S10", "override-loop, store, debate(skeptic context)",
        "override stored + fed to fresh debate skeptic context",
        f"overrides={len(store.get_overrides())} "
        f"skeptic_has_reason="
        f"{any(S10_OVERRIDE_REASON in p for h, p in (rec.calls if rec else []) if h == 'skeptic')}",
        c)

    # ------------------------------------------------------------- boss view
    c = Checker()
    summary = await pipe.boss_summary()
    c.expect("Curing" in summary.department_status,
             f"boss summary misses Curing: {summary.department_status}")
    c.expect(summary.open_advisories >= 1,
             f"open advisories {summary.open_advisories} < 1")
    c.expect(bool(summary.text.strip()), "boss text empty")
    if not c.fails:
        COVER["boss"] = True
    row("BOSS", "boss", "covers Curing; open advisories >= 1",
        f"depts={list(summary.department_status)} "
        f"open={summary.open_advisories}", c)

    # ------------------------------------------------------------ CSV export
    csv_rows = export_csv()
    from backend.agent.scenarios import DEFAULT_CSV
    print(f"\n  synthetic dataset: {DEFAULT_CSV} ({csv_rows} data rows)")

    # -------------------------------------------------------------- scorecard
    print("\n" + "=" * 118)
    print("SCORECARD — EXPECTED vs ACHIEVED per architecture component")
    print("=" * 118)
    hdr = f"{'scen':5} | {'component(s)':44} | {'expected':52} | status"
    print(hdr)
    print("-" * 118)
    for r in ROWS:
        print(f"{r['scenario']:5} | {r['components'][:44]:44} | "
              f"{r['expected'][:52]:52} | {r['status']}")
        print(f"{'':5} |   achieved: {r['achieved'][:100]}")
        for f in r["fails"]:
            print(f"{'':5} |   DEVIATION -> {f}")
    print("-" * 118)
    cov_line = " ".join(
        f"{name}:{'OK' if ok_ else 'X'}" for name, ok_ in COVER.items())
    n_dev = sum(1 for r in ROWS if r["status"] == "DEVIATION")
    print(f"COMPONENT COVERAGE ({sum(COVER.values())}/{len(COVER)}): {cov_line}")
    print(f"RESULT: {len(ROWS) - n_dev}/{len(ROWS)} rows PASS, "
          f"{n_dev} DEVIATION(S)")

    out_dir = Path("/tmp") if Path("/tmp").is_dir() else Path(tempfile.gettempdir())
    scorecard = {"rows": ROWS, "coverage": COVER,
                 "all_pass": n_dev == 0, "csv_rows": csv_rows,
                 "csv_path": str(DEFAULT_CSV)}
    (out_dir / "scenario_scorecard.json").write_text(
        json.dumps(scorecard, indent=2, default=str), encoding="utf-8")
    print(f"scorecard JSON: {out_dir / 'scenario_scorecard.json'}")

    store.close()
    try:
        os.unlink(tmp.name)
    except OSError:
        pass
    if n_dev:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
