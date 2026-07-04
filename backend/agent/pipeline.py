"""AdvisoryPipeline — the orchestrator Builder B's loop calls per reading.

verify HMAC -> parse -> Tier1 -> (Tier2) -> if HIGH/CRITICAL:
Tier3 draft -> debate -> jury -> Advisory persisted.

Contract guarantees:
- process_reading NEVER raises: every stage is wrapped; a dead stage degrades
  (advisory without debate, advisory without jury, deterministic fallback
  draft) and the degradation is written into the advisory justification.
- Everything lands in the shared store: save_advisory, save_tick, and a
  log_event per stage (flight-recorder pattern).
- Per-stage latencies land in TriageResult.latency_ms (tier1/tier2/tier3/
  debate/jury) so the demo can show the tier transitions with timings.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from . import boss_llm, debate as debate_mod, jury as jury_mod, line_llm, triage as triage_mod
from .crusoe_client import LLMClient
from .hmac_auth import verify_payload
from .prompts import reading_digest
from .schemas import (Advisory, PinnReading, PlantSummary, RiskLabel,
                      SignedReading, TickResult)

_ADVISORY_RISKS = (RiskLabel.HIGH, RiskLabel.CRITICAL)


class AdvisoryPipeline:
    def __init__(self, client: LLMClient, store, knowledge=None) -> None:
        self.client = client
        self.store = store
        self.knowledge = knowledge

    # ------------------------------------------------------------- helpers
    def _safe_store(self, method: str, *args: Any) -> None:
        try:
            getattr(self.store, method)(*args)
        except Exception as e:  # noqa: BLE001 — the store must never kill a tick
            print(f"[pipeline] store.{method} failed: {type(e).__name__}: {e}")

    def _log(self, kind: str, data: dict) -> None:
        self._safe_store("log_event", kind, data)

    @staticmethod
    def _ids(payload: dict) -> tuple[str, str, int]:
        try:
            return (str(payload.get("machine_id", "unknown")),
                    str(payload.get("department", "unknown")),
                    int(payload.get("epoch", 0) or 0))
        except Exception:  # noqa: BLE001
            return "unknown", "unknown", 0

    def _reject(self, payload: dict, reason: str, verified: bool) -> TickResult:
        machine_id, department, epoch = self._ids(payload if isinstance(payload, dict) else {})
        tick = TickResult(machine_id=machine_id, department=department, epoch=epoch,
                          hmac_verified=verified, rejected_reason=reason)
        self._log("hmac_rejected" if not verified else "reading_rejected",
                  {"machine_id": machine_id, "epoch": epoch, "reason": reason})
        self._safe_store("save_tick", tick)
        return tick

    # ------------------------------------------------------------ pipeline
    async def process_reading(self, signed: SignedReading) -> TickResult:
        """Full tick. Must never raise — outer catch is the last resort."""
        payload: dict = {}
        try:
            if not isinstance(signed, SignedReading):
                signed = SignedReading.model_validate(signed)
            payload = signed.payload if isinstance(signed.payload, dict) else {}

            if not verify_payload(payload, signed.signature):
                return self._reject(payload, "HMAC signature mismatch — origin not authenticated", False)

            try:
                reading = PinnReading.model_validate(payload)
            except Exception as e:  # noqa: BLE001
                return self._reject(payload, f"malformed payload: {e}", True)

            return await self._process_verified(reading)
        except Exception as e:  # noqa: BLE001 — absolute backstop
            machine_id, department, epoch = self._ids(payload)
            self._log("pipeline_error", {"machine_id": machine_id, "error": repr(e)[:200]})
            return TickResult(machine_id=machine_id, department=department, epoch=epoch,
                              hmac_verified=True, rejected_reason=f"pipeline error: {e}")

    async def _process_verified(self, reading: PinnReading) -> TickResult:
        triage = await triage_mod.run_triage(self.client, reading)
        self._log("triage", {"machine_id": reading.machine_id, "epoch": reading.epoch,
                             "tier": triage.tier_reached, "risk": triage.risk.value,
                             "latency_ms": triage.latency_ms})

        advisory: Optional[Advisory] = None
        if triage.risk in _ADVISORY_RISKS:
            advisory = await self._advisory_stage(reading, triage)
            if advisory is not None:
                self._safe_store("save_advisory", advisory)
                self._log("advisory", {"advisory_id": advisory.id,
                                       "machine_id": advisory.machine_id,
                                       "severity": advisory.severity.value,
                                       "verdict": advisory.debate.verdict if advisory.debate else "none",
                                       "jury_passed": advisory.jury.passed if advisory.jury else None})

        tick = TickResult(machine_id=reading.machine_id, department=reading.department,
                          epoch=reading.epoch, hmac_verified=True,
                          triage=triage, advisory=advisory)
        self._safe_store("save_tick", tick)
        return tick

    async def _advisory_stage(self, reading: PinnReading, triage) -> Optional[Advisory]:
        notes: list[str] = []
        t0 = time.perf_counter()
        try:
            draft = await line_llm.draft_advisory(self.client, reading, triage, self.knowledge)
        except Exception as e:  # noqa: BLE001 — degrade to deterministic draft
            notes.append(f"Line-LLM unavailable ({type(e).__name__}); deterministic fallback used.")
            draft = line_llm.fallback_draft(reading, triage)
        triage.latency_ms["tier3"] = round((time.perf_counter() - t0) * 1000.0, 1)

        t1 = time.perf_counter()
        try:
            overrides = self.store.get_overrides(limit=20) or []
        except Exception:  # noqa: BLE001
            overrides = []
        try:
            debate = await debate_mod.run_debate(self.client, draft, reading, overrides)
        except Exception as e:  # noqa: BLE001 — advisory without debate, flagged
            debate = None
            notes.append(f"Debate unavailable ({type(e).__name__}); advisory not adversarially reviewed.")
        triage.latency_ms["debate"] = round((time.perf_counter() - t1) * 1000.0, 1)
        if debate is not None:
            self._log("debate", {"machine_id": reading.machine_id, "verdict": debate.verdict,
                                 "rounds": debate.rounds_used})

        advisory = line_llm.build_advisory(draft, reading, severity=triage.risk, debate=debate)
        if debate is not None and debate.verdict == "escalated":
            advisory.recommended_action = debate.final_proposal  # safe fallback
            notes.append("Debate escalated: skeptic upheld an objection; action replaced by the safe hold.")
        elif debate is not None and debate.verdict == "revised":
            advisory.recommended_action = debate.final_proposal

        # Jury scores EVERY advisory that will reach the operator — including
        # escalated ones and debate-failures (contract rule).
        jury = None
        t2 = time.perf_counter()
        try:
            jury = await jury_mod.judge_advisory(self.client, advisory, reading_digest(reading))
        except Exception as e:  # noqa: BLE001 — advisory without jury, flagged
            notes.append(f"Jury unavailable ({type(e).__name__}); advisory shown unscored.")
        triage.latency_ms["jury"] = round((time.perf_counter() - t2) * 1000.0, 1)
        if jury is not None:
            self._log("jury", {"machine_id": reading.machine_id, "overall": jury.overall,
                               "passed": jury.passed})

        advisory.jury = jury
        if jury is not None and not jury.passed:
            advisory.title = f"[UNVERIFIED] {advisory.title}"[:120]
            notes.append(f"Jury flagged (overall {jury.overall:.1f}/5): {jury.critique}")
        if notes:
            advisory.justification = (advisory.justification + "\n" +
                                      "\n".join(f"[pipeline] {n}" for n in notes)).strip()
        return advisory

    # ---------------------------------------------------------------- boss
    async def boss_summary(self) -> PlantSummary:
        try:
            summary = await boss_llm.generate_plant_summary(self.client, self.store)
        except Exception as e:  # noqa: BLE001
            summary = PlantSummary(text=f"Plant summary unavailable: {e}", generated_by="error")
        self._log("boss_summary", {"open_advisories": summary.open_advisories,
                                   "generated_by": summary.generated_by})
        return summary
