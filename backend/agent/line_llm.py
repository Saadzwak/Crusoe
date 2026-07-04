"""Tier 3 — the Line-LLM: structured advisory drafting on the reasoning model.

Produces an AdvisoryDraft (intermediate schema, all fields required so the
offline mock populates every one), then maps it to the shared Advisory.

Grounding guard: an operator-facing message MUST carry measured values. If the
model (or the offline mock) returned a message with no digits, we append the
key readings from the actual PinnReading — deterministic code, real telemetry,
never invented numbers. The jury's independent digit-check stays in place as
the safety net for any other path.
"""
from __future__ import annotations

import inspect
from typing import Optional

from pydantic import BaseModel

from .crusoe_client import LLMClient
from .prompts import ADVISORY_SYSTEM, advisory_user, reading_digest
from .schemas import Advisory, AdvisoryStatus, PinnReading, RiskLabel, TriageResult


class AdvisoryDraft(BaseModel):
    """What the reasoning model must return. Every field required on purpose."""

    title: str
    message: str            # plain-language, must cite values
    justification: str      # the cited data behind it
    recommended_action: str
    severity: RiskLabel     # model's opinion; the pipeline keeps triage's word
    eur_impact: str
    safety_impact: str
    ttf_estimate: str


async def _gather_snippets(knowledge, reading: PinnReading, k: int = 2) -> list[str]:
    """Duck-typed knowledge.lookup(query) -> list[(snippet, citation)]."""
    if knowledge is None:
        return []
    queries = ["cost of downtime stoppage curing",
               f"{reading.department} safety human risk"]
    out: list[str] = []
    for q in queries:
        try:
            res = knowledge.lookup(q)
            if inspect.isawaitable(res):
                res = await res
            for snippet, citation in list(res)[:1]:
                cit = citation if str(citation).startswith("[") else f"[{citation}]"
                out.append(f"{cit} {str(snippet)[:280]}")
        except Exception as e:  # noqa: BLE001 — knowledge is optional, never fatal
            print(f"[line_llm] knowledge lookup failed ({type(e).__name__}): {e}")
        if len(out) >= k:
            break
    return out[:k]


async def draft_advisory(
    client: LLMClient,
    reading: PinnReading,
    triage: TriageResult,
    knowledge=None,
) -> AdvisoryDraft:
    """One reasoning-model call -> validated AdvisoryDraft (mock-safe)."""
    snippets = await _gather_snippets(knowledge, reading)
    messages = [
        {"role": "system", "content": ADVISORY_SYSTEM},
        {"role": "user", "content": advisory_user(
            reading_digest(reading), triage.risk.value, triage.causal_context, snippets)},
    ]
    return await client.complete_json(
        role="reasoning", messages=messages, schema=AdvisoryDraft,
        max_tokens=900, temperature=0.2, hint="advisory",
    )


def fallback_draft(reading: PinnReading, triage: TriageResult) -> AdvisoryDraft:
    """Deterministic draft if even the client (with its own mock fallback) died.

    Built only from the causal matrix + measured values — fully grounded.
    """
    p = reading.pinn
    causal = triage.causal_context or {}
    rul = "unknown" if p.rul_cycles is None else f"~{p.rul_cycles:.0f} cycles"
    return AdvisoryDraft(
        title=f"{causal.get('tag', 'Degradation')} signature on {reading.machine_id}",
        message=(f"{reading.machine_id} ({reading.department}) reads health "
                 f"{p.health_index * 100:.0f}%, residual {p.residual:.2f}. "
                 f"{causal.get('probable_cause', 'Cause not classified.')}"),
        justification=f"Automated fallback (LLM unavailable). Telemetry: {reading_digest(reading)}",
        recommended_action=causal.get(
            "remedy", "Keep the machine under dense polling and notify maintenance."),
        severity=triage.risk,
        eur_impact="A curing-stage stop costs ~EUR 20-30k per idle hour [site_dossier p.5].",
        safety_impact="Treat as a high-severity stage until inspected [site_dossier p.4].",
        ttf_estimate=rul,
    )


def build_advisory(
    draft: AdvisoryDraft,
    reading: PinnReading,
    severity: RiskLabel,
    debate=None,
    jury=None,
) -> Advisory:
    """Map AdvisoryDraft -> shared Advisory. Severity comes from triage."""
    p = reading.pinn
    message = draft.message.strip()
    if not any(ch.isdigit() for ch in message):  # grounding guard (see module doc)
        message += (f" Key readings: health {p.health_index * 100:.0f}%, "
                    f"residual {p.residual:.2f}.")
    justification = draft.justification.strip()
    if not any(ch.isdigit() for ch in justification):
        justification += f"\nMeasured data: {reading_digest(reading)}"
    return Advisory(
        machine_id=reading.machine_id,
        department=reading.department,
        epoch=reading.epoch,
        severity=severity,
        title=draft.title.strip()[:120],
        message=message,
        justification=justification,
        recommended_action=draft.recommended_action.strip(),
        eur_impact=draft.eur_impact.strip(),
        safety_impact=draft.safety_impact.strip(),
        ttf_estimate=draft.ttf_estimate.strip(),
        debate=debate,
        jury=jury,
        status=AdvisoryStatus.PENDING,
    )
