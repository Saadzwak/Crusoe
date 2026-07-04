"""DebateRoom — Advocate (reasoning) proposes, Skeptic (fast) attacks, <=2 rounds.

Protocol: the skeptic must answer 'NO-OBJECTION: ...' or 'OBJECT: <reason>'.
Verdicts:
  approved  — skeptic accepts in round 1 (final = advocate's PROPOSAL line)
  revised   — skeptic accepts in round 2 after a rebuttal/revision
  escalated — skeptic still objects after max rounds -> final proposal becomes
              the safe fallback (hold action, notify supervisor, dense polling)

Offline determinism: the mock echoes prompts without protocol tokens. An
unparseable skeptic reply defaults to 'objection' in round 1 (skeptics are
skeptical) and 'no objection' in round 2 — so mock debates run the full two
rounds and settle as 'revised', never silently rubber-stamping round 1.
"""
from __future__ import annotations

import re
from typing import Optional

from .crusoe_client import LLMClient
from .line_llm import AdvisoryDraft
from .prompts import (ADVOCATE_SYSTEM, SAFE_FALLBACK_ACTION, SKEPTIC_SYSTEM,
                      advocate_open_user, advocate_rebut_user, format_overrides,
                      reading_digest, skeptic_user)
from .schemas import DebateResult, DebateTurn, PinnReading

MAX_ROUNDS = 2
_PROPOSAL_RE = re.compile(r"PROPOSAL\s*:\s*(.+?)(?:\bWHY\s*:|\n\n|$)", re.DOTALL | re.IGNORECASE)


def _extract_proposal(advocate_text: str, fallback: str) -> str:
    m = _PROPOSAL_RE.search(advocate_text or "")
    if m:
        prop = " ".join(m.group(1).split()).strip()
        if prop:
            return prop[:400]
    return fallback


def _skeptic_objects(text: str, rnd: int) -> bool:
    """True if the skeptic objects. Token parse first, round-based default."""
    up = (text or "").upper()
    if "NO-OBJECTION" in up or "NO OBJECTION" in up:
        return False
    if "OBJECT" in up:
        return True
    return rnd == 1  # unparseable (mock): challenge once, then concede


async def run_debate(
    client: LLMClient,
    draft: AdvisoryDraft,
    reading: PinnReading,
    overrides: Optional[list] = None,
    max_rounds: int = MAX_ROUNDS,
) -> DebateResult:
    digest = reading_digest(reading)
    ov_text = format_overrides(overrides)
    machine, dept = reading.machine_id, reading.department
    turns: list[DebateTurn] = []
    skeptic_text = ""

    for rnd in range(1, max_rounds + 1):
        if rnd == 1:
            adv_user = advocate_open_user(machine, dept, draft.severity.value,
                                          draft.recommended_action, digest, ov_text)
        else:
            adv_user = advocate_rebut_user(machine, skeptic_text)
        adv_text = await client.complete(
            role="reasoning", hint="advocate", max_tokens=350, temperature=0.3,
            messages=[{"role": "system", "content": ADVOCATE_SYSTEM},
                      {"role": "user", "content": adv_user}],
        )
        turns.append(DebateTurn(role="advocate", round=rnd, content=adv_text.strip()))

        proposal_now = _extract_proposal(adv_text, draft.recommended_action)
        skeptic_text = await client.complete(
            role="fast", hint="skeptic", max_tokens=120, temperature=0.2,
            messages=[{"role": "system", "content": SKEPTIC_SYSTEM},
                      {"role": "user", "content": skeptic_user(
                          rnd, machine, dept, proposal_now, digest, ov_text)}],
        )
        turns.append(DebateTurn(role="skeptic", round=rnd, content=skeptic_text.strip()))

        if not _skeptic_objects(skeptic_text, rnd):
            return DebateResult(
                verdict="approved" if rnd == 1 else "revised",
                final_proposal=proposal_now,
                turns=turns,
                rounds_used=rnd,
            )

    return DebateResult(
        verdict="escalated",
        final_proposal=SAFE_FALLBACK_ACTION,
        turns=turns,
        rounds_used=max_rounds,
    )
