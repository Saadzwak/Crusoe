"""Jury — LLM-as-judge gate on every advisory before it reaches the operator.

Scores grounding / safety / actionability / human_in_charge (0-5 each),
overall, passed vs settings.jury_pass_threshold, one-sentence critique.

Deterministic post-checks (never trust the judge blindly):
- passed requires BOTH the model's own verdict AND overall >= threshold;
- if the advisory MESSAGE contains zero digits (no cited values), grounding
  is capped at 2 and passed forced False, whatever the model said.
"""
from __future__ import annotations

from .config import settings
from .crusoe_client import LLMClient
from .prompts import JURY_SYSTEM, jury_user
from .schemas import Advisory, JuryScore


def _advisory_text(adv: Advisory) -> str:
    return (
        f"title: {adv.title}\nmessage: {adv.message}\n"
        f"justification: {adv.justification}\n"
        f"recommended_action: {adv.recommended_action}\n"
        f"eur_impact: {adv.eur_impact}\nsafety_impact: {adv.safety_impact}\n"
        f"ttf_estimate: {adv.ttf_estimate}"
    )


async def judge_advisory(client: LLMClient, advisory: Advisory, digest: str) -> JuryScore:
    threshold = settings.jury_pass_threshold
    score = await client.complete_json(
        role="reasoning",
        messages=[{"role": "system", "content": JURY_SYSTEM},
                  {"role": "user", "content": jury_user(
                      _advisory_text(advisory), digest, threshold)}],
        schema=JuryScore, max_tokens=500, temperature=0.1, hint="jury",
    )

    passed = bool(score.passed) and score.overall >= threshold
    grounding = score.grounding
    critique = score.critique.strip()
    if not critique:  # mock leaves optional fields empty — keep the card readable
        critique = f"Overall {score.overall:.1f}/5 against threshold {threshold:.1f}."

    if not any(ch.isdigit() for ch in advisory.message):
        grounding = min(grounding, 2.0)
        passed = False
        critique = (critique + " [deterministic check] The operator message "
                    "cites no measured value.").strip()

    return score.model_copy(update={
        "grounding": grounding, "passed": passed, "critique": critique,
    })
