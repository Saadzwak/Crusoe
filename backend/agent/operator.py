"""Operator Q&A agent + override handler (Builder B).

gbrain pattern: gather tool context FIRST (sensor history, advisories, past
overrides, knowledge lookup), build ONE prompt with clearly labeled evidence
blocks, then synthesize an answer that
  - cites every number with its bracketed source label,
  - NEVER commands — it advises, the operator decides,
  - ends with a one-line "Gaps:" note naming what the data does NOT cover.

Citations are collected by scanning the answer for [ ... ] markers.
"""
from __future__ import annotations

import re
from typing import Any, AsyncIterator, Optional

from .schemas import (Advisory, AdvisoryStatus, OperatorOverride, OperatorTurn,
                      RiskLabel)

_MACHINES = {"RC-07": "Curing", "CL-03": "Calendering", "MX-02": "Mixing"}
_CITE_RE = re.compile(r"\[[^\[\]\n]{2,80}\]")

_SYSTEM = (
    "You are PRAETOR, the operator-assist agent on the Michelin Roanne UHP "
    "tyre line (C3M process; machines: RC-07 curing press — the bottleneck, "
    "CL-03 calender, MX-02 Banbury mixer). Answer the operator's question "
    "using ONLY the labeled evidence blocks provided.\n"
    "Rules:\n"
    "1. Cite every number or factual claim with its bracketed source label, "
    "e.g. [sensor_history RC-07], [site_dossier p.4], [advisory <id>].\n"
    "2. You advise — you never command. The operator stays in charge; frame "
    "recommendations as options with trade-offs.\n"
    "3. If the evidence does not cover something, say so explicitly.\n"
    "4. End with exactly one line starting 'Gaps:' naming what the provided "
    "data does NOT cover for this question.\n"
    "Keep it under 180 words, plain language for a non-technical operator."
)


def _fmt_history(rows: list[dict]) -> str:
    lines = []
    for r in rows:
        sig = ", ".join(f"{k}={v}" for k, v in r.get("signals", {}).items())
        p = r.get("pinn", {})
        lines.append(
            f"epoch {r.get('epoch')}: {sig} | health={p.get('health_index')} "
            f"rul={p.get('rul_cycles')} residual={p.get('residual')} | {r.get('note', '')}"
        )
    return "\n".join(lines) if lines else "(no readings yet)"


def _fmt_advisories(advs: list[Advisory]) -> str:
    if not advs:
        return "(no advisories yet)"
    return "\n".join(
        f"[advisory {a.id}] {a.machine_id} {a.severity.value} '{a.title}' "
        f"status={a.status.value} action: {a.recommended_action[:120]}"
        for a in advs
    )


def _fmt_overrides(ovs: list[OperatorOverride]) -> str:
    if not ovs:
        return "(no operator decisions recorded yet)"
    return "\n".join(
        f"[override {o.advisory_id}] {o.decision}"
        + (f" — reason: {o.reason}" if o.reason else "")
        for o in ovs
    )


class OperatorAgent:
    """Stateless — store/knowledge/client are passed per call (contract)."""

    # ----------------------------------------------------------- gathering
    def _mentioned_machines(self, question: str) -> list[str]:
        q = question.upper()
        found = [m for m in _MACHINES if m in q]
        if not found:  # department names count too
            for mid, dept in _MACHINES.items():
                if dept.upper() in q:
                    found.append(mid)
        return found or ["RC-07"]  # curing press is the default subject

    def _gather(self, question: str, store: Any, knowledge: Any) -> tuple[str, list[str]]:
        machines = self._mentioned_machines(question)
        blocks: list[str] = []
        for mid in machines:
            rows = store.get_sensor_history(mid, limit=8)
            blocks.append(f"=== SENSOR HISTORY [sensor_history {mid}] "
                          f"({_MACHINES.get(mid, '?')}) ===\n{_fmt_history(rows)}")
        blocks.append("=== OPEN & RECENT ADVISORIES ===\n"
                      + _fmt_advisories(store.get_advisories(limit=5)))
        blocks.append("=== PAST OPERATOR DECISIONS ===\n"
                      + _fmt_overrides(store.get_overrides(limit=5)))
        if knowledge is not None:
            kn = knowledge.lookup(question, k=3)
            kn_txt = "\n".join(f"{cit} {text}" for text, cit in kn)
            blocks.append(f"=== SITE KNOWLEDGE ===\n{kn_txt}")
        evidence = "\n\n".join(blocks)
        return evidence, machines

    def _messages(self, question: str, evidence: str) -> list[dict]:
        user = (f"{evidence}\n\n=== OPERATOR QUESTION ===\n{question}\n\n"
                "Answer with citations and finish with the one-line 'Gaps:' note.")
        return [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user}]

    @staticmethod
    def _citations(answer: str) -> list[str]:
        seen: list[str] = []
        for m in _CITE_RE.findall(answer):
            if m not in seen:
                seen.append(m)
        return seen

    # ----------------------------------------------------------- answering
    async def answer(self, question: str, store: Any, knowledge: Any,
                     client: Any) -> OperatorTurn:
        evidence, _machines = self._gather(question, store, knowledge)
        text = await client.complete(
            role="reasoning", messages=self._messages(question, evidence),
            max_tokens=450, temperature=0.3, hint="operator",
        )
        return OperatorTurn(role="agent", content=text,
                            citations=self._citations(text))

    async def stream_answer(self, question: str, store: Any, knowledge: Any,
                            client: Any) -> AsyncIterator[dict]:
        """SSE-friendly: yields {'type':'token','text':...} chunks, then one
        final {'type':'turn','turn': OperatorTurn.model_dump()}."""
        evidence, _machines = self._gather(question, store, knowledge)
        parts: list[str] = []
        async for tok in client.stream(
            role="reasoning", messages=self._messages(question, evidence),
            max_tokens=450, temperature=0.3, hint="operator",
        ):
            parts.append(tok)
            yield {"type": "token", "text": tok}
        text = "".join(parts)
        turn = OperatorTurn(role="agent", content=text,
                            citations=self._citations(text))
        yield {"type": "turn", "turn": turn.model_dump()}

    # ----------------------------------------------------------- overrides
    def handle_override(self, advisory_id: str, decision: str, reason: str,
                        store: Any) -> Optional[Advisory]:
        """Update advisory status + persist the OperatorOverride (which feeds
        future prompts via PAST OPERATOR DECISIONS). Returns the updated
        advisory, or None if the id is unknown."""
        adv = store.get_advisory(advisory_id)
        if adv is None:
            return None
        decision = decision.strip().lower()
        if decision not in ("accepted", "overridden"):
            decision = "overridden"
        adv.status = (AdvisoryStatus.ACCEPTED if decision == "accepted"
                      else AdvisoryStatus.OVERRIDDEN)
        if decision == "overridden":
            adv.override_reason = reason or ""
        store.save_advisory(adv)
        ov = OperatorOverride(advisory_id=advisory_id, decision=decision,
                              reason=reason or "")
        store.save_override(ov)
        store.log_event("override", {"advisory_id": advisory_id,
                                     "decision": decision, "reason": reason})
        return adv


__all__ = ["OperatorAgent", "RiskLabel"]
