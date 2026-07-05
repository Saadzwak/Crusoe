"""Operator-app flow — notification routing, intervention lifecycle, VLM watch.

Spec: docs/superpowers/specs/2026-07-04-operator-app-design.md. This module is
the backend of static/operator.html and the future 3D map:

- OPERATORS / ASSIGNMENTS route every machine alert to ONE operator persona.
- InterventionManager owns the per-machine repair lifecycle
  (notified → acknowledged → claimed_done → verified) and publishes every
  transition on the existing SSE hub as `intervention`, plus `notify` toasts
  ("alert" = routed to the assigned operator, "broadcast" = everyone).
- A simulated VLM watcher timestamps what it "sees" (activity after
  take-charge, completion after the operator claims done) so the UI can show
  the claimed-vs-verified timeline. `_observe()` is the seam where the real
  camera + omni-model call will go.
- Optional real Teams card: set TEAMS_WEBHOOK_URL (fire-and-forget, the demo
  never depends on it).
- quick chat: fast-lane answer (role="fast", no tool loop) with citations and
  provenance built programmatically from the context actually injected.

No agent-to-agent mesh: everything reads/writes the shared store (architecture
rule) and talks to the UI through the hub `publish` callable.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, AsyncIterator, Callable, Optional

from .config import settings
from .operator import OperatorAgent
from .operator_tools import MACHINE_SPECS
from .schemas import (Advisory, AdvisoryStatus, Intervention,
                      InterventionStatus, RiskLabel, VlmObservation)

Publish = Callable[[str, dict], None]

# ================================================================= operators
OPERATORS: dict[str, dict[str, str]] = {
    "eric": {"id": "eric", "name": "Eric Bjarstal", "initials": "EB",
             "area": "Curing & Calendering"},
    "lina": {"id": "lina", "name": "Lina Moreau", "initials": "LM",
             "area": "Mixing"},
}
ASSIGNMENTS: dict[str, str] = {"RC-07": "eric", "CL-03": "eric", "MX-02": "lina"}
_DEFAULT_OPERATOR = "eric"


def assigned_operator(machine_id: str) -> dict[str, str]:
    """Operator persona in charge of a machine (graceful fallback)."""
    key = ASSIGNMENTS.get(str(machine_id).upper().strip(), _DEFAULT_OPERATOR)
    return OPERATORS[key]


def _latest_pending_advisory(store: Any, machine_id: str) -> Optional[Advisory]:
    for a in store.get_advisories(status="pending", limit=50):
        if a.machine_id == machine_id:
            return a
    return None


# ============================================================= notifications
def _notification(kind: str, machine_id: str, *,
                  to_operator: Optional[dict] = None, advisory_id: str = "",
                  severity: str = "", title: str = "", body: str = "") -> dict:
    return {
        "id": uuid.uuid4().hex[:12],
        "kind": kind,  # "alert" (routed) | "broadcast" (everyone)
        "to_operator": to_operator,  # {"id","name"} or None for broadcast
        "machine_id": machine_id,
        "advisory_id": advisory_id,
        "severity": severity,
        "title": title,
        "body": body,
        "at": time.time(),
    }


_SEV_COLOR = {"CRITICAL": "F85149", "HIGH": "F0883E",
              "WATCH": "D29922", "CLEAR": "3FB950"}


def teams_card(n: dict) -> dict:
    """MessageCard payload for a Teams incoming webhook (pure builder)."""
    mid = n.get("machine_id", "?")
    sev = (n.get("severity") or "INFO").upper()
    facts = [{"name": "Machine", "value": mid},
             {"name": "Severity", "value": sev}]
    if n.get("to_operator"):
        facts.append({"name": "Assigned to", "value": n["to_operator"]["name"]})
    if n.get("advisory_id"):
        facts.append({"name": "Advisory", "value": f"#{n['advisory_id']}"})
    return {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "themeColor": _SEV_COLOR.get(sev, "6264A7"),
        "summary": f"PRAETOR {sev} — {mid}",
        "sections": [{
            "activityTitle": f"**PRAETOR — {n.get('title') or 'plant notification'}**",
            "activitySubtitle": "Michelin Roanne · C3M line",
            "facts": facts,
            "text": n.get("body", ""),
        }],
        "potentialAction": [{
            "@type": "OpenUri",
            "name": "Open operator console",
            "targets": [{"os": "default",
                         "uri": f"{settings.public_app_url}#machine={mid}"}],
        }],
    }


# Responsible-person routing (concept, filled with DEMO personas — see
# ASSIGNMENTS below; NO real contacts hardcoded). In production each machine's
# responsible operator would carry their own destination — a per-person Teams
# chat/channel webhook, an @mention id, or an email — resolved here instead of
# posting every alert to one shared channel. MVP: one TEAMS_WEBHOOK_URL, and we
# name the assigned operator inside the card so the routing intent is visible.
def workflows_message(n: dict) -> dict:
    """Payload for a Microsoft Teams **Workflows** Incoming Webhook.

    The old Office 365 *Connectors* webhook (the MessageCard in `teams_card`)
    is retired (Microsoft, end-2025). The current path is a Power Automate /
    Workflows flow ("Post to a channel when a webhook request is received"),
    which accepts an **Adaptive Card** wrapped in the message/attachments
    envelope below. Kept separate from teams_card so nothing legacy breaks.
    """
    mid = n.get("machine_id", "?")
    sev = (n.get("severity") or "INFO").upper()
    who = (n.get("to_operator") or {}).get("name") or "on-call operator"
    facts = [
        {"title": "Machine", "value": mid},
        {"title": "Severity", "value": sev},
        {"title": "Responsible", "value": who},
    ]
    if n.get("advisory_id"):
        facts.append({"title": "Advisory", "value": f"#{n['advisory_id']}"})
    body = [
        {"type": "TextBlock", "size": "Large", "weight": "Bolder",
         "color": "Attention" if sev in ("HIGH", "CRITICAL") else "Default",
         "text": f"PRAETOR {sev} — {mid}"},
        {"type": "TextBlock", "spacing": "None", "isSubtle": True,
         "text": "Michelin Roanne · C3M curing line"},
        {"type": "TextBlock", "wrap": True, "weight": "Bolder",
         "text": n.get("title") or "Plant notification"},
        {"type": "TextBlock", "wrap": True, "text": n.get("body", "")},
        {"type": "FactSet", "facts": facts},
    ]
    card = {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.4",
        "body": body,
        "actions": [{
            "type": "Action.OpenUrl", "title": "Open operator console",
            "url": f"{settings.public_app_url}#machine={mid}",
        }],
    }
    return {"type": "message",
            "attachments": [{
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": card,
            }]}


# ================================================================== manager
class InterventionManager:
    """One active intervention per machine; every transition is persisted to
    the shared store and published on the hub (`intervention` events — the 3D
    map renders its hologram from these)."""

    def __init__(self, store: Any, publish: Publish,
                 operator_agent: Optional[OperatorAgent] = None) -> None:
        self.store = store
        self.publish = publish
        self._operator = operator_agent or OperatorAgent()
        self._last_alert: dict[str, float] = {}  # machine_id → last toast time
        self._tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------ plumbing
    def _spawn(self, coro) -> None:
        """Schedule background work if a loop is running; else drop quietly
        (sync test contexts) — the state machine never depends on it."""
        try:
            task = asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            coro.close()
            return
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _emit(self, iv: Intervention) -> None:
        self.store.save_intervention(iv)
        self.publish("intervention", iv.model_dump())
        self.store.log_event("intervention", {
            "id": iv.id, "machine_id": iv.machine_id, "status": iv.status.value})

    def _notify(self, n: dict) -> None:
        self.publish("notify", n)
        self.store.log_event("notify", {
            "kind": n["kind"], "machine_id": n["machine_id"],
            "severity": n.get("severity", ""), "title": n.get("title", ""),
            "to": (n.get("to_operator") or {}).get("name", "")})
        if settings.teams_webhook_url:
            self._spawn(self._post_teams(n))

    async def _post_teams(self, n: dict) -> None:
        """Optional real Teams alert — never raises, never blocks the demo.

        Uses the current **Workflows** Adaptive-Card format (the legacy
        Connectors MessageCard is retired). If someone still runs an old
        connector URL, set TEAMS_WEBHOOK_LEGACY=1 to fall back to teams_card.
        """
        try:
            import os

            import httpx  # transitive dep of openai — always present
            payload = (teams_card(n) if os.environ.get("TEAMS_WEBHOOK_LEGACY") == "1"
                       else workflows_message(n))
            async with httpx.AsyncClient(timeout=6.0) as cli:
                r = await cli.post(settings.teams_webhook_url, json=payload)
                if r.status_code >= 300:
                    print(f"[operator_flow] Teams webhook returned {r.status_code}: "
                          f"{r.text[:160]}")
                else:
                    print(f"[operator_flow] Teams alert delivered for "
                          f"{n.get('machine_id')} ({n.get('severity')})")
        except Exception as e:  # noqa: BLE001
            print(f"[operator_flow] Teams webhook failed ({type(e).__name__}: {e}); "
                  "in-app notification already delivered")

    # ------------------------------------------------------------ lifecycle
    def active(self, machine_id: str) -> Optional[Intervention]:
        return self.store.get_active_intervention(str(machine_id).upper().strip())

    def on_advisory(self, adv: Advisory) -> Optional[Intervention]:
        """Loop hook: route a fresh advisory into the intervention flow."""
        mid = adv.machine_id
        active = self.store.get_active_intervention(mid)
        if active is not None:  # machine already in flow — link silently
            if active.advisory_id != adv.id:
                active.advisory_id = adv.id
                self._emit(active)
            return active
        if adv.severity not in (RiskLabel.HIGH, RiskLabel.CRITICAL):
            return None  # WATCH/CLEAR never page an operator
        now = time.time()
        op = assigned_operator(mid)
        iv = Intervention(machine_id=mid, advisory_id=adv.id,
                          status=InterventionStatus.NOTIFIED,
                          notified_operator_id=op["id"], notified_at=now)
        self._emit(iv)
        if now - self._last_alert.get(mid, 0.0) >= settings.notify_cooldown_s:
            self._last_alert[mid] = now
            self._notify(_notification(
                "alert", mid,
                to_operator={"id": op["id"], "name": op["name"]},
                advisory_id=adv.id, severity=adv.severity.value,
                title=adv.title, body=adv.message))
        return iv

    async def take_charge(self, machine_id: str, operator_id: str) -> Intervention:
        mid = str(machine_id).upper().strip()
        op = OPERATORS.get(str(operator_id).lower().strip())
        if op is None:
            raise ValueError(f"unknown operator '{operator_id}'")
        iv = self.store.get_active_intervention(mid)
        if iv is None:  # proactive take-charge, no alert needed first
            iv = Intervention(machine_id=mid, status=InterventionStatus.NOTIFIED,
                              notified_operator_id=op["id"],
                              notified_at=time.time())
            adv = _latest_pending_advisory(self.store, mid)
            if adv is not None:
                iv.advisory_id = adv.id
        if iv.status == InterventionStatus.CLAIMED_DONE:
            raise ValueError(f"{mid} repair already claimed done by "
                             f"{iv.operator_name} — awaiting VLM verification")
        if (iv.status == InterventionStatus.ACKNOWLEDGED
                and iv.operator_id != op["id"]):
            raise ValueError(f"{iv.operator_name} already took charge of {mid}")
        iv.status = InterventionStatus.ACKNOWLEDGED
        iv.operator_id, iv.operator_name = op["id"], op["name"]
        iv.acknowledged_at = time.time()
        self._emit(iv)
        self._accept_linked_advisory(iv, op)
        adv_sev = ""
        if iv.advisory_id:
            adv = self.store.get_advisory(iv.advisory_id)
            adv_sev = adv.severity.value if adv else ""
        self._notify(_notification(
            "broadcast", mid, advisory_id=iv.advisory_id, severity=adv_sev,
            title=f"{mid} — intervention in progress",
            body=f"{op['name']} took charge of {mid}. "
                 "The VLM watcher is monitoring the work zone."))
        self._spawn(self._vlm_activity_watch(iv.id))
        return iv

    async def repair_done(self, machine_id: str, operator_id: str) -> Intervention:
        mid = str(machine_id).upper().strip()
        op = OPERATORS.get(str(operator_id).lower().strip())
        if op is None:
            raise ValueError(f"unknown operator '{operator_id}'")
        iv = self.store.get_active_intervention(mid)
        if iv is None:
            raise ValueError(f"no active intervention on {mid}")
        if iv.status != InterventionStatus.ACKNOWLEDGED:
            raise ValueError(f"cannot mark {mid} done from status {iv.status.value}")
        if iv.operator_id != op["id"]:
            raise ValueError(f"only {iv.operator_name} (who took charge) can "
                             f"close the intervention on {mid}")
        iv.status = InterventionStatus.CLAIMED_DONE
        iv.claimed_done_at = time.time()
        self._emit(iv)
        self._spawn(self._vlm_verify_watch(iv.id))
        return iv

    def _accept_linked_advisory(self, iv: Intervention, op: dict) -> None:
        """Taking charge = accepting the advisory's call to act. Reuses the
        existing override path so the learning loop keeps its history."""
        if not iv.advisory_id:
            return
        adv = self.store.get_advisory(iv.advisory_id)
        if adv is None or adv.status != AdvisoryStatus.PENDING:
            return
        updated = self._operator.handle_override(
            adv.id, "accepted", f"taken in charge by {op['name']} (operator app)",
            self.store)
        if updated is not None:
            self.publish("override", updated.model_dump())

    # ----------------------------------------------------------- VLM watch
    def _observe(self, machine_id: str, phase: str) -> VlmObservation:
        """SEAM for the physical layer: replace this body with a camera frame
        + role="omni" call when the shop-floor camera is integrated. Until
        then the watcher is scripted and clearly labeled as simulated."""
        if phase == "activity":
            return VlmObservation(
                kind="activity_detected", confidence=0.91,
                detail=f"[simulated VLM] Operator presence confirmed in the "
                       f"{machine_id} work zone; movements consistent with "
                       "the recommended action.")
        return VlmObservation(
            kind="repair_verified", confidence=0.87,
            detail=f"[simulated VLM] {machine_id} work zone clear, guards "
                   "closed; machine signature back inside its envelope.")

    async def _vlm_activity_watch(self, intervention_id: str) -> None:
        await asyncio.sleep(max(0.0, settings.vlm_activity_delay_s))
        iv = self.store.get_intervention(intervention_id)
        if (iv is None or iv.status != InterventionStatus.ACKNOWLEDGED
                or iv.vlm_activity_at is not None):
            return  # operator was faster than the camera — timeline shows the gap
        obs = self._observe(iv.machine_id, "activity")
        iv.vlm_activity_at = obs.at
        iv.vlm_observations.append(obs)
        self._emit(iv)

    async def _vlm_verify_watch(self, intervention_id: str) -> None:
        await asyncio.sleep(max(0.0, settings.vlm_verify_delay_s))
        iv = self.store.get_intervention(intervention_id)
        if iv is None or iv.status != InterventionStatus.CLAIMED_DONE:
            return
        obs = self._observe(iv.machine_id, "verify")
        iv.vlm_verified_at = obs.at
        iv.vlm_observations.append(obs)
        iv.status = InterventionStatus.VERIFIED
        self._emit(iv)
        delta = ""
        if iv.claimed_done_at:
            delta = f" {obs.at - iv.claimed_done_at:.0f}s after the claim"
        self._notify(_notification(
            "broadcast", iv.machine_id, advisory_id=iv.advisory_id,
            title=f"{iv.machine_id} — repair verified",
            body=f"VLM verified the repair on {iv.machine_id}{delta} "
                 f"(confidence {obs.confidence:.2f}). Claimed by "
                 f"{iv.operator_name or 'operator'}."))


# ============================================================ factory state
def build_factory_state(store: Any, manager: Optional[InterventionManager],
                        *, mode: str, epoch: int) -> dict:
    """Initial-state contract for the operator app AND the 3D map: everything
    per machine in one call (live updates then come from /api/stream)."""
    snap: dict = {}
    try:
        snap = store.department_snapshot() or {}
    except Exception:  # noqa: BLE001 — empty store is fine
        pass
    risk_by_machine = {v.get("machine_id"): v.get("risk", "UNKNOWN")
                       for v in snap.values()}
    pending: dict[str, Advisory] = {}
    for a in store.get_advisories(status="pending", limit=100):
        pending.setdefault(a.machine_id, a)  # newest-first → keep newest
    machines = []
    for mid, spec in MACHINE_SPECS.items():
        rows = store.get_sensor_history(mid, limit=1)
        iv = manager.active(mid) if manager is not None else None
        adv = pending.get(mid)
        machines.append({
            "machine_id": mid,
            "department": spec.get("department", "?"),
            "type": spec.get("type", ""),
            "role": spec.get("role", ""),
            "assigned_operator": assigned_operator(mid),
            "risk": risk_by_machine.get(mid, "UNKNOWN"),
            "last_reading": rows[-1] if rows else None,
            "active_advisory": adv.model_dump() if adv else None,
            "intervention": iv.model_dump() if iv else None,
            # verified limits straight from limits.py via MACHINE_SPECS — the
            # UI colors out-of-range values without hardcoding a copy.
            "signals_spec": spec.get("signals", {}),
        })
    return {"mode": mode, "epoch": epoch, "machines": machines,
            "at": time.time()}


# ================================================================ quick chat
_QUICK_SYSTEM = (
    "You are PRAETOR's line assistant on the Michelin Roanne curing line, "
    "talking to a shop-floor operator who needs a fast, clear read.\n"
    "Answer ONLY from the CONTEXT block. Never invent a number.\n\n"
    "FORMAT (follow exactly):\n"
    "- First line: the bottom line — is the machine OK or not, and how urgent, "
    "in one short sentence.\n"
    "- Then 2 to 4 bullet points starting with '- ', each one fact with its "
    "number and what it means in plain words.\n"
    "- If a value is dangerously past its limit (temperature, pressure or "
    "vibration well over the max), say so bluntly and put it FIRST.\n"
    "- Close with a one-line recommendation offered as a choice — you advise, "
    "the operator decides. Never give an order.\n\n"
    "RULES:\n"
    "- Use everyday time like 'just now' or 'a few minutes ago' exactly as the "
    "context gives it. NEVER say 'epoch', cycle numbers, ids or codes.\n"
    "- No bracketed citations, no 'Data Provenance', no 'Gaps' line, no "
    "jargon. Plain language a non-engineer reads in one pass.\n"
    "- Keep it under 90 words total."
)


def _ago(ts: Any, now: float) -> str:
    """Wall-clock reading age as plain operator language (never epochs)."""
    try:
        d = max(0.0, now - float(ts))
    except (TypeError, ValueError):
        return "moments ago"
    if d < 8:
        return "just now"
    if d < 90:
        return f"{int(round(d))} seconds ago"
    m = d / 60.0
    if m < 90:
        return f"{int(round(m))} minutes ago"
    return f"{int(round(m / 60.0))} hours ago"


def _quick_context(machine_id: Optional[str], store: Any,
                   manager: Optional[InterventionManager]
                   ) -> tuple[str, str, list[str], list[dict]]:
    """Evidence block + programmatic citations/provenance for the fast lane."""
    mid = str(machine_id or "RC-07").upper().strip()
    if mid not in MACHINE_SPECS:
        mid = "RC-07"
    blocks: list[str] = []
    citations: list[str] = []
    provenance: list[dict] = []
    now = time.time()

    rows = store.get_sensor_history(mid, limit=8)
    if rows:
        latest = rows[-1]
        sig = ", ".join(f"{k}={v}" for k, v in latest.get("signals", {}).items())
        p = latest.get("pinn", {}) or {}
        # latest reading in plain relative time (no epoch anywhere)
        head = (f"Latest reading ({_ago(latest.get('at'), now)}): {sig} | "
                f"health={p.get('health_index')} rul_cycles={p.get('rul_cycles')}")
        # window trend: first vs last of the stored window, per signal
        trend_bits = []
        first = rows[0]
        for k, v_last in (latest.get("signals") or {}).items():
            v_first = (first.get("signals") or {}).get(k)
            try:
                if v_first is not None and abs(float(v_last) - float(v_first)) > 1e-9:
                    arrow = "rising" if float(v_last) > float(v_first) else "falling"
                    trend_bits.append(f"{k} {arrow} {float(v_first):g}->{float(v_last):g}")
            except (TypeError, ValueError):
                continue
        span = _ago(first.get("at"), now)
        trend = (f"\nOver the last few readings (since {span}): "
                 + "; ".join(trend_bits)) if trend_bits else ""
        blocks.append(f"[sensor_history {mid}]\n{head}{trend}")
        citations.append(f"sensor_history {mid}")
        provenance.append({"source": f"sensor_history {mid}",
                           "detail": f"{len(rows)} stored readings, latest "
                                     f"{_ago(latest.get('at'), now)}"})

    adv = _latest_pending_advisory(store, mid)
    if adv is not None:
        blocks.append(f"[advisory {adv.id}] {adv.severity.value} — {adv.title}. "
                      f"{adv.message} Evidence: {adv.justification} "
                      f"Proposed: {adv.recommended_action}")
        citations.append(f"advisory {adv.id}")
        provenance.append({"source": f"advisory {adv.id}", "detail": adv.title})

    spec = MACHINE_SPECS.get(mid, {})
    lims = []
    for name, s in (spec.get("signals") or {}).items():
        rng = s.get("operating")
        if rng and len(rng) == 2:
            lim = f"{name} {rng[0]}–{rng[1]} {s.get('unit', '')}".rstrip()
            if s.get("trip") is not None:
                lim += f" (trip {s['trip']})"
            lims.append(lim)
    if lims:
        blocks.append(f"[limits {mid}] verified operating limits: "
                      + "; ".join(lims))
        citations.append(f"limits {mid}")
        provenance.append({"source": f"limits {mid}",
                           "detail": "verified limits (backend/agent/limits.py, "
                                     "with source provenance)"})

    iv = manager.active(mid) if manager is not None else None
    if iv is not None:
        who = f", {iv.operator_name} on site" if iv.operator_name else ""
        blocks.append(f"[intervention {iv.id}] status {iv.status.value}{who}")
        citations.append(f"intervention {iv.id}")
        provenance.append({"source": f"intervention {iv.id}",
                           "detail": f"status {iv.status.value}"})

    context = "\n\n".join(blocks) if blocks else \
        "(no stored data for this machine yet)"
    return mid, context, citations, provenance


async def stream_quick(question: str, machine_id: Optional[str], store: Any,
                       client: Any,
                       manager: Optional[InterventionManager] = None
                       ) -> AsyncIterator[dict]:
    """Fast-lane chat: role='fast', no tool loop — SSE-friendly events
    ({'type':'token'|'turn'}) matching the legacy stream contract."""
    mid, context, citations, provenance = _quick_context(machine_id, store,
                                                         manager)
    messages = [
        {"role": "system", "content": _QUICK_SYSTEM},
        {"role": "user", "content": f"=== CONTEXT ({mid}) ===\n{context}\n\n"
                                    f"=== OPERATOR QUESTION ===\n{question}"},
    ]
    parts: list[str] = []
    async for tok in client.stream(role="fast", messages=messages,
                                   max_tokens=260, temperature=0.2,
                                   hint="operator"):
        parts.append(tok)
        yield {"type": "token", "text": tok}
    content = "".join(parts)
    cites = [f"[{c}]" for c in citations]
    for extra in OperatorAgent._citations(content):  # model's own brackets
        if extra not in cites:
            cites.append(extra)
    turn = {"role": "agent", "content": content, "citations": cites,
            "at": time.time(), "provenance": provenance, "mode": "quick",
            "machine_id": mid}
    yield {"type": "turn", "turn": turn}


async def quick_answer(question: str, machine_id: Optional[str], store: Any,
                       client: Any,
                       manager: Optional[InterventionManager] = None
                       ) -> tuple[dict, list[dict]]:
    """Non-streaming wrapper around stream_quick (JSON API + tests)."""
    turn: Optional[dict] = None
    async for ev in stream_quick(question, machine_id, store, client, manager):
        if ev.get("type") == "turn":
            turn = ev["turn"]
    assert turn is not None  # stream_quick always ends with a turn
    return turn, turn["provenance"]


__all__ = ["OPERATORS", "ASSIGNMENTS", "assigned_operator", "teams_card",
           "workflows_message", "_notification", "InterventionManager",
           "build_factory_state", "stream_quick", "quick_answer"]
