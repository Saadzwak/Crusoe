# Operator App — design spec (validated 2026-07-04)

Operator-facing UX for the PRAETOR digital twin: alert notification → machine
view → fast chatbot → take-charge → VLM-verified repair timeline. Approved by
Eric in session 2026-07-04 (options: page in this repo, simulated Teams toast
+ optional real webhook, English UI). The 3D factory map is built in a
SEPARATE conversation and will mount into this app later — §7 is its contract.

## 1. Scope

IN: operator web app at `/operator` (single self-contained HTML file, like the
existing console), notification routing per machine→operator assignment,
intervention state machine with VLM-verified timeline, quick-answer chat mode,
optional real Teams webhook, docs + mock-mode tests.

OUT (other conversations / later): the 3D map itself, real camera + live VLM
calls (simulated verifier behind a seam), auth (persona dropdown instead),
Gradium TTS.

## 2. Operator journey (demo script)

1. Loop emits advisory HIGH/CRITICAL on RC-07 → in-app Teams-style toast for
   the ASSIGNED operator only (Eric gets RC-07/CL-03; Lina gets MX-02);
   optional real Teams card if `TEAMS_WEBHOOK_URL` is set.
2. Click toast → machine view: severity, advisory (message, cited evidence,
   recommended action, € impact, TTF, jury badge), live signals vs verified
   limits, chatbot pre-scoped to the machine.
3. Chat answers FAST (quick mode ≈1–3 s); "Deep analysis" toggle runs the
   existing tool-calling agent with visible tool trace.
4. `Take charge` → intervention `acknowledged`, linked advisory auto-accepted,
   broadcast toast "RC-07 — Eric Bjarstal is on it", SSE `intervention` event
   (3D map renders its hologram from this; the app shows a 2D badge).
5. Simulated VLM detects activity (delay ~8 s), then after `Mark repair done`
   confirms completion (delay ~10 s) with confidence values.
6. Timeline block shows claimed-vs-verified with deltas:
   notified → taken charge → VLM activity (+Δ) → claimed done → VLM verified (+Δ).

## 3. State machine (per machine, one active intervention max)

`notified → acknowledged → claimed_done → verified`

- `on_advisory(adv)`: severity ≥ HIGH; if an active intervention exists, link
  the new advisory to it silently; else create `notified` + targeted notify
  (per-machine cooldown `NOTIFY_COOLDOWN_S`, default 90 s, so a degrading
  machine doesn't spam).
- `take_charge(machine, operator)`: any operator may take charge (also
  proactively, without a prior notification). Auto-accepts the linked pending
  advisory via the existing override path (feeds the learning loop).
- `repair_done(machine, operator)`: only the operator who took charge.
- VLM watcher (asyncio tasks): while `acknowledged`, after
  `VLM_ACTIVITY_DELAY_S` (8 s) emit observation `activity_detected`
  (confidence .91); while `claimed_done`, after `VLM_VERIFY_DELAY_S` (10 s)
  emit `repair_verified` (confidence .87) and close the intervention. If the
  operator claims done before the activity check fired, `vlm_activity_at`
  stays null — the timeline shows the gap honestly. Seam
  `_observe(machine_id, phase)` is where the real omni-model VLM call will go.

## 4. Data contracts (additive to `schemas.py`)

```python
class InterventionStatus(str, Enum):
    NOTIFIED = "notified"; ACKNOWLEDGED = "acknowledged"
    CLAIMED_DONE = "claimed_done"; VERIFIED = "verified"

class VlmObservation(BaseModel):
    kind: str          # "activity_detected" | "repair_verified"
    detail: str
    confidence: float
    at: float

class Intervention(BaseModel):
    id: str; machine_id: str; advisory_id: str = ""
    status: InterventionStatus
    operator_id: str = ""; operator_name: str = ""     # who took charge
    notified_operator_id: str = ""                     # who was alerted
    notified_at/acknowledged_at/claimed_done_at/vlm_activity_at/vlm_verified_at:
        Optional[float]
    vlm_observations: list[VlmObservation] = []
    created_at: float
```

Notification (SSE `notify` payload, also flight-recorded via `log_event`):
`{id, kind: "alert"|"broadcast", to_operator: {id,name}|null, machine_id,
advisory_id, severity, title, body, at}` — `alert` is routed (only the
assigned operator's UI toasts it), `broadcast` toasts for everyone.

## 5. Backend surface (additive)

- `backend/agent/operator_flow.py` — OPERATORS, ASSIGNMENTS
  (RC-07,CL-03→eric; MX-02→lina), `InterventionManager` (state machine + VLM
  watcher + notifications + optional Teams POST), `build_factory_state()`,
  `quick_chat` context builder + streamer.
- `StateStore` additions: `interventions` table (upsert by id) +
  `save_intervention / get_intervention / get_active_intervention /
  get_interventions`.
- Endpoints (main.py):
  - `GET /operator` → static/operator.html
  - `GET /api/operators` → `{operators, assignments}`
  - `GET /api/factory/state` → `{mode, epoch, machines: [{machine_id,
    department, assigned_operator, risk, last_reading, active_advisory,
    intervention}]}` (machines enumerated from MACHINE_SPECS)
  - `POST /api/machine/{id}/take_charge` `{operator_id}` → Intervention
  - `POST /api/machine/{id}/repair_done` `{operator_id}` → Intervention
  - `POST /api/chat` gains `mode: "quick"|"deep"` (default deep = unchanged)
    and `machine_id`. Quick = no tool loop: context (latest reading + active
    advisory + verified limits from `limits.py` via MACHINE_SPECS +
    intervention) → `role="fast"` streamed; citations + provenance built
    programmatically from the context actually injected.
- Loop hook: after publishing an advisory, call
  `S.interventions.on_advisory(adv)`.
- New SSE kinds on the EXISTING `/api/stream`: `notify`, `intervention`.
- Config/env: `TEAMS_WEBHOOK_URL` (empty = skip), `NOTIFY_COOLDOWN_S`,
  `VLM_ACTIVITY_DELAY_S`, `VLM_VERIFY_DELAY_S`, `PUBLIC_APP_URL` (for the
  Teams card link). Teams POST: httpx (already a transitive dep of openai),
  5 s timeout, fire-and-forget, never raises.

## 6. Frontend — `backend/agent/static/operator.html`

Professional, calm, English. Distinct from the amber debug console: dark
slate, system font stack, severity palette shared with backend labels, teal
reserved for VLM/hologram accents. Layout: header (persona selector = demo
routing, live/mock badge, epoch) · left rail (machine cards: status dot from
last tick risk, key signals, assigned operator chip, pulsing "on-site" badge
during interventions, + reserved 3D slot) · center (machine detail: alert
card, actions Take charge / Mark repair done / dismiss-with-reason, live
signals vs limits, claimed-vs-verified timeline) · right (chat: quick by
default with suggested questions, Deep analysis toggle showing tool trace) ·
toast stack top-right (Teams-style, click → machine view) + bell history.
Boot: `/api/factory/state` + `/api/operators` + SSE; hash deep-link
`#machine=RC-07`; persona in localStorage; idle banner offers to start the
loop. No external assets — fully offline.

## 7. 3D map integration contract (for the other conversation)

- Initial state: `GET /api/factory/state` (shape above).
- Live: subscribe `GET /api/stream` — `tick` (risk/signals per machine),
  `advisory`, `notify`, `intervention` (render/remove hologram from
  `status` + `operator_name`), `boss`.
- Mount point: `<div id="map3d-slot">` in operator.html; replace its
  placeholder content. Everything needed (machine ids, departments, statuses,
  who-is-repairing) comes from the two URLs above — no extra wiring needed.

## 8. Testing & verification

- New `backend/agent/tests/test_operator_flow.py` (MOCK_LLM=1, temp DB, tiny
  VLM delays): notified-on-HIGH + routing, cooldown dedup, take_charge
  side-effects (accept + broadcast), VLM event order + timestamps, repair_done
  guard, factory_state shape, quick-chat citations. Existing 4 suites must
  stay green.
- End-to-end (mock): own uvicorn on port 8010 + temp DB — never touching the
  live :8000 server — drive loop → notify → take_charge → repair_done → SSE
  assertions; page served. Live smoke happens on the user's :8000 after
  restart.

## 9. Risks

- Live Nemotron latency → quick mode pinned to the fast model.
- Advisory spam → notification cooldown.
- Windows/WSL SQLite quirks → tests and e2e use temp-dir DBs (store already
  falls back).
