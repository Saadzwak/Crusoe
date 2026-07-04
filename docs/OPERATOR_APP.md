# Operator App — `/operator`

The operator-facing side of PRAETOR: alert → machine view → fast chat →
take-charge → VLM-verified repair timeline. Design spec:
`docs/superpowers/specs/2026-07-04-operator-app-design.md`. Backend module:
`backend/agent/operator_flow.py` · UI: `backend/agent/static/operator.html`
(single file, no external assets, fully offline) · tests:
`backend/agent/tests/test_operator_flow.py` (mock, 14 checks).

## The journey (also the demo script)

1. The loop raises a HIGH/CRITICAL advisory on a machine. PRAETOR routes it
   to the operator ASSIGNED to that machine (`RC-07`,`CL-03` → Eric;
   `MX-02` → Lina) — a Teams-style toast in the app, plus a real Teams card
   if `TEAMS_WEBHOOK_URL` is set. Other operators don't get paged.
2. The operator clicks **Open machine** → machine view: severity, the
   advisory with its cited evidence, proposed action, cost/TTF/safety
   impacts, jury gate result, live signals vs the verified limits.
3. Questions go to the **Line assistant** — *Quick* mode answers in seconds
   (DeepSeek Flash, context injected, no tool loop; citations + sources are
   built programmatically). *Deep tools* switches to the tool-calling agent
   with the audited Data-Provenance trail.
4. **Take charge** → the linked advisory is auto-accepted (learning loop
   preserved), the whole line gets a broadcast ("Eric is on it"), the 3D map
   shows its hologram, and the VLM watcher starts.
5. The simulated VLM logs `activity_detected` (~8 s), and after the operator
   clicks **Mark repair done** it logs `repair_verified` (~10 s) and closes
   the intervention with a broadcast.
6. The **claimed-vs-witnessed timeline** shows both rails with deltas:
   what the operator claimed (yellow) vs what the VLM saw (cyan).

Persona dropdown in the header switches operator (demo of routing).
"Line idle — start simulation" appears if the loop isn't running.

## Intervention state machine

`notified → acknowledged → claimed_done → verified` — one active
intervention per machine, persisted in the `interventions` table.

- `on_advisory` (loop hook): severity ≥ HIGH creates `notified` + routed
  alert; while an intervention is active, new advisories link silently.
  Per-machine toast cooldown `NOTIFY_COOLDOWN_S` (default 90 s).
- `take_charge`: any known operator, also proactively. Auto-accepts the
  linked pending advisory through the existing override path.
- `repair_done`: only the operator who took charge (409 otherwise).
- VLM watcher (`_observe()` in operator_flow.py is the SEAM where the real
  camera + `role="omni"` call goes): `VLM_ACTIVITY_DELAY_S` after
  take-charge → `activity_detected` (0.91); `VLM_VERIFY_DELAY_S` after the
  claim → `repair_verified` (0.87) + closure broadcast. Observations are
  labeled `[simulated VLM]` until the camera lands.

## HTTP API

| Method & path | Body | Returns |
|---|---|---|
| `GET /operator` | – | the app |
| `GET /api/operators` | – | `{operators:[{id,name,initials,area}], assignments:{machine→operator_id}}` |
| `GET /api/factory/state` | – | `{mode, epoch, at, machines:[…]}` (below) |
| `POST /api/machine/{id}/take_charge` | `{"operator_id":"eric"}` | Intervention (409 on conflict) |
| `POST /api/machine/{id}/repair_done` | `{"operator_id":"eric"}` | Intervention (409 on conflict) |
| `POST /api/chat` | `{"question","stream",` **`"mode":"quick"|"deep"`**`, "machine_id"}` | quick: token*/turn SSE or JSON turn with `citations[]` + `provenance[{source,detail}]`; deep: unchanged tool-calling contract |

`machines[]` entry: `{machine_id, department, type, role, assigned_operator,
risk, last_reading{epoch,signals,pinn,at}, active_advisory|null,
intervention|null, signals_spec}` — `signals_spec` carries the verified
limits from `limits.py` so no client hardcodes a copy.

## SSE additions (same `GET /api/stream`)

- `notify` —
  `{id, kind:"alert"|"broadcast", to_operator:{id,name}|null, machine_id,
  advisory_id, severity, title, body, at}`. `alert` is routed: only the
  named operator's UI should toast it. `broadcast` is for everyone.
- `intervention` — full Intervention dump on EVERY transition:
  `{id, machine_id, advisory_id, status, operator_id, operator_name,
  notified_operator_id, notified_at, acknowledged_at, claimed_done_at,
  vlm_activity_at, vlm_verified_at, vlm_observations:[{kind, detail,
  confidence, at}], created_at}`.

## 3D map integration (other conversation)

Mount into `<div id="map3d-slot">` in `operator.html`. Boot from
`GET /api/factory/state`; animate from `GET /api/stream`:
- machine status → `tick` (risk) / `advisory`,
- **hologram** → `intervention`: show it while `status ∈ {acknowledged,
  claimed_done}` with `operator_name`; remove on `verified`,
- toasts already handled by the host page.
Nothing else is needed — ids, departments, and who-is-on-site all come from
those two URLs.

## Real Teams webhook (optional)

Create an incoming webhook on a Teams channel, then in `.env`:
`TEAMS_WEBHOOK_URL=https://…`. Every notify also POSTs a MessageCard
(fire-and-forget, 5 s timeout, failures only logged — the in-app toast is
the demo path and never depends on the network). `PUBLIC_APP_URL` sets the
card's "Open operator console" link.

## Env vars (see `.env.example`)

`TEAMS_WEBHOOK_URL` (empty = skip) · `PUBLIC_APP_URL` ·
`NOTIFY_COOLDOWN_S=90` · `VLM_ACTIVITY_DELAY_S=8` · `VLM_VERIFY_DELAY_S=10`.

## Verify locally

```bash
MOCK_LLM=1 python backend/agent/tests/test_operator_flow.py   # 14 checks
MOCK_LLM=1 python -m uvicorn backend.agent.main:app --port 8000
# open http://localhost:8000/operator → start simulation → alert ~epoch 6
```
