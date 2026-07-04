# End-to-end architecture & API audit — 2026-07-04 late

Verdict: **end-to-end COMPLETE for the demo scope, with 7 known, documented
gaps** (list below — none blocks the demo). All 5 test suites green after this
audit's fixes.

## Endpoints (15, all in backend/agent/main.py)

| Route | Purpose | Auth |
|---|---|---|
| GET `/` | agent console | none |
| GET `/operator` | operator app (alerts, interventions, quick chat) | none |
| GET `/api/health` | mode mock/live, models, chat_agent | none |
| GET `/api/operators` | roster + assignments | none |
| GET `/api/factory/state` | machines, risks, interventions — 3D map boot contract | none |
| POST `/api/machine/{id}/take_charge` | operator claims an alert (409 if taken) | none |
| POST `/api/machine/{id}/repair_done` | operator claims completion → VLM verify | none |
| POST `/api/loop/start` `?interval&reset` / `.../stop` | telemetry loop | none |
| GET `/api/stream` | SSE: hello/tick/advisory/boss/override/notify/intervention | none |
| GET `/api/advisories` `?status&limit` | list advisories | none |
| POST `/api/advisory/{id}/accept` / `override` | human decision + reason | none |
| POST `/api/chat` | operator Q&A; SSE tool_call/tool_result/token/turn/done; `mode:"quick"` = fast lane | none |
| GET `/api/plant` | boss summary on demand — RESTORED by this audit (was dropped in the operator-app rework) | none |

No authentication anywhere — acceptable for a localhost/LAN demo, NOT for
deployment. CORS restricted to FRONTEND_URL/localhost:3000.

## Audit findings & fixes applied

1. **3 torn files healed** (config.py, schemas.py, state_store.py — the mount
   serves in-place edits corrupted; stale __pycache__ masked it and made suites
   look green). All .py verified whole; caches cleared; 5/5 suites pass.
2. **/api/plant restored** — docs and the frontend contract promised it; the
   evening rework dropped it.
3. **Known drift (documented, not fixed):** chat tool_call/tool_result events
   stream on the CHAT response but are no longer re-published to `/api/stream`
   watchers. The consoles read the chat stream directly, so nothing breaks;
   external `/api/stream` watchers won't see chat tool activity.
4. `.claude/` (local tool settings) added to .gitignore.

## Security path (HMAC)

sign at source (`telemetry_source` — later: the real PINN) → payload+signature
travel together (`SignedReading`) → `verify_payload` FIRST thing in
`pipeline.process_reading` (before parse; reject → `hmac_rejected` event) and
again in the loop before `save_reading` → signature+exact payload stored at
rest → `verify_custody_chain` tool re-derives every stored link on demand and
names any broken one. Secret: `PINN_HMAC_SECRET` (demo fallback exists —
provision a real one).

## Honest gap list (demo-safe, say them out loud if asked)

1. Real PINN not plugged yet — `TelemetrySource` replays benchmark data,
   signed with the same contract the PINN team must emit (`SignedReading`).
2. CCCL 3-pass compression from the planning doc: NOT built. What exists is
   single-pass digesting (scalars → one cited line, full payload kept at rest).
3. SM2 adaptive polling: NOT built — loop interval is a fixed API parameter.
4. `get_pinn_reconstruction` / `get_camera_frame`: honest stubs by design.
5. Operator-app VLM verification is SIMULATED (timed transitions; seam ready
   for Nemotron Omni + camera).
6. No API auth / rate limiting (demo scope).
7. Live Crusoe still unverified end-to-end (sandbox egress blocked; run
   `python scripts/crusoe_sanity.py` on the host).
