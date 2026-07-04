# PRAETOR — LLM information layer (Crusoe track)

Adds the complete information layer: everything from the moment physics-derived
telemetry arrives (HMAC-signed) to the moment a human decides.

## What's inside

**Advisory pipeline** (background, ~5 s loop, per machine)
`HMAC verify → Tier1 python thresholds (clears ~80%, zero cost) → Tier2 DeepSeek V4
Flash classify (~100 ms) → Tier3 Nemotron Ultra 550B advisory (cites real values +
[site_dossier p.N]) → Advocate/Skeptic debate (≤2 rounds, else safe-hold escalation)
→ Jury LLM-as-judge (grounding / safety / actionability / human-in-charge; advisories
with zero cited digits auto-fail) → shared SQLite store → SSE`.

**Tool-calling operator chatbot** (on demand, per team spec)
LangChain `bind_tools` on Nemotron Ultra (temp 0.1, ≤6 calls/turn):
`get_sensor_history · get_machine_spec · run_diagnostic · verify_custody_chain
(HMAC at rest, names the broken link) · analyze_drift` + PINN/camera stubs that
answer "physical subsystem not integrated". Every answer ends with a **Data
Provenance** section built programmatically from the actual call log.

**Boss-LLM** plant brief every 5 epochs — reads only the shared store (the
"no agent-to-agent mesh" rule).

**Demo surface**: FastAPI + SSE (`/api/stream`, `/api/chat` streams
`tool_call/tool_result/token/turn`) + single-file agent console showing tier
transitions, debate transcripts, jury stamps, accept/override with reasons
feeding back into future prompts.

**Michelin Roanne case**: C-MAPSS FD001 → RC-07 (C3M curing press, run-to-failure
arc, CRITICAL ~epoch 10), AI4I 2020 → CL-03 (calender, HDF spike) & MX-02 (mixer);
knowledge tool over the site dossier with page-level citations.

## Run it

```bash
pip install -r backend/requirements.txt
MOCK_LLM=1 python -m uvicorn backend.agent.main:app --port 8000   # offline demo
# live: put CRUSOE_API_KEY in .env (see .env.example), drop MOCK_LLM
python scripts/crusoe_sanity.py                                    # live check
```

Full dataflow + trust boundaries: `docs/DATAFLOW.md`. Architecture mapping:
`docs/LLM_LAYER_README.md`. API reference used: `docs/CRUSOE.md`.

## Tests

Three suites, all green in mock (no network needed):
`backend/agent/tests/test_pipeline_mock.py` · `test_service_mock.py` (14) ·
`test_operator_agent_mock.py` (12 — includes custody tamper detection).

## Notes for reviewers

- Public-repo hygiene: no keys anywhere; `.env` gitignored; HMAC demo fallback
  clearly marked not-for-prod.
- Mock/live parity: identical code path; no business logic branches on mock.
- Known stubs: PINN reconstruction & camera tools (await the physical layer),
  Champion/Challenger loop out of scope for this layer.
