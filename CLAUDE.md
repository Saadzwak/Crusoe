# CLAUDE.md — PRAETOR / Crusoe track (RAISE Summit hackathon)

Context handoff from the Cowork session of 2026-07-04. Read this first; the
detailed docs it points to are all in `docs/`.

## The project in five lines

Factory digital twin for the Michelin Roanne case (UHP tyres, C3M electric
curing line, ~5,000 tyres/day): a PINN understands each machine's physics
(teammates, `pinn-physical-layer` branch), and an LLM information layer —
THIS code — turns physics into advisories a human can question and override.
Demo due Sunday 2026-07-05 12:00. Track rule: the demo must show the agent
ACTING, never a passive dashboard. Repo is PUBLIC: no keys in code, ever
(`.env` is gitignored; `.env.example` documents every variable).

## State of the work (as of commit cdf4ede)

DONE and green (in mock): 3-tier advisory pipeline (HMAC verify → python
thresholds → DeepSeek Flash classify → Nemotron Ultra advisory with cited
values) → Advocate/Skeptic debate (≤2 rounds, safe-hold escalation) → Jury
LLM-as-judge gate on every advisory (+ deterministic backstop: no cited digits
= fail) → shared SQLite store (NO agent-to-agent mesh — architecture rule) →
FastAPI + SSE console (`backend/agent/static/index.html`). Tool-calling
operator chatbot per team spec (LangChain `bind_tools`, 5 real tools + 2
PINN/camera stubs, programmatic "Data Provenance" section). Verified machine
limits with source provenance in `backend/agent/limits.py` (temp 180–210 °C,
pressure >20 bar OR 16–19 bar steam — both source formulations kept, 19–20 gap
= indeterminate WATCH; cycle 10–15/30 min; ISO 10816-3 vib zones A≤1.12
B≤2.8 C≤7.1 D>7.1 — flagged as generic reference, not machine-measured).
Scenario harness: 10 scenarios + boss, **11/11 components PASS**
(`backend/agent/tests/test_scenarios.py`); labeled synthetic dataset at
`pinn/data/synthetic_rc07_scenarios.csv`. Five test suites total, all green
offline.

Session 2026-07-04 (evening) added the **operator app** (`/operator`,
`backend/agent/operator_flow.py` + `static/operator.html`, spec + contract in
`docs/OPERATOR_APP.md`): machine→operator alert routing with Teams-style
toasts (+ optional real webhook via `TEAMS_WEBHOOK_URL`), intervention state
machine notified→acknowledged→claimed_done→verified with simulated-VLM
verification (seam ready for the real camera + omni model), claimed-vs-
witnessed timeline, quick-lane chat (`mode:"quick"` on `/api/chat`, DeepSeek
Flash, ~seconds, programmatic citations), and a documented mount point for
the 3D factory map being built in a separate conversation (`#map3d-slot`).
E2E-verified in mock on :8010 (full journey incl. 409 guards + auto-accept).

## Which model runs where (all via Crusoe, OpenAI-compatible)

| Layer | Model |
|---|---|
| Tier 1 thresholds | none — pure python |
| Tier 2 classify | `deepseek-ai/Deepseek-V4-Flash` |
| Tier 3 advisory, Advocate, Jury, Boss, Operator chatbot | `nvidia/NVIDIA-Nemotron-3-Ultra-550B` |
| Skeptic | `deepseek-ai/Deepseek-V4-Flash` |
| Reserved (unused) | `nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B` |

Every call goes through ONE wrapper: `backend/agent/crusoe_client.py`
(`get_client()` → `.complete(role="fast"|"reasoning", ...)`). It owns:
base_url `https://api.inference.crusoecloud.com/v1/`, key from `.env`,
thinking-token disabling per model family, 412-capacity retry (0/3/8 s), JSON
extraction + one repair retry, and per-call degradation to a deterministic
MockClient. Do NOT scatter raw `OpenAI(...)` calls in layers — extend the
wrapper. API reference: `docs/CRUSOE.md` (model quirks, payload formats).

## ⚠️ THE ONE CRITICAL FACT

**Everything so far ran in MOCK mode.** The previous session ran in a sandbox
whose egress proxy blocks `api.inference.crusoecloud.com` (403 on CONNECT) —
the key was loaded, the code is live-wired, but no request ever reached
Crusoe. All jury scores of exactly 4.1 are the mock's signature. Nothing has
been live-verified yet. That is THIS session's first job.

## First actions on this machine (in order)

1. `pip install -r backend/requirements.txt` (once).
2. `python scripts/crusoe_sanity.py` — expects: models listed on the key,
   LIVE completions on both tiers, and a native tool-call round-trip. If any
   check fails, debug from the printed error (401 = key; 404 model id =
   compare with the listed catalog; 412 = capacity, retry in 60 s).
3. Live smoke of the real thing: `python -m uvicorn backend.agent.main:app
   --port 8000` (NO `MOCK_LLM` env var), open `http://localhost:8000`, start
   the loop, and watch: tier transitions, a real advisory (~epoch 6), the
   debate transcript, a REAL jury score (will vary, not 4.1), then ask the
   chatbot "Why the alarm on RC-07? cite data" and check the Data Provenance
   block lists real tools.
4. Expect live-only behaviors: Nemotron Ultra latency (advisory can take
   10–60 s — raise loop interval if needed: `POST /api/loop/start?interval=15`),
   `<think>` leakage is stripped but watch for it in streams, 412 under load
   degrades that one call to mock and logs it (demo never dies).
5. The 5 test suites are DESIGNED for mock — keep running them with
   `MOCK_LLM=1` (deterministic; they'd be flaky live):
   `MOCK_LLM=1 python backend/agent/tests/test_scenarios.py` etc.
   (5th suite: `test_operator_flow.py` — operator app, 14 checks.)

## Open items (priority order)

1. Push: local `main` == `feat/llm-information-layer` == `cdf4ede`, both ahead
   of origin. `git push origin main feat/llm-information-layer` (if main is
   protected, push the branch and merge via PR — body ready in
   `docs/PR_BODY_LLM_LAYER.md`).
2. Live-tune prompts if real outputs disappoint (they were only ever exercised
   by the mock): prompts live in `backend/agent/prompts.py` +
   `operator_agent.py`. Mind the mock-keyword discipline notes in
   `backend/agent/NOTES-B.md` §footgun if you touch digests/notes.
3. Policy question for Eric: should pressure < 16 bar ALONE force CRITICAL
   (scrap risk) instead of riding along at HIGH? One-line change in
   `limits.py`/`triage.py`.
4. Privacy decision: `backend/agent/knowledge/rapport_socio_economique.md` is
   extracted from Eric's PERSONAL internship report and is in the public
   history (source .docx is gitignored). Remove if he says so.
5. Integration: PINN teammates implement the `SignedReading` contract
   (`backend/agent/schemas.py` — payload + HMAC-SHA256 hex via
   `hmac_auth.sign_payload`, shared secret `PINN_HMAC_SECRET`). Frontend
   teammate consumes the SSE contract (below). Gradium TTS (coupon RAISE-2026)
   optional if time remains.

## Cheat sheet

- Run server: `python -m uvicorn backend.agent.main:app --port 8000`
  (mock: prefix `MOCK_LLM=1`). Console at `/`. Health: `GET /api/health`
  (shows mode mock/live + `chat_agent`).
- Loop: `POST /api/loop/start?interval=5&reset=1` · `POST /api/loop/stop`.
- Events: `GET /api/stream` (SSE: hello/tick/advisory/boss/override/
  tool_call/tool_result).
- Chat: `POST /api/chat` `{"question": "...", "stream": true}` → SSE
  `tool_call{name,params}` / `tool_result{name,ok,summary}` pairs →
  `token{text}`× → `turn{content, citations[], provenance[]}` → `done{}`.
  `stream:false` → same turn as JSON.
- Overrides: `POST /api/advisory/{id}/accept|override` `{"reason": "..."}` —
  reasons feed future Skeptic + operator prompts (the learning loop).
- Operator app: page `GET /operator` · `GET /api/operators` ·
  `GET /api/factory/state` (also the 3D map's boot contract) ·
  `POST /api/machine/{id}/take_charge|repair_done` `{"operator_id":"eric"}` ·
  chat quick lane: add `"mode":"quick","machine_id":"RC-07"` to `/api/chat`.
  New SSE kinds on the same stream: `notify` + `intervention`.
- Docs map: architecture `docs/LLM_LAYER_README.md` · runtime flow + trust
  boundaries `docs/DATAFLOW.md` · file/dependency model `docs/REPO_MAP.md` ·
  Crusoe API `docs/CRUSOE.md` · findings in plain words
  `docs/TROUBLESHOOTING_SIMPLE.md` · build contract
  `backend/agent/PIPELINE_CONTRACT.md` · operator app + 3D-map contract
  `docs/OPERATOR_APP.md` · builder logs `backend/agent/NOTES-*.md`.

## Conventions

- Public repo: never commit `.env`, keys, or personal documents.
- Single source of truth for limits: `backend/agent/limits.py` — tests must
  read from it, never hardcode limit copies (that exact bug already happened).
- Every operator-facing LLM output must cite values/sources or state gaps.
- Human stays in charge: agents propose, never command.
- `core.fileMode false` is set (files show mode 755 from WSL — ignore).
- Datasets in `pinn/data/` are committed on purpose (public benchmarks, a
  fresh clone must run).
