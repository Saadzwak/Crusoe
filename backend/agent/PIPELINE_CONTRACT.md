# PRAETOR backend — build contract (do not edit; architect-owned)

Two builders work in parallel. This file pins the interfaces and file ownership
so the merge is clean. Shared files (`config.py`, `schemas.py`, `crusoe_client.py`,
`hmac_auth.py`, this file) are READ-ONLY for builders — if you need a change,
write a `NOTES-<yourname>.md` instead and keep building around it.

## File ownership

**Builder A — reasoning modules (`backend/agent/`):**
- `prompts.py` — every system/user prompt template, in one place
- `triage.py` — Tier 1 (pure python thresholds) + Tier 2 (fast model classify) + causal context
- `line_llm.py` — Tier 3 advisory generation (reasoning model, structured output, cites data)
- `debate.py` — DebateRoom: Advocate (reasoning) proposes, Skeptic (fast) attacks, ≤2 rounds, judge verdict
- `jury.py` — Jury LLM-as-judge: scores every advisory (JuryScore) before operator display
- `boss_llm.py` — plant-wide summary from the shared store (reads store, never talks to line agents)
- `pipeline.py` — `AdvisoryPipeline` orchestrator (see interface below)
- `tests/test_pipeline_mock.py` — plain-script test, runs fully in mock mode

**Builder B — service layer (`backend/agent/` + `backend/api/`):**
- `state_store.py` — SQLite shared store (see interface below)
- `telemetry_source.py` — PINN stand-in: replays C-MAPSS FD001 (`pinn/data/train_FD001.txt`) and
  AI4I 2020 (`pinn/data/ai4i2020.csv`) as `SignedReading`s (HMAC-signed), mapped to the three
  Roanne departments: Mixing (Banbury), Calendering, Curing (C3M electric presses — the bottleneck)
- `knowledge.py` — Michelin case knowledge tool over `knowledge/site_dossier.md` +
  `knowledge/rapport_socio_economique.md`: keyword/section retrieval returning
  `(snippet, citation)` pairs like `[site_dossier p.4]`
- `operator.py` — operator Q&A agent: answers WITH citations (gbrain pattern: synthesized answer
  + explicit "what I don't know" gap note), tools = sensor history, PINN state, knowledge, past
  fixes/overrides; and the override handler (stores OperatorOverride, feeds future prompts)
- `main.py` — FastAPI app (root `backend/agent/main.py`), endpoints below + serves `static/index.html`
- `static/index.html` — single-file demo page (see demo rules)
- `tests/test_service_mock.py` — plain-script test

## Shared interfaces (already implemented — import, don't redefine)

```python
from backend.agent.config import settings
from backend.agent.crusoe_client import get_client   # .complete / .complete_json / .stream
from backend.agent.schemas import (PinnReading, SignedReading, TriageResult, RiskLabel,
    DebateTurn, DebateResult, JuryScore, Advisory, AdvisoryStatus, OperatorOverride,
    PlantSummary, TickResult, OperatorTurn, PinnState)
from backend.agent.hmac_auth import sign_payload, verify_payload
```

Client roles: `"fast"` = DeepSeek V4 Flash (~100ms, loop/skeptic/classify),
`"reasoning"` = Nemotron Ultra 550B (advocate/advisory/jury/boss/operator).
Always pass `hint="tier2_classify" | "advocate" | "skeptic" | "jury" | "advisory" |
"boss" | "operator"` — the mock keys on it. Mock mode is automatic without a key;
NEVER branch on `client.is_mock` in business logic.

## Interfaces to implement

```python
# pipeline.py (Builder A)
class AdvisoryPipeline:
    def __init__(self, client, store, knowledge=None): ...
    async def process_reading(self, signed: SignedReading) -> TickResult:
        """verify HMAC → reject or claim → Tier1 → Tier2 → (Tier3 advisory + debate + jury)
        → persist everything to store → return TickResult. Must never raise."""
    async def boss_summary(self) -> PlantSummary: ...

# state_store.py (Builder B) — SQLite at settings.db_path, std sqlite3, thread-safe
class StateStore:
    def save_tick(self, tick: TickResult) -> None: ...
    def save_advisory(self, adv: Advisory) -> None: ...          # upsert by id
    def save_override(self, ov: OperatorOverride) -> None: ...
    def get_advisories(self, status=None, limit=50) -> list[Advisory]: ...
    def get_advisory(self, advisory_id) -> Advisory | None: ...
    def get_sensor_history(self, machine_id, limit=40) -> list[dict]: ...  # ticks w/ signals+pinn
    def get_overrides(self, limit=20) -> list[OperatorOverride]: ...
    def department_snapshot(self) -> dict[str, dict]: ...        # per-dept last risk/epoch
    def log_event(self, kind: str, data: dict) -> None: ...      # flight-recorder rows
```

Builder A: code against this StateStore signature (duck-typed); Builder B implements it.
Builder A's test uses a tiny in-memory fake with the same methods.

## FastAPI endpoints (Builder B)

- `GET /` → demo page
- `GET /api/health` → `{mode: "mock"|"live", models: {...}}`
- `POST /api/loop/start?interval=5` / `POST /api/loop/stop` — background loop task:
  each tick pulls next telemetry batch → `pipeline.process_reading` per machine
- `GET /api/stream` — SSE: every TickResult, advisory, debate turn, jury score, boss summary
  as `event: <kind>\ndata: <json>\n\n`
- `GET /api/advisories`, `POST /api/advisory/{id}/accept`, `POST /api/advisory/{id}/override` (reason)
- `POST /api/chat` — operator Q&A (SSE or JSON w/ citations)
- `GET /api/plant` — boss summary on demand
- CORS open to FRONTEND_URL + localhost:3000 (Next.js teammate).

## Demo page rules (anti-dashboard — hackathon guardrail)

The demo must show THE AGENT ACTING, not a wall of gauges: a live advisory feed
(tier transitions visible: "Tier1 clear → … → Tier2 HIGH → Tier3 advisory"),
the debate transcript unfolding, the jury verdict stamped on the advisory card,
Accept / Override buttons (override reason visibly fed back), operator chat with
citations, boss summary strip. Dark industrial look, vanilla JS + SSE, ONE file.
Machines: RC-07 (Curing press, C3M), CL-03 (Calender), MX-02 (Banbury mixer).

## Narrative constants (from the Michelin case — cite them)

- Site: Michelin Roanne, UHP passenger tyres, ≈5,000 tyres/day, C3M electric curing.
- Curing = bottleneck; calendering nip points + hot high-pressure presses = human-risk stages.
- Cost of stoppage: see `knowledge/site_dossier.md` p.4+ — cite as `[site_dossier p.N]`.
- Datasets stand in for confidential C3M signals (see `pinn/data/README.md` mapping table).

## Ground rules

- Python 3.10+, only deps in `backend/requirements.txt` (+ stdlib). No LangChain.
- Public repo: no keys, no dataset dumps in git besides what's already there.
- Every LLM output shown to the operator must carry its citations or say what it doesn't know.
- Everything must run offline: `MOCK_LLM=1 python -m backend.agent.main` then open localhost:8000.
