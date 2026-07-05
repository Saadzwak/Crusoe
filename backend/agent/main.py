"""PRAETOR FastAPI service (Builder B).

Run:  MOCK_LLM=1 uvicorn backend.agent.main:app --port 8000
  or  MOCK_LLM=1 python3 -m backend.agent.main

Endpoints (PIPELINE_CONTRACT.md):
  GET  /                    demo page (static/index.html)
  GET  /api/health          {mode, models, pipeline_available, loop_running, epoch}
  POST /api/loop/start?interval=5[&reset=1]
  POST /api/loop/stop
  GET  /api/stream          SSE: tick / advisory / boss / override / hello events
                            (+ tool_call / tool_result while the chat agent works)
  GET  /api/advisories      [?status=&limit=]
  POST /api/advisory/{id}/accept
  POST /api/advisory/{id}/override   {"reason": "..."}
  POST /api/chat            {"question": "...", "stream": true} → SSE or JSON
                            (+ mode: "quick"|"deep", machine_id — operator app)
  GET  /api/plant           boss summary on demand

Operator app (operator_flow.py — spec docs/OPERATOR_APP.md):
  GET  /operator            operator web app (static/operator.html)
  GET  /api/operators       personas + machine assignments
  GET  /api/factory/state   per-machine snapshot (also feeds the 3D map)
  POST /api/machine/{id}/take_charge  {"operator_id": "eric"}
  POST /api/machine/{id}/repair_done  {"operator_id": "eric"}
  SSE adds `notify` (alert/broadcast toasts) + `intervention` (lifecycle).

Builder A's pipeline is imported at module top inside try/except. If absent,
the loop runs in "triage-echo" degraded mode: HMAC-verify each reading and
publish raw ticks with a keyword triage — so the service demos without A.

C3-toolchat-v1 (Builder C): /api/chat now routes through the tool-calling
ToolCallingOperator (operator_agent.py) when it loads — SSE order per turn:
tool_call* / tool_result* / token* / turn / done — and every tool_call /
tool_result is ALSO published on the main hub so /api/stream watchers see the
agent working. The original OperatorAgent stays as fallback and still owns
the override handling.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles  # integration: serve CureWatch assets
from pydantic import BaseModel

from .config import settings
from .crusoe_client import get_client
from .knowledge import Knowledge
from .operator import OperatorAgent
from .operator_flow import (ASSIGNMENTS, OPERATORS, InterventionManager,
                            build_factory_state, quick_answer, stream_quick)
from .schemas import (PinnReading, PlantSummary, SignedReading, TickResult,
                      TriageResult)
from .state_store import StateStore
from .telemetry_source import REPLAY_ONLY_IDS, TelemetrySource, note_risk
from .hmac_auth import verify_payload

# Builder A's orchestrator — optional at boot, hot when the file lands.
try:
    from .pipeline import AdvisoryPipeline  # type: ignore
    PIPELINE_AVAILABLE = True
except Exception:  # noqa: BLE001 — missing file or half-written module
    AdvisoryPipeline = None  # type: ignore
    PIPELINE_AVAILABLE = False

_STATIC = Path(__file__).resolve().parent / "static"


# ===================================================================== hub
class Hub:
    """In-process SSE fanout: publish(kind, data) → every connected client."""

    def __init__(self) -> None:
        self._subs: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    def publish(self, kind: str, data: Any) -> None:
        for q in list(self._subs):
            try:
                q.put_nowait((kind, data))
            except asyncio.QueueFull:  # slow client — drop, never block the loop
                pass


def _sse(kind: str, data: Any) -> str:
    return f"event: {kind}\ndata: {json.dumps(data, default=str)}\n\n"


# ===================================================================== state
class ServiceState:
    def __init__(self) -> None:
        # Integration/demo: start every launch from a clean slate. The SQLite
        # store persists across restarts; without this, stale readings and
        # advisories from a previous run leak into the fresh heartbeat (the
        # chat once described a "critical" machine that was actually healthy).
        # Within a running session the history still accumulates normally.
        if os.environ.get("DEMO_FRESH_START", "1") == "1":
            try:
                dbp = Path(settings.db_path)
                for p in (dbp, Path(str(dbp) + "-journal"), Path(str(dbp) + "-wal")):
                    if p.exists():
                        p.unlink()
            except Exception as e:  # noqa: BLE001
                print(f"[main] fresh-start db wipe skipped ({e!r})")
        self.store = StateStore()
        self.knowledge = Knowledge()
        self.client = get_client()
        try:
            self.telemetry = TelemetrySource()
        except Exception as e:  # noqa: BLE001 — e.g. fresh clone without pinn/data files
            print(f"[main] TelemetrySource init failed ({e!r}) — put the demo datasets "
                  f"in pinn/data/ (see pinn/data/README.md). Loop disabled until then.")
            self.telemetry = None
        self.operator = OperatorAgent()
        # C3-toolchat-v1: tool-calling chat agent — lazy import; on ANY failure
        # the legacy OperatorAgent keeps serving /api/chat (demo never dies).
        self.tool_operator: Any = None
        try:
            from .operator_agent import ToolCallingOperator  # type: ignore
            self.tool_operator = ToolCallingOperator(self.store, self.knowledge)
            print("[main] ToolCallingOperator ready — /api/chat is tool-calling")
        except Exception as e:  # noqa: BLE001
            print(f"[main] ToolCallingOperator unavailable ({e!r}); "
                  "/api/chat falls back to the evidence-block OperatorAgent")
        self.hub = Hub()
        # Operator app: intervention lifecycle + notification routing
        # (operator_flow.py) — publishes `notify`/`intervention` on the hub.
        self.interventions = InterventionManager(self.store, self.hub.publish,
                                                 operator_agent=self.operator)
        self.pipeline: Any = None
        self.loop_task: Optional[asyncio.Task] = None
        self.running = False
        self.epoch = 0

    def ensure_pipeline(self) -> None:
        if self.pipeline is None and PIPELINE_AVAILABLE and AdvisoryPipeline:
            try:
                self.pipeline = AdvisoryPipeline(self.client, self.store,
                                                 knowledge=self.knowledge)
            except Exception as e:  # noqa: BLE001
                print(f"[main] AdvisoryPipeline init failed ({e!r}); staying in triage-echo mode")


S = ServiceState()

if settings.demo_skip_hmac:  # integration: make the bypass impossible to miss
    print("=" * 64)
    print("!!  DEMO BYPASS ACTIVE — HMAC verification is SKIPPED         !!")
    print("!!  (DEMO_SKIP_HMAC=1)  Remove before any real deployment.    !!")
    print("=" * 64)


# ===================================================================== loop
def _echo_tick(signed: SignedReading) -> TickResult:
    """Degraded-mode triage: HMAC verify + keyword risk from the PINN note."""
    # DEMO BYPASS (temporary, reversible): see config.demo_skip_hmac.
    ok = settings.demo_skip_hmac or verify_payload(signed.payload, signed.signature)
    if not ok:
        return TickResult(
            machine_id=str(signed.payload.get("machine_id", "?")),
            department=str(signed.payload.get("department", "?")),
            epoch=int(signed.payload.get("epoch", -1)),
            hmac_verified=False,
            rejected_reason="HMAC signature mismatch — reading rejected before processing",
        )
    reading = PinnReading.model_validate(signed.payload)
    risk = note_risk(reading.note)
    triage = TriageResult(
        tier_reached=1 if risk.value == "CLEAR" else 2,
        risk=risk,
        reason=f"triage-echo (pipeline offline): note-level screen → {risk.value}",
    )
    return TickResult(machine_id=reading.machine_id, department=reading.department,
                      epoch=reading.epoch, hmac_verified=True, triage=triage)


async def _boss_summary() -> PlantSummary:
    """Pipeline boss if available, else a service-side fallback that reads the
    shared store (same inputs, humbler author)."""
    S.ensure_pipeline()
    if S.pipeline is not None:
        try:
            return await S.pipeline.boss_summary()
        except Exception as e:  # noqa: BLE001
            print(f"[main] pipeline.boss_summary failed ({e!r}); using fallback")
    snap = S.store.department_snapshot()
    open_adv = len(S.store.get_advisories(status="pending", limit=200))
    text = await S.client.complete(
        role="reasoning", hint="boss", max_tokens=220,
        messages=[
            {"role": "system",
             "content": "You are the plant-wide summary agent for the Michelin "
                        "Roanne line. Two sentences, plain language, name each "
                        "department and its state, mention open advisories."},
            {"role": "user",
             "content": "Department snapshot (last tick per department): "
                        + json.dumps(snap, default=str)
                        + f"\nOpen advisories: {open_adv}"},
        ],
    )
    return PlantSummary(
        text=text,
        department_status={d: v.get("risk", "UNKNOWN") for d, v in snap.items()},
        open_advisories=open_adv,
        generated_by="service-fallback" if S.pipeline is None else "pipeline",
    )


async def _run_loop(interval: float) -> None:
    S.running = True
    S.ensure_pipeline()
    mode = "pipeline" if S.pipeline is not None else "triage-echo (degraded)"
    print(f"[main] loop started — interval={interval}s, mode={mode}")
    try:
        while S.running:
            S.epoch += 1
            epoch = S.epoch
            try:
                batch = S.telemetry.next_batch(epoch)
            except Exception as e:  # noqa: BLE001 — a bad batch must not kill the loop
                print(f"[main] next_batch failed at epoch {epoch} ({e!r}); skipping tick")
                await asyncio.sleep(interval)
                continue
            for signed in batch:
                try:
                    # Raw signals persist regardless of pipeline (sensor history
                    # tool); signature stored at rest for custody re-verification
                    # (C3-toolchat-v1).
                    # DEMO BYPASS (temporary, reversible): config.demo_skip_hmac.
                    if settings.demo_skip_hmac or \
                            verify_payload(signed.payload, signed.signature):
                        S.store.save_reading(signed.payload,
                                             signature=signed.signature)
                    tick: Optional[TickResult] = None
                    # Integration: hall replay presses are readings-only — they
                    # never enter the LLM pipeline (echo tick stores them CLEAR).
                    # And at REST (heartbeat) nothing goes through the LLM
                    # pipeline either: the dashboard just moves, no advisories,
                    # no idle LLM cost. The pipeline engages when the operator
                    # injects a fault (arc_mode == "fault").
                    _mid = str(signed.payload.get("machine_id", ""))
                    _arc = getattr(S.telemetry, "arc_mode", "fault")
                    if (S.pipeline is not None and _mid not in REPLAY_ONLY_IDS
                            and _arc == "fault"):
                        try:
                            tick = await S.pipeline.process_reading(signed)
                        except Exception as e:  # noqa: BLE001 — contract says never raise, belt+braces
                            print(f"[main] process_reading blew up ({e!r}); echoing")
                    if tick is None:
                        tick = _echo_tick(signed)
                        S.store.save_tick(tick)  # pipeline persists its own ticks
                    S.hub.publish("tick", tick.model_dump())
                    S.store.log_event("tick", {"machine_id": tick.machine_id,
                                               "epoch": tick.epoch,
                                               "risk": tick.triage.risk.value if tick.triage else "REJECTED"})
                    if tick.advisory is not None:
                        S.hub.publish("advisory", tick.advisory.model_dump())
                        S.store.log_event("advisory", {"id": tick.advisory.id,
                                                       "severity": tick.advisory.severity.value})
                        try:  # operator app: route the alert (notify/intervention)
                            S.interventions.on_advisory(tick.advisory)
                        except Exception as e:  # noqa: BLE001 — never kill the loop
                            print(f"[main] on_advisory failed ({e!r}); continuing")
                except Exception as e:  # noqa: BLE001 — survive anything, the show must go on
                    print(f"[main] tick handling failed at epoch {epoch} ({e!r}); continuing")
            if epoch % 5 == 0:
                try:
                    bs = await _boss_summary()
                    S.hub.publish("boss", bs.model_dump())
                    S.store.log_event("boss", {"text": bs.text[:200]})
                except Exception as e:  # noqa: BLE001
                    print(f"[main] boss summary failed: {e!r}")
            await asyncio.sleep(interval)
    finally:
        S.running = False
        print("[main] loop stopped")


# ===================================================================== app
app = FastAPI(title="PRAETOR agent backend", version="0.1.0")

# integration (CureWatch UI): vendor JS, runtime and assets under /static/
app.mount("/static", StaticFiles(directory=_STATIC), name="static")

_frontend = os.environ.get("FRONTEND_URL", "").strip()
_origins = {o for o in (_frontend, "http://localhost:3000", "http://127.0.0.1:3000") if o}
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    stream: bool = True
    mode: str = "deep"  # "deep" = tool-calling agent · "quick" = fast lane
    machine_id: Optional[str] = None  # scopes the quick lane's context


class ReasonBody(BaseModel):
    reason: str = ""


class OperatorBody(BaseModel):
    operator_id: str


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html", media_type="text/html")


@app.get("/operator")
async def operator_page() -> FileResponse:
    """Operator app (notification → machine view → chat → take charge → VLM)."""
    return FileResponse(_STATIC / "operator.html", media_type="text/html")


@app.get("/api/operators")
async def operators() -> dict:
    return {"operators": list(OPERATORS.values()), "assignments": ASSIGNMENTS}


@app.get("/api/factory/state")
async def factory_state() -> dict:
    """Initial-state contract for the operator app and the 3D factory map."""
    return build_factory_state(S.store, S.interventions,
                               mode="mock" if S.client.is_mock else "live",
                               epoch=S.epoch)


@app.post("/api/machine/{machine_id}/take_charge")
async def machine_take_charge(machine_id: str, body: OperatorBody) -> dict:
    try:
        iv = await S.interventions.take_charge(machine_id, body.operator_id)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return iv.model_dump()


@app.post("/api/machine/{machine_id}/repair_done")
async def machine_repair_done(machine_id: str, body: OperatorBody) -> dict:
    try:
        iv = await S.interventions.repair_done(machine_id, body.operator_id)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return iv.model_dump()


@app.get("/api/health")
async def health() -> dict:
    return {
        "mode": "mock" if S.client.is_mock else "live",
        "models": {
            "fast": settings.model_fast,
            "reasoning": settings.model_reasoning,
            "omni": settings.model_omni,
        },
        "pipeline_available": PIPELINE_AVAILABLE,
        "chat_agent": "tool-calling" if S.tool_operator is not None else "evidence-block",
        "loop_running": S.running,
        "epoch": S.epoch,
    }


@app.post("/api/loop/start")
async def loop_start(interval: float = 5.0, reset: int = 0) -> dict:
    if S.telemetry is None:
        raise HTTPException(status_code=503, detail="Telemetry source unavailable — "
                            "demo datasets missing from pinn/data/ (see pinn/data/README.md).")
    if S.running:
        return {"status": "already-running", "epoch": S.epoch}
    if reset:
        S.epoch = 0
    interval = max(0.5, min(interval, 60.0))
    S.loop_task = asyncio.create_task(_run_loop(interval))
    return {"status": "started", "interval": interval, "epoch": S.epoch,
            "mode": "pipeline" if PIPELINE_AVAILABLE else "triage-echo"}


@app.post("/api/loop/stop")
async def loop_stop() -> dict:
    S.running = False
    if S.loop_task:
        S.loop_task.cancel()
        S.loop_task = None
    return {"status": "stopped", "epoch": S.epoch}


# ---- integration: live scenario control (fault injection / reset) ----------
def _ensure_loop(interval: float = 2.5) -> None:
    """Start the always-on heartbeat loop if it isn't running (dashboard lives)."""
    if S.telemetry is None or S.running:
        return
    interval = max(0.5, min(interval, 60.0))
    S.loop_task = asyncio.create_task(_run_loop(interval))


@app.post("/api/scenario/fault")
async def scenario_fault(kind: str = "bearing") -> dict:
    """Inject a fault on the hero press RC-07 (drives the real degradation arc)."""
    if S.telemetry is None:
        raise HTTPException(503, "Telemetry source unavailable.")
    S.telemetry.set_scenario("fault", kind=kind, epoch=S.epoch)
    _ensure_loop()
    return {"status": "fault", "kind": S.telemetry.fault_kind, "epoch": S.epoch}


@app.post("/api/scenario/reset")
async def scenario_reset() -> dict:
    """Back to a healthy, gently-moving heartbeat — every machine healthy again."""
    if S.telemetry is not None:
        S.telemetry.set_scenario("heartbeat")
    # Clear the open advisories so the plant reads healthy after a reset.
    try:
        for a in S.store.get_advisories(status="pending", limit=100):
            S.operator.handle_override(a.id, "accepted",
                                       "Auto-cleared on Reset to Normal", S.store)
    except Exception as e:  # noqa: BLE001
        print(f"[main] reset advisory clear failed ({e!r})")
    _ensure_loop()
    return {"status": "heartbeat", "epoch": S.epoch}


@app.get("/api/history")
async def history(limit: int = 40) -> dict:
    """Decision audit trail for the site manager: what the agent PROPOSED and
    what the operator DECIDED, newest first, in plain relative time."""
    from .operator_flow import _ago  # relative-time helper (no epochs)

    now = time.time()
    out: list[dict] = []
    for ov in S.store.get_overrides(limit=limit):
        adv = S.store.get_advisory(ov.advisory_id)
        out.append({
            "at": ov.at,
            "ago": _ago(ov.at, now),
            "machine_id": adv.machine_id if adv else "?",
            "severity": adv.severity.value if adv else "?",
            "proposed_title": adv.title if adv else "(advisory not found)",
            "proposed_action": adv.recommended_action if adv else "",
            "decision": ov.decision,               # "accepted" | "overridden"
            "reason": ov.reason,
        })
    return {"entries": out, "count": len(out), "at": now}


@app.on_event("startup")
async def _boot_heartbeat() -> None:
    """A real factory dashboard is never idle — start the healthy heartbeat so
    CP-07's PINN state ticks live from the first page load."""
    _ensure_loop()


@app.get("/api/stream")
async def stream():
    q = S.hub.subscribe()

    async def gen():
        yield _sse("hello", {"mode": "mock" if S.client.is_mock else "live",
                             "epoch": S.epoch, "loop_running": S.running,
                             "pipeline_available": PIPELINE_AVAILABLE})
        try:
            while True:
                try:
                    kind, data = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield _sse(kind, data)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            S.hub.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/advisories")
async def advisories(status: Optional[str] = None, limit: int = 50) -> list[dict]:
    return [a.model_dump() for a in S.store.get_advisories(status=status, limit=limit)]


@app.post("/api/advisory/{advisory_id}/accept")
async def advisory_accept(advisory_id: str, body: Optional[ReasonBody] = None) -> dict:
    adv = S.operator.handle_override(advisory_id, "accepted",
                                     body.reason if body else "", S.store)
    if adv is None:
        raise HTTPException(404, f"unknown advisory {advisory_id}")
    S.hub.publish("override", adv.model_dump())
    return adv.model_dump()


@app.post("/api/advisory/{advisory_id}/override")
async def advisory_override(advisory_id: str, body: ReasonBody) -> dict:
    adv = S.operator.handle_override(advisory_id, "overridden", body.reason, S.store)
    if adv is None:
        raise HTTPException(404, f"unknown advisory {advisory_id}")
    S.hub.publish("override", adv.model_dump())
    return adv.model_dump()


@app.post("/api/chat")
async def chat(req: ChatRequest):
    # ------------------------------------------------------------------
    # INTEGRATION REPAIR (2026-07-05, CureWatch merge): the committed
    # handler body was TRUNCATED at "ra" (mid-`raise`) — the whole chat
    # path was missing from the repo even though CLAUDE.md documents it
    # E2E-green locally (push lost the body). Minimal faithful
    # reconstruction so the merged system works: 422 on empty question,
    # quick lane via operator_flow.quick_answer, deep lane via the
    # tool-calling operator, non-stream JSON {"answer": ...}.
    # @owner: please graft your full streaming version back on top —
    # the CureWatch UI only relies on stream:false + {"answer"}.
    # ------------------------------------------------------------------
    if not req.question.strip():
        raise HTTPException(422, "empty question")
    # DEFAULT = the DEEP tool-calling operator: it actually calls retrieval
    # tools (run_diagnostic / sensor history / machine limits / drift /
    # integrity) per question and answers from what it consulted, scoped to
    # the machine the operator is looking at (req.machine_id). The quick lane
    # (single fixed-context call) is kept only as an explicit opt-in / fallback
    # — it was giving the same answer to every question (see git log).
    if S.tool_operator is not None and req.mode != "quick":
        turn, provenance = await S.tool_operator.answer(
            req.question, machine_hint=req.machine_id)
        data = turn.model_dump() if hasattr(turn, "model_dump") else dict(turn)
        answer = data.get("content", "")
        # Display consistency: the store id is RC-07 but the operator's screen
        # shows the hero card CP-07 — speak the name they see.
        if str(req.machine_id or "").upper() == "CP-07":
            answer = answer.replace("RC-07", "CP-07")
            data["content"] = answer
        return {"answer": answer, "mode": "deep",
                "provenance": provenance, "turn": data}
    turn, provenance = await quick_answer(req.question, req.machine_id,
                                          S.store, S.client, S.interventions)
    return {"answer": turn.get("content", ""), "mode": "quick",
            "provenance": provenance, "turn": turn}

@app.get("/api/plant")
async def plant() -> dict:
    """Boss summary on demand (restored after the operator-app rework — audit fix)."""
    bs = await _boss_summary()
    S.hub.publish("boss", bs.model_dump())
    return bs.model_dump()
