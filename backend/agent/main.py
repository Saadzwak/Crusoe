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
  GET  /api/plant           boss summary on demand

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
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .config import settings
from .crusoe_client import get_client
from .knowledge import Knowledge
from .operator import OperatorAgent
from .schemas import (PinnReading, PlantSummary, SignedReading, TickResult,
                      TriageResult)
from .state_store import StateStore
from .telemetry_source import TelemetrySource, note_risk
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


# ===================================================================== loop
def _echo_tick(signed: SignedReading) -> TickResult:
    """Degraded-mode triage: HMAC verify + keyword risk from the PINN note."""
    ok = verify_payload(signed.payload, signed.signature)
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
                    if verify_payload(signed.payload, signed.signature):
                        S.store.save_reading(signed.payload,
                                             signature=signed.signature)
                    tick: Optional[TickResult] = None
                    if S.pipeline is not None:
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


class ReasonBody(BaseModel):
    reason: str = ""


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html", media_type="text/html")


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
    if not req.question.strip():
        raise HTTPException(400, "empty question")

    # ---------------------------------------------------------- legacy agent
    if S.tool_operator is None:
        if not req.stream:
            turn = await S.operator.answer(req.question, S.store, S.knowledge,
                                           S.client)
            return JSONResponse(turn.model_dump())

        async def gen_legacy():
            try:
                async for ev in S.operator.stream_answer(req.question, S.store,
                                                         S.knowledge, S.client):
                    if ev.get("type") == "token":
                        yield _sse("token", {"text": ev["text"]})
                    elif ev.get("type") == "turn":
                        yield _sse("turn", ev["turn"])
            except Exception as e:  # noqa: BLE001 — keep the console alive
                yield _sse("error", {"detail": repr(e)})
            yield _sse("done", {})

        return StreamingResponse(gen_legacy(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    # ---------------------------------------------- tool-calling agent (C3)
    if not req.stream:
        turn, provenance = await S.tool_operator.answer(req.question)
        return JSONResponse({**turn.model_dump(), "provenance": provenance})

    async def gen_tools():
        try:
            async for ev in S.tool_operator.stream_events(req.question):
                kind = ev.get("type")
                if kind == "tool_call":
                    data = {"name": ev["name"], "params": ev["params"]}
                    S.hub.publish("tool_call", data)  # /api/stream watchers too
                    yield _sse("tool_call", data)
                elif kind == "tool_result":
                    data = {"name": ev["name"], "ok": ev["ok"],
                            "summary": ev.get("summary", "")}
                    S.hub.publish("tool_result", data)
                    yield _sse("tool_result", data)
                elif kind == "token":
                    yield _sse("token", {"text": ev["text"]})
                elif kind == "turn":
                    yield _sse("turn", ev["turn"])
        except Exception as e:  # noqa: BLE001 — keep the console alive
            yield _sse("error", {"detail": repr(e)})
        yield _sse("done", {})

    return StreamingResponse(gen_tools(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.get("/api/plant")
async def plant() -> dict:
    bs = await _boss_summary()
    S.hub.publish("boss", bs.model_dump())
    return bs.model_dump()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.agent.main:app", host="0.0.0.0",
                port=settings.backend_port, log_level="info")
