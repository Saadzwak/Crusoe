"""Line-LLM Operator Q&A — REAL tool-calling agent (Builder C) — C3-agent-v1.

Two execution paths, one contract:

LIVE  — langchain_openai.ChatOpenAI against Crusoe (settings.model_reasoning,
        temperature 0.1, thinking disabled per docs/CRUSOE.md), tools bound
        via .bind_tools(); loop: invoke → execute tool_calls locally
        (OperatorToolbox) → append ToolMessage → re-invoke. Budget: max 6
        tool calls / 3 LLM rounds, then a forced final answer streamed with
        llm.astream. ANY failure falls back to the mock path — the demo
        never dies.

MOCK  — deterministic plan (run_diagnostic + get_sensor_history +
        analyze_drift, plus custody/spec/stub tools when the question asks);
        the tools run FOR REAL against the local store, and the answer is
        composed from their actual outputs — evidence-backed even offline.

BOTH  — every executed tool lands in a provenance list, and the final
        answer ALWAYS ends with a "Data Provenance" section built
        programmatically from that list (never trusted to the model).

Event stream (stream_events) — the SSE contract for the frontend:
  {"type":"tool_call",   "name": str, "params": dict}
  {"type":"tool_result", "name": str, "ok": bool, "summary": str}
  {"type":"token",       "text": str}                       # final answer
  {"type":"turn",        "turn": OperatorTurn dump + {"provenance": [...]}}
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncIterator, Optional

from .config import settings, thinking_off_extra_body
from .crusoe_client import strip_thinking
from .operator_tools import OperatorToolbox
from .schemas import OperatorTurn

MAX_TOOL_CALLS = 6
MAX_LLM_ROUNDS = 3
_TOKEN_CHUNK = 24

_CITE_RE = re.compile(r"\[[^\[\]\n]{2,80}\]")
_MACHINES = {"RC-07": "Curing", "CL-03": "Calendering", "MX-02": "Mixing",
             "CP-01": "Curing", "CP-03": "Curing", "CP-10": "Curing"}
# The UI hero card CP-07 is the physical curing press RC-07 in the store;
# CP-01/03/10 are stored under their own ids (real FD001 replays). Everything
# else in the hall is display-only ambiance with no stored telemetry.
_UI_ALIAS = {"CP-07": "RC-07"}

# --------------------------------------------------------------- system prompt
# Spec voice: evidence-backed, context separation, stub honesty, provenance,
# propose-never-command (SAFE rules echoed from prompts.py without touching it).
_SYSTEM_TEMPLATE = """You are the PRAETOR Line-LLM operator diagnosis agent on the Michelin Roanne UHP tyre line (C3M process — machines: RC-07 curing press, the bottleneck; CL-03 calender; MX-02 Banbury mixer). You interact with line operators on demand to diagnose machinery state, analyze anomalies and verify data lineage. Use your tools to gather evidence BEFORE answering.

HARD RULES
1. Evidence only — every number and factual claim must come from a tool result returned in THIS conversation. No tool result, no claim. If a tool returns ok:false or data is missing, report that transparently instead of guessing.
2. Context separation — background epoch-triage state (ticks, advisories, overrides under "background") and real-time tool outputs ("realtime") are different sources: do not blend them into one narrative unless the operator asks; say which is which.
3. Stubs — get_pinn_reconstruction and get_camera_frame are not integrated: if they answer "physical subsystem not integrated", relay exactly that; never invent a reconstruction or a camera frame.
4. You propose, never command — recommendations are options with trade-offs; the operator decides.
5. Ground every number in a tool result, but write for a shop-floor operator: everyday time ("just now", "a few minutes ago") — NEVER say "epoch", cycle counts or internal ids — and NO bracketed citations in the text.
6. ANSWER THE QUESTION THAT WAS ASKED — the shape follows the question, never a fixed template:
   - status / "why is it flagged?" → one-line verdict, then ONLY the readings that justify it.
   - trend / history → that signal's movement (from → to, how fast, still moving or settled). Other signals only if they explain it.
   - "what happens if…" / risk → the concrete consequences ahead (failure mode, what breaks, roughly when) from the causal matrix and RUL — NOT a re-dump of current readings.
   - spec / limits → the limits themselves and where the machine sits against them.
7. CONVERSATION MEMORY — if earlier turns of this conversation are provided, do not restate numbers you already gave unless they changed (then give old → new). Never repeat the same recommendation twice: if you already advised it, refer back in a few words ("still my advice") or say what changed.
8. Under ~110 words. Bullets ("- ") only when listing 2+ distinct facts — a simple question deserves a plain sentence. End with a recommendation ONLY if the operator asked what to do, or a limit is being crossed right now and no advisory is pending; one line, offered as a choice.
9. Do NOT write a "Sources" or "Data Provenance" section yourself — the platform adds the list of tools you consulted automatically.

SITE CONTEXT (static dossier extracts — background reference, cite tags when used):
{site_context}"""


def _fmt_params(params: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in (params or {}).items())


class ToolCallingOperator:
    """On-demand, conversational, strictly evidence-backed diagnosis agent."""

    def __init__(self, store: Any, knowledge: Any = None) -> None:
        self.toolbox = OperatorToolbox(store, knowledge)
        self.knowledge = knowledge
        self._site_context = self._build_site_context()

    # ------------------------------------------------------------ site frame
    def _build_site_context(self) -> str:
        if self.knowledge is None:
            return "(no site dossier loaded)"
        try:
            snippets = self.knowledge.downtime_context()[:2]
            lines = [f"{cit} {text[:260]}" for text, cit in snippets]
            return "\n".join(lines) if lines else "(no site dossier loaded)"
        except Exception:  # noqa: BLE001
            return "(site dossier unavailable)"

    def _system(self) -> str:
        return _SYSTEM_TEMPLATE.format(site_context=self._site_context)

    # ------------------------------------------------------------- execution
    def _execute(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        return self.toolbox.call(name, params or {})

    @staticmethod
    def _summarize(result: dict[str, Any]) -> str:
        if not result.get("ok"):
            return str(result.get("error", "error"))[:160]
        tool = result.get("tool", "")
        if tool == "get_sensor_history":
            top = ""
            for name, s in (result.get("signal_summary") or {}).items():
                if s.get("trend") != "stable":
                    top = f"; {name} {s['trend']} {s['change_pct']:+.1f}%"
                    break
            return f"{result.get('count')} recent readings{top}"
        if tool == "run_diagnostic":
            hits = (result.get("realtime") or {}).get("out_of_limits") or []
            named = f" — {hits[0]['signal']} {hits[0]['kind']}" if hits else ""
            return f"risk {result.get('risk')}; {len(hits)} signal(s) out of limits{named}"
        if tool in ("verify_custody_chain", "analyze_drift"):
            return str(result.get("verdict", "done"))[:160]
        if tool == "get_machine_spec":
            return (f"{result.get('type', '?')}; {len(result.get('signals') or {})} "
                    f"signal specs; tags {', '.join(result.get('causal_tags') or [])}")
        return "done"

    def _prov_entry(self, result: dict[str, Any]) -> dict[str, Any]:
        return {"tool": result.get("tool", "?"), "params": result.get("params", {}),
                "at": result.get("at", ""), "ok": bool(result.get("ok")),
                "summary": self._summarize(result)}

    # ------------------------------------------------------------ provenance
    @staticmethod
    def _finalize(text: str, provenance: list[dict]) -> str:
        """Append the programmatic Data Provenance section (never the model's)."""
        text = (text or "").strip()
        cut = text.find("Data Provenance")
        if cut > 0:
            text = text[:cut].rstrip(" \n-*#:")
        if not text:
            text = ("I couldn't put together an answer this turn — see the sources "
                    "I checked below.")
        # Operator-facing sources line (clean, plain-English names of the real
        # tools consulted). The full machine-readable trail stays on the turn's
        # `provenance` field for the audit log, not in the chat bubble.
        pretty = {"run_diagnostic": "live diagnostic",
                  "get_sensor_history": "sensor history",
                  "analyze_drift": "drift analysis",
                  "get_machine_spec": "machine limits",
                  "verify_custody_chain": "data-integrity check",
                  "get_pinn_reconstruction": "PINN reconstruction",
                  "get_camera_frame": "camera"}
        names: list[str] = []
        for p in provenance:
            nm = pretty.get(p["tool"], p["tool"])
            if p["ok"] and nm not in names:
                names.append(nm)
        if names:
            return text + "\n\n_Checked: " + ", ".join(names) + "._"
        return text

    @staticmethod
    def _citations(content: str, provenance: list[dict]) -> list[str]:
        seen: list[str] = []
        for p in provenance:
            if p["tool"] not in seen:
                seen.append(p["tool"])
        for m in _CITE_RE.findall(content or ""):
            if m not in seen:
                seen.append(m)
        return seen

    def _turn_event(self, text: str, provenance: list[dict]) -> dict[str, Any]:
        content = self._finalize(text, provenance)
        turn = OperatorTurn(role="agent", content=content,
                            citations=self._citations(content, provenance))
        return {"type": "turn", "turn": {**turn.model_dump(), "provenance": provenance}}

    # ---------------------------------------------------------------- public
    async def answer(self, question: str,
                     machine_hint: Optional[str] = None,
                     history: Optional[list[dict]] = None
                     ) -> tuple[OperatorTurn, list[dict]]:
        """Non-streaming variant: returns (OperatorTurn, provenance).

        machine_hint = the machine the operator is currently looking at, so
        "how is it doing?" resolves to that press unless the question names
        another one. history = prior turns of THIS conversation
        ([{role: user|agent, content}] oldest→newest) so follow-ups don't
        restate what was already said.
        """
        turn_payload: Optional[dict] = None
        async for ev in self.stream_events(question, machine_hint, history):
            if ev.get("type") == "turn":
                turn_payload = ev["turn"]
        assert turn_payload is not None  # stream always ends with a turn
        provenance = turn_payload.get("provenance", [])
        turn = OperatorTurn.model_validate(
            {k: v for k, v in turn_payload.items() if k != "provenance"})
        return turn, provenance

    async def stream_events(self, question: str,
                            machine_hint: Optional[str] = None,
                            history: Optional[list[dict]] = None
                            ) -> AsyncIterator[dict]:
        """Agent run as an event stream (see module docstring for the contract)."""
        question = (question or "").strip()
        hint = self._resolve_hint(machine_hint)
        if settings.mock_mode:
            async for ev in self._stream_mock(question, hint):
                yield ev
            return
        try:
            async for ev in self._stream_live(question, hint, history):
                yield ev
            return
        except Exception as e:  # noqa: BLE001 — live path must never kill the demo
            print(f"[operator_agent] live path failed ({type(e).__name__}: {e}); "
                  "falling back to mock plan")
        async for ev in self._stream_mock(question, hint):
            yield ev

    @staticmethod
    def _resolve_hint(machine_hint: Optional[str]) -> str:
        h = str(machine_hint or "").upper().strip()
        h = _UI_ALIAS.get(h, h)
        return h if h in _MACHINES else "RC-07"

    # ------------------------------------------------------------- LIVE path
    def _lc_tools(self):
        """@tool wrappers around the toolbox (schemas for bind_tools)."""
        from langchain_core.tools import tool  # lazy — langchain optional at import
        tb = self.toolbox

        @tool
        def get_sensor_history(machine_id: str, limit: int = 20) -> dict:
            """Archived telemetry trends for one machine (RC-07, CL-03 or MX-02):
            per-signal first/last/min/max/%change/trend plus the last 5 raw
            readings. Use for questions about history, trends or past values."""
            return tb.get_sensor_history(machine_id, limit)

        @tool
        def get_machine_spec(machine_id: str) -> dict:
            """Design boundaries for one machine: operating ranges, alert/trip
            limits per signal, its role in the line, human-risk notes and the
            causal matrix (probable cause / downstream effect / remedy) for its
            typical failure modes. Use before judging whether a value is normal."""
            return tb.get_machine_spec(machine_id)

        @tool
        def run_diagnostic(machine_id: str) -> dict:
            """Real-time health check of one machine: latest signals vs spec
            limits (names any out-of-limit signal), PINN state, background
            triage risk, recent advisories and operator overrides, plus a
            WHERE/WHAT/WHY/HOW verdict. Start here for 'how is X doing?'."""
            return tb.run_diagnostic(machine_id)

        @tool
        def verify_custody_chain(machine_id: str, epoch: Optional[int] = None) -> dict:
            """Data-lineage audit: re-verify the HMAC signature of every stored
            reading for a machine (optionally one epoch) and name any broken
            link. Use for questions about data integrity, tampering or trust."""
            return tb.verify_custody_chain(machine_id, epoch)

        @tool
        def analyze_drift(machine_id: str, metric: Optional[str] = None) -> dict:
            """Deviation of the recent window vs the baseline window per signal
            (%shift, drifting/chronic/stable — chronic means bad-but-flat, which
            is NOT drift). Use for 'is X drifting / degrading / moving?'."""
            return tb.analyze_drift(machine_id, metric)

        @tool
        def get_pinn_reconstruction(machine_id: str) -> dict:
            """Full-field PINN physics reconstruction for a machine. NOTE: stub —
            the physical subsystem is not integrated in this layer yet."""
            return tb.get_pinn_reconstruction(machine_id)

        @tool
        def get_camera_frame(camera_id: str) -> dict:
            """Latest shop-floor camera frame for a camera id. NOTE: stub — the
            physical subsystem is not integrated in this layer yet."""
            return tb.get_camera_frame(camera_id)

        return [get_sensor_history, get_machine_spec, run_diagnostic,
                verify_custody_chain, analyze_drift, get_pinn_reconstruction,
                get_camera_frame]

    async def _stream_live(self, question: str, hint: str = "RC-07",
                           history: Optional[list[dict]] = None
                           ) -> AsyncIterator[dict]:
        from langchain_core.messages import (AIMessage, HumanMessage,
                                             SystemMessage, ToolMessage)
        from langchain_openai import ChatOpenAI

        # Chat is conversational — latency matters more than deep reasoning.
        # model_chat (default: the fast tier) keeps tool-calling but answers in
        # seconds; CRUSOE_MODEL_CHAT pins a different model without code change.
        _model = settings.model_chat or settings.model_fast
        kwargs: dict[str, Any] = dict(
            model=_model,
            base_url=settings.base_url,
            api_key=settings.api_key,
            temperature=0.1,          # spec: deterministic diagnosis voice
            max_tokens=500,
            timeout=45,
            max_retries=1,
        )
        extra = thinking_off_extra_body(_model)
        if extra:
            kwargs["extra_body"] = extra
        llm = ChatOpenAI(**kwargs)
        bound = llm.bind_tools(self._lc_tools())

        provenance: list[dict] = []
        _dept = _MACHINES.get(hint, "Curing")
        framing = (f"The operator is currently looking at machine {hint} "
                   f"({_dept}). Unless they name another machine, 'it'/'this "
                   f"machine' means {hint}. Call your tools with that machine_id.")
        messages: list[Any] = [SystemMessage(content=self._system()),
                               SystemMessage(content=framing)]
        # Prior turns of THIS conversation (rule 7: don't restate, don't
        # re-recommend). UI names the hero press CP-07 — translate back so the
        # model reasons about the store id it will query.
        for h in (history or [])[-8:]:
            txt = str(h.get("content") or h.get("text") or "").strip()[:700]
            if not txt:
                continue
            for ui, real in _UI_ALIAS.items():
                txt = txt.replace(ui, real)
            role = str(h.get("role", "user")).lower()
            messages.append(AIMessage(content=txt)
                            if role in ("agent", "assistant", "ai")
                            else HumanMessage(content=txt))
        messages.append(HumanMessage(content=question))
        total_calls = 0
        final_text: Optional[str] = None

        for _round in range(MAX_LLM_ROUNDS):
            ai = await bound.ainvoke(messages)
            messages.append(ai)
            calls = list(getattr(ai, "tool_calls", None) or [])
            if not calls:
                final_text = strip_thinking(self._content_str(ai))
                break
            for tc in calls:
                name = tc.get("name", "?")
                params = dict(tc.get("args") or {})
                tcid = tc.get("id") or f"call-{total_calls}"
                if total_calls >= MAX_TOOL_CALLS:
                    messages.append(ToolMessage(
                        content=json.dumps({"ok": False,
                                            "error": "tool budget exhausted — answer now"}),
                        tool_call_id=tcid))
                    continue
                yield {"type": "tool_call", "name": name, "params": params}
                result = self._execute(name, params)
                prov = self._prov_entry(result)
                provenance.append(prov)
                yield {"type": "tool_result", "name": name,
                       "ok": prov["ok"], "summary": prov["summary"]}
                messages.append(ToolMessage(
                    content=json.dumps(result, default=str)[:6000],
                    tool_call_id=tcid))
                total_calls += 1

        if final_text is None:
            # Budget exhausted with the model still asking for tools — force the
            # final answer and stream it token-by-token (llm.astream, no tools).
            messages.append(HumanMessage(content=(
                "Tool budget reached. Write your final answer NOW from the tool "
                "results above, following the answer rules.")))
            parts: list[str] = []
            inside_think = False
            async for chunk in llm.astream(messages):
                piece = self._content_str(chunk)
                if not piece:
                    continue
                if "<think>" in piece:
                    inside_think = True
                    continue
                if "</think>" in piece:
                    inside_think = False
                    continue
                if inside_think:
                    continue
                parts.append(piece)
                yield {"type": "token", "text": piece}
            final_text = strip_thinking("".join(parts))
        else:
            for i in range(0, len(final_text), _TOKEN_CHUNK):
                yield {"type": "token", "text": final_text[i: i + _TOKEN_CHUNK]}

        yield self._turn_event(final_text, provenance)

    @staticmethod
    def _content_str(msg: Any) -> str:
        c = getattr(msg, "content", "") or ""
        if isinstance(c, list):  # some providers return content blocks
            c = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
        return str(c)

    # ------------------------------------------------------------- MOCK path
    def _machine_from(self, question: str, hint: str = "RC-07") -> str:
        q = (question or "").upper()
        # an explicit machine named in the question always wins over the hint
        for mid in list(_MACHINES) + list(_UI_ALIAS):
            if mid in q:
                return _UI_ALIAS.get(mid, mid)
        for mid, dept in _MACHINES.items():
            if dept.upper() in q:
                return mid
        return hint or "RC-07"  # else the press the operator is looking at

    def _plan(self, question: str, hint: str = "RC-07"
              ) -> tuple[str, list[tuple[str, dict]]]:
        q = (question or "").lower()
        mid = self._machine_from(question, hint)
        plan: list[tuple[str, dict]] = [
            ("run_diagnostic", {"machine_id": mid}),
            ("get_sensor_history", {"machine_id": mid, "limit": 20}),
            ("analyze_drift", {"machine_id": mid}),
        ]
        if any(k in q for k in ("spec", "limit", "design", "envelope", "rating", "bound")):
            plan.insert(1, ("get_machine_spec", {"machine_id": mid}))
        if any(k in q for k in ("custody", "integr", "tamper", "lineage", "hmac",
                                "signature", "trust", "authentic")):
            plan.append(("verify_custody_chain", {"machine_id": mid}))
        if "camera" in q or "frame" in q or "visual" in q:
            plan.append(("get_camera_frame", {"camera_id": f"CAM-{mid}"}))
        if "reconstruct" in q or "pinn field" in q or "physics field" in q:
            plan.append(("get_pinn_reconstruction", {"machine_id": mid}))
        return mid, plan

    def _compose_mock(self, question: str, mid: str,
                      results: dict[str, dict]) -> str:
        lines: list[str] = []
        diag = results.get("run_diagnostic")
        hist = results.get("get_sensor_history")
        drift = results.get("analyze_drift")
        spec = results.get("get_machine_spec")
        custody = results.get("verify_custody_chain")

        if diag and diag.get("ok"):
            v = diag["verdict"]
            hits = (diag.get("realtime") or {}).get("out_of_limits") or []
            concl = f"{v['WHERE']} — {v['WHAT']}."
            lines.append(concl)
            lines.append(f"Why (real-time): {v['WHY']} [run_diagnostic {mid}]")
            if hits:
                h = hits[0]
                lines.append(f"Named excursion: {h['signal']} = {h['value']} is "
                             f"{h['kind']} (limit {h['limit']}) [run_diagnostic {mid}]")
            bg = diag.get("background") or {}
            if bg.get("advisories"):
                a = bg["advisories"][0]
                lines.append(f"Background triage (separate source): last advisory "
                             f"[advisory {a['id']}] {a['severity']} '{a['title']}' "
                             f"status={a['status']}.")
        elif diag:
            lines.append(f"Diagnostic unavailable: {diag.get('error')} [run_diagnostic {mid}]")

        if drift and drift.get("ok"):
            top = (drift.get("drifting") or [None])[0]
            if top:
                s = drift["signals"][top]
                lines.append(f"Trend: {drift['verdict']} — {top} moved "
                             f"{s['baseline_mean']} → {s['recent_mean']} "
                             f"({s['shift_pct']:+.1f}%) [analyze_drift {mid}]")
            else:
                lines.append(f"Trend: {drift['verdict']} [analyze_drift {mid}]")
        if hist and hist.get("ok"):
            name, s = next(iter((hist.get("signal_summary") or {}).items()),
                           (None, None))
            if name:
                lines.append(f"Recent history: {hist['count']} readings; "
                             f"{name} ranged {s['min']}–{s['max']}.")
        if spec and spec.get("ok"):
            lines.append(f"Spec context: {spec['type']} — "
                         f"{spec['human_risk']} [get_machine_spec {mid}]")
        if custody:
            lines.append(f"Data lineage: {custody.get('verdict', custody.get('error'))} "
                         f"[verify_custody_chain {mid}]")
        for stub_name in ("get_pinn_reconstruction", "get_camera_frame"):
            r = results.get(stub_name)
            if r:
                lines.append(f"{stub_name}: physical subsystem not integrated in "
                             "this layer yet — no data to show, as designed.")

        # dossier economics when the operator asks about cost/stops
        q = (question or "").lower()
        if self.knowledge is not None and any(
                k in q for k in ("cost", "€", "eur", "downtime", "stop", "loss")):
            try:
                text, cit = self.knowledge.downtime_context()[0]
                lines.append(f"Cost frame: {text[:180]} {cit}")
            except Exception:  # noqa: BLE001
                pass

        if diag and diag.get("ok"):
            lines.append(f"{diag['verdict']['HOW']} — your call.")
        return "\n".join(lines)

    async def _stream_mock(self, question: str, hint: str = "RC-07"
                           ) -> AsyncIterator[dict]:
        mid, plan = self._plan(question, hint)
        provenance: list[dict] = []
        results: dict[str, dict] = {}
        for name, params in plan:
            yield {"type": "tool_call", "name": name, "params": params}
            await asyncio.sleep(0.02)  # visible cadence in the console
            result = self._execute(name, params)
            results[name] = result
            prov = self._prov_entry(result)
            provenance.append(prov)
            yield {"type": "tool_result", "name": name,
                   "ok": prov["ok"], "summary": prov["summary"]}
        answer = self._compose_mock(question, mid, results)
        content = self._finalize(answer, provenance)
        for i in range(0, len(content), _TOKEN_CHUNK):
            await asyncio.sleep(0.01)
            yield {"type": "token", "text": content[i: i + _TOKEN_CHUNK]}
        turn = OperatorTurn(role="agent", content=content,
                            citations=self._citations(content, provenance))
        yield {"type": "turn",
               "turn": {**turn.model_dump(), "provenance": provenance}}


__all__ = ["ToolCallingOperator", "MAX_TOOL_CALLS", "MAX_LLM_ROUNDS"]
