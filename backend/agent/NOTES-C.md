# NOTES-C — Builder C build notes (tool-calling operator agent)

Deliverable: the Line-LLM Operator Q&A Agent per the team spec — a REAL
tool-calling agent (LangChain ChatOpenAI on Crusoe, Nemotron Ultra 550B,
temperature 0.1) with a deterministic evidence-backed mock path. No Builder A
file and no shared file was edited (triage.py's CAUSAL_MATRIX is *imported*,
guarded, in operator_tools.py).

## Files
- `operator_tools.py` (NEW) — `OperatorToolbox(store, knowledge)`: 5 real tools
  + 2 stubs, every result `{"tool", "params", "at", "ok", ...}`; `MACHINE_SPECS`
  for RC-07 / CL-03 / MX-02 (operating ranges, alert/trip limits, human-risk,
  role, causal-matrix context).
- `operator_agent.py` (NEW) — `ToolCallingOperator(store, knowledge)`:
  live bind_tools loop (max 6 tool calls / 3 LLM rounds → forced final via
  `llm.astream`), mock deterministic plan, programmatic Data Provenance,
  `stream_events()` + `answer()`.
- `state_store.py` (EDIT, heal procedure) — `save_reading(payload,
  signature=None)` stores signature + exact payload JSON at rest (nullable
  columns, `ALTER TABLE` migration for old DBs); new
  `get_readings_with_signatures(machine_id, limit=50)`. Marker `C3-custody-v1`.
- `main.py` (EDIT, heal) — ServiceState lazily builds `ToolCallingOperator`
  (try/except → legacy OperatorAgent fallback); loop passes
  `signature=signed.signature` into `save_reading`; `/api/chat` streams
  tool_call/tool_result/token/turn and republishes tool events on the hub;
  `/api/health` gains `"chat_agent": "tool-calling"|"evidence-block"`.
  Override endpoints and operator.py untouched. Marker `C3-toolchat-v1`.
- `static/index.html` (EDIT, heal) — chat bubble shows live
  `⚙ tool(params) → ok/error · summary` lines while the agent works, and the
  Data Provenance block renders as a collapsible audit trail. Vanilla JS, no
  CDN, everything else intact. Marker `C3-toolchat-v1`.
- `tests/test_operator_agent_mock.py` (NEW) — 12 checks, all green.

## SSE contract (frontend teammate)
`POST /api/chat` body `{"question": str, "stream": true}` → `text/event-stream`,
per turn, in order:

```
event: tool_call    data: {"name": str, "params": {..}}          (0..6 times)
event: tool_result  data: {"name": str, "ok": bool, "summary": str}   (pairs FIFO with tool_call)
event: token        data: {"text": str}                          (many — final answer text)
event: turn         data: {role, content, citations[], at, provenance[]}
event: done         data: {}
event: error        data: {"detail": str}   (only on failure, then done)
```
- `provenance[i] = {"tool", "params", "at" (ISO-8601 UTC), "ok", "summary"}`.
- `turn.content` ALWAYS ends with a `Data Provenance:` section (one `- tool(params)
  @ time → ok/ERROR` line per executed tool) — built in code, never by the model.
- `citations` = executed tool names + any bracketed markers found (`[site_dossier p.4]`,
  `[run_diagnostic RC-07]`).
- `{"stream": false}` → plain JSON `{role, content, citations, at, provenance}`.
- `GET /api/stream` ALSO carries `tool_call` / `tool_result` events (same payloads)
  while any chat turn runs, so the main console shows the agent working.

## Decisions / deviations
1. **Custody at rest**: `save_reading` stores the *exact* payload JSON next to
   the signature; `verify_custody_chain` re-signs `json.loads(payload_json)`
   (canonical serialization is key-order independent). Rebuilding the payload
   from the scalar columns risked float/format drift — the verbatim copy makes
   re-verification exact. Backwards compatible: unsigned rows (old callers,
   Builder B's test) are reported as "pre-custody, skipped", and a machine with
   zero signed rows → `ok:false "no custody records"`.
2. **Knowledge is prompt context, not a tool**: the spec fixes the 7-tool kit,
   so dossier economics enter as a compact SITE CONTEXT block in the system
   prompt (live) / a cost line on cost-keyword questions (mock), always with
   `[site_dossier p.N]` tags. Keeps "cite dossier tags when knowledge used"
   without inventing an 8th tool.
3. **Context separation** is enforced structurally: `run_diagnostic` returns
   separate `realtime` vs `background` objects, and the system prompt (rule 2)
   tells the model to keep them apart unless asked.
4. **Live final round**: if the model stops calling tools naturally, its final
   content is chunk-streamed (no second LLM call); only when the 6-call/3-round
   budget is exhausted do we force a final `llm.astream` round (with a crude
   `<think>` gate on the stream, thinking already disabled via
   `chat_template_kwargs` per docs/CRUSOE.md).
5. **Fallback ladder**: mock_mode → mock path directly; live path exception
   (anywhere) → printed one-liner + full mock re-run, which still ends with a
   `turn` event. Verified with a fake key: `APIConnectionError` → clean mock
   answer. The demo cannot die on stage. NOTE: live Crusoe egress is blocked
   from this sandbox — live path is structure-verified only (bind_tools schema,
   extra_body, fallback); please run a real live smoke on Windows
   (`scripts/crusoe_sanity.py` + one chat turn with the key loaded).
6. **`/api/chat` response shape**: additive only (`provenance` key, new SSE
   event names). `test_service_mock.py` needed NO changes (it tests
   OperatorAgent directly) — reran green (14 checks, with Builder A pipeline);
   `test_pipeline_mock.py` also green.
7. **MACHINE_SPECS numbers** were calibrated against the actual telemetry maps
   so the demo arc reads true: RC-07 vibration alert 4.5 / trip 7.0 → epochs
   10–11 (4.86 / 5.78 mm/s) are named as excursions while mould temp
   (150–185 °C op, 195 trip) stays in range; CL-03 thermal_margin floor 8 K per
   AI4I HDF logic; MX-02 torque/wear bands per AI4I PWF/OSF.
8. **Provenance is authoritative**: any model-written "Data Provenance" section
   is stripped and replaced by the programmatic one; zero-tool turns get an
   explicit "(no tools executed — answer is unbacked)" line.

## Mount-quirk compliance
All three existing-file edits went through write-new → `cp` → `ast.parse` /
marker grep from bash, and markers were re-checked from BOTH the Windows and
Linux views afterwards (`C3-custody-v1` ×4, `C3-toolchat-v1` main ×3 /
index ×4). No leftover `.heal_*` files.
