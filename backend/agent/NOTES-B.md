# NOTES-B — Builder B build notes (service layer)

Friction log + heads-up for Builder A and the architect. No shared files were edited.

## 1. `StateStore.save_reading()` — one extra method beyond the contract
`TickResult` carries no `signals`/`pinn`, so nothing in the contract ever gets raw
sensor values into the store — yet `get_sensor_history()` must return "ticks w/
signals+pinn". Resolution: the service loop (`main.py`) persists every
HMAC-verified payload via `store.save_reading(payload)` into a `readings` table,
and `get_sensor_history()` serves from there. Contract methods are unchanged;
Builder A's duck-typed fake needs nothing new (the pipeline never calls it).

## 2. Mock tier2 substring triggers are fragile — format your prompts, A
`MockClient` flags CRITICAL when the prompt contains `"0.9"` or `"rul_cycles': 1"`.
Both match Python-repr dumps of perfectly healthy readings (`health_index=0.95`
→ contains `0.9`; `'rul_cycles': 152.0` starts with `'rul_cycles': 1`). If tier2
prompts embed raw `repr(reading.model_dump())`, healthy machines go CRITICAL in
mock mode — verified: early RC-07 readings (rul 152/140/128) trip it in repr
form. Mitigations on my side:
- telemetry keeps healthy `health_index` ≤ 0.89 and `_safe_round()` nudges any
  value whose repr would contain `"0.9"` (scanned epochs 0–15: zero hits);
- note text follows the trigger discipline exactly (healthy notes clean,
  anomalous notes contain `rising`/`anomal`/`critical`/`failure`/`imminent`).
Recommendation for A: build tier2 prompts from formatted summaries or
`json.dumps` (double quotes — doesn't match the mock's single-quote pattern),
never Python repr.

## 3. Degraded mode when `pipeline.py` is absent
`main.py` imports `backend.agent.pipeline` at module top in try/except
(`PIPELINE_AVAILABLE` flag). Without it the loop runs "triage-echo": HMAC verify
→ keyword triage from the PINN note (`telemetry_source.note_risk`) → tick
published + stored. The moment `pipeline.py` lands, a restart picks it up; no
code change needed. If `AdvisoryPipeline.__init__` or `process_reading` raises,
the loop logs and falls back to echo for that reading (contract says
process_reading never raises — this is belt + braces).
Status at my last test run: pipeline imported fine once and all 14 service
checks passed through it; a later run fell back to echo (A mid-edit) and also
passed — both paths are exercised.

## 4. `StateStore` falls back to tempdir when SQLite can't lock
On this Cowork Linux mount, `sqlite3` raises `OperationalError: disk I/O error`
at the default `settings.db_path` (mounted FS refuses locking). `StateStore`
now catches that on open and falls back to `<tempdir>/praetor_state.db` with a
printed warning. On a normal disk (demo laptop, CI) the configured path is used
as-is. No contract change.

## 5. Cowork mount quirk (extends the known git-in-/tmp rule)
In-place EDITS of existing files on the Windows side can leave the Linux mount
serving a stale, truncated read (state_store.py was pinned at its pre-edit size
→ SyntaxError mid-token; page cache kept old length). New-file creates sync
fine. Fix that worked: write the full correct file through the mount from the
Linux side (`cp` from /tmp), then verify both views. If a mounted .py suddenly
fails with a nonsense SyntaxError/traceback, suspect this before suspecting the
code — and run tests from a /tmp copy.

## 6. Endpoint shapes (for the Next.js teammate)
- `/api/stream` is GET + EventSource. Events: `hello`, `tick`, `advisory`,
  `override`, `boss`. `data:` is always one JSON object (schema = pydantic dumps).
- `/api/chat` is POST returning SSE (`token` → `turn` → `done`); the demo page
  parses it with a fetch reader since EventSource ca