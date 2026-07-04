# Generalized Industrial PINN-MAS

A factory digital twin split into two layers: a **physical layer** that understands what a machine is actually doing (a PINN, physics-informed neural network — not built in this repo), and an **information layer** that takes whatever the physical layer reports, compresses it, reasons about it, debates the right response, and — critically — can prove after the fact exactly what happened, why, and whether anything is quietly drifting off course.

This repo is the information layer. It was built against the architecture described in the team's hackathon planning doc (RAISE Summit 2026, Crusoe track); this README maps the two together honestly, including what isn't built yet.

---

## 1. The two layers

| Layer | Role | Status here |
|---|---|---|
| **Physical layer (PINN)** | Understands the machine's real physical state from sensor data — the sensors plus the physical intuition of an experienced technician | **Not built.** Represented by `main.py`'s synthetic telemetry generator standing in for whatever a real PINN would emit |
| **Information layer** | Takes that physics-informed reading, compresses it, reasons over it, debates the right response with the operator kept in charge, and can prove its own integrity after the fact | **Built — this repo** |

The hackathon doc's vision is "a model that understands the physics, supervised by an AI agent that talks to the operator in plain language, justifies its decisions, and always leaves the human in charge." The physics half is out of scope here. Everything from the moment physics-derived telemetry arrives onward — compression, reasoning, debate, audit — is what's implemented.

---

## 2. System architecture

```
                                PHYSICAL LAYER (external, not built here)
                            Upstream PINN — produces telemetry, signs each
                                   payload with HMAC before sending it
                                                  │
                    ┌─────────────────────────────┼─────────────────────────────┐
                    ▼                              ▼                              ▼
         ┌────────────────────┐        ┌────────────────────┐        ┌────────────────────┐
         │ FeedstockCompound-  │        │ ContinuousExtrusion │        │ ThermalVesselNode   │
         │ ingNode             │        │ Node                │        │                     │
         │ (EdgeMachineAgent)  │        │                     │        │                     │
         │                     │        │                     │        │                     │
         │ 1. verify HMAC      │        │ 1. verify HMAC      │        │ 1. verify HMAC      │
         │ 2. CCCL 3-pass prune│        │ 2. CCCL 3-pass prune│        │ 2. CCCL 3-pass prune│
         │ 3. seal crit. values│        │ 3. seal crit. values│        │ 3. seal crit. values│
         │ 4. archive raw data │        │ 4. archive raw data │        │ 4. archive raw data │
         └──────────┬──────────┘        └──────────┬──────────┘        └──────────┬──────────┘
                    │ claim                        │ claim                        │ claim
                    └──────────────────────────────┼──────────────────────────────┘
                                                    ▼
                              ┌───────────────────────────────────────────┐
                              │     ADVISORY SWARM  (one shared instance    │
                              │     for the whole plant)                    │
                              │                                              │
                              │   TriageAgent  → CausalMatrix lookup          │
                              │        ▼                                    │
                              │   DebateRoom   → Advocate proposes            │
                              │                  Skeptic attacks              │
                              │                  ≤2 rounds, else escalate      │
                              └──────────────────────┬──────────────────────┘
                                                      │ verdict
                    ┌─────────────────────────────────┼─────────────────────────────────┐
                    ▼                                 ▼                                 ▼
         ┌─────────────────────┐          ┌─────────────────────┐          ┌─────────────────────┐
         │  SM2Scheduler        │          │  ExplorationLedger    │          │  (operator — not      │
         │  per-department      │          │  every proposed fix   │          │   built here) would    │
         │  polling cadence     │          │  + its outcome, ever   │          │   read answers via     │
         └─────────────────────┘          └─────────────────────┘          │   the tools below      │
                                                                             └─────────────────────┘

         ── every node, every epoch, publishes into the shared observability layer below ──
         ── (this is the "no point-to-point mesh, write to a shared store" pattern) ──

                    ▼                                 ▼                                 ▼
         ┌─────────────────────┐          ┌─────────────────────┐          ┌─────────────────────┐
         │  DiagnosticsStore    │          │  InformationStateBus  │          │  CustodyChainLedger   │
         │  (SQLite, on disk)   │          │  (async pub/sub,      │          │  (in-memory)          │
         │                      │          │   in-memory)           │          │                      │
         │  WHERE/WHAT/WHY/HOW  │          │  flight-data-recorder  │          │  independently re-    │
         │  trouble codes        │          │  snapshots; subscriber │          │  derives HMAC +        │
         │                      │          │  queues fan out live   │          │  commitment + archive  │
         └──────────┬──────────┘          └──────────┬──────────┘          │  links per node/epoch  │
                    │                                 │                     └──────────┬──────────┘
                    ▼                                 ▼                                ▼
          DiagnosticPort.scan()              DriftAnalyzer.analyze()          CustodyChainLedger.verify()
          plug in anytime, from any          baseline vs. recent window       proves the trail wasn't
          process, same file on disk         per node — is behavior           broken or substituted,
          → "what failed, why, how"          quietly changing?                names the specific link
```

```
generalized-industrial-pinn-mas/
├── src/
│   ├── main.py                    bootstrap — wires everything, runs the 3-epoch demo
│   ├── cccl_reduction/             compression layer
│   │   ├── scrapers.py             TelemetryScraper — 3-pass surgical prune
│   │   └── persistent_archive.py   SQLite-backed raw-array archive
│   ├── agents/                     edge nodes + reasoning/decision layer
│   │   ├── base_agent.py           hash-commitment sealing (per-value)
│   │   ├── hmac_auth.py            origin authentication (per-payload)
│   │   ├── machine_agents.py       EdgeMachineAgent + the 3 department nodes
│   │   ├── triage.py               "LLM" reasoning stand-in
│   │   ├── debate_room.py          Advocate vs. Skeptic adversarial debate
│   │   ├── exploration_ledger.py   append-only fix/outcome memory
│   │   └── advisory_swarm.py       orchestrates triage + debate + diagnostics
│   └── intelligence_core/          Central Intelligence + observability
│       ├── causal_matrix.py        anomaly tag → cause / effect / remedy
│       ├── sm2_scheduler.py        polling cadence
│       ├── diagnostics.py          DiagnosticsStore / DiagnosticPort (DTCs)
│       ├── state_bus.py            InformationStateBus (flight recorder)
│       ├── custody_chain.py        CustodyChainLedger (integrity re-verification)
│       └── drift_analyzer.py       DriftAnalyzer (behavioral drift detection)
└── tests/                          38 tests, 8 files, stdlib only
```

---

## 3. Architecture mapping — hackathon doc ↔ this repo

| Hackathon doc concept | This repo | Status |
|---|---|---|
| PINN (physics) | Upstream, external | Not built — synthetic stand-in only |
| PINN signs messages with HMAC so the receiver can verify origin/integrity | `agents/hmac_auth.py` — `sign_payload`/`verify_payload`, checked in `EdgeMachineAgent.ingest()` before anything else runs | **Built, matches spec directly** |
| Line-LLM (one agent per machine, reasons over PINN output, talks to the operator, justifies with data) | `agents/triage.py` (attaches causal context) + `agents/debate_room.py` (Advocate/Skeptic reasoning) | **Partially built** — the reasoning/justification structure exists; it's rule-based logic standing in for a real LLM call, and there's no operator-facing chat or tool-use yet (`get_sensor_history`, `get_pinn_reconstruction`, etc.) |
| Boss-LLM (plant-wide view, escalates when needed) | No single "boss" agent exists yet; its role is currently split across `intelligence_core/diagnostics.py` (aggregate trouble codes across all nodes) and `intelligence_core/drift_analyzer.py` (cross-epoch behavioral view per node) | **Partially built** — the plant-wide *view* exists, the plant-wide *agent* doesn't |
| **"Line-LLMs don't talk to each other — each writes to a shared data store, tagged by origin; any Line-LLM or the Boss-LLM can read it"** | This is exactly `intelligence_core/state_bus.py` (`InformationStateBus`) and `intelligence_core/diagnostics.py` (`DiagnosticsStore`): every node publishes its outcome; nothing subscribes point-to-point | **Built, matches spec directly** — this was the one explicit engineering trap the doc called out, and the architecture here avoids it the same way |
| Champion/Challenger continuous learning | N/A — there's no ML model in this repo to have a champion or challenger | **Not built** (depends on the PINN existing first) |
| IFC visual layer (factory scan → colored alerts) | N/A | **Not built** |
| Operator asks a question, agent answers with cited real data | `DiagnosticPort.run_diagnostic()` and `TelemetryScraper.get_archived_payload()` are structurally the retrieval tools an operator-facing agent would call, but there's no conversational interface | **Partially built** (the tools exist, the conversation doesn't) |
| Matrix/liaison structure (cross-department consequences flagged quickly) | Not a literal org-chart implementation, but `CustodyChainLedger` and `DriftAnalyzer` both operate *across* all nodes rather than within one, in the same spirit | **Loosely analogous** |

---

## 4. What's actually built, layer by layer

```
Upstream PINN (external)
   │  signs payload with HMAC
   ▼
EdgeMachineAgent.ingest()                    agents/machine_agents.py
   1. verify HMAC                              agents/hmac_auth.py       — reject tampered/unsigned data outright
   2. CCCL surgical_prune()                    cccl_reduction/scrapers.py — 3-pass compression, "find the issues"
   3. seal each critical value                 agents/base_agent.py       — hash commitment + blinding factor
   4. archive raw arrays to disk                cccl_reduction/persistent_archive.py — SQLite, survives the process
   ▼
claim {department, sealed_critical_values, compression stats, ...}
   ▼
AdvisorySwarm.adjudicate()                    agents/advisory_swarm.py
   5. SM2Scheduler cadence update               intelligence_core/sm2_scheduler.py
   6. TriageAgent ("LLM" step)                  agents/triage.py          — attach CausalMatrix context
   7. DebateRoom.deliberate()                   agents/debate_room.py     — Advocate proposes, Skeptic attacks,
                                                                             ≤2 rounds before escalating to fallback
   8. record trouble code (WHERE/WHAT/WHY/HOW)  intelligence_core/diagnostics.py
   9. record custody artifacts                  intelligence_core/custody_chain.py
   10. publish to the state bus                 intelligence_core/state_bus.py
   ▼
verdict {polling_interval, decisions[]}
```

### Compression — `cccl_reduction/`
Three-pass pruning: critical-value extraction (never touched by later passes), matrix/array statistical summarization (raw data archived, not discarded), noise culling. Reversibility and the actual information cost of the inline summary (vs. the archive) are both tested explicitly — see `tests/test_pinn_dataset_fidelity.py`.

### Central Intelligence — `intelligence_core/`
- `causal_matrix.py` — anomaly tag → {probable cause, downstream effect, recommended remedy}.
- `sm2_scheduler.py` — spaced-repetition-style polling cadence; anomalies collapse it to the densest step.
- `diagnostics.py` — the OBD-II layer: every verdict (resolved or escalated) and every HMAC rejection becomes a structured trouble code, persisted to disk, scannable by a completely separate process later.
- `state_bus.py` — the flight-data-recorder layer: async, subscriber-based, decoupled from the reasoning pipeline; nothing blocks on it.

### Reasoning / decision — `agents/`
- `triage.py` — the "Line-LLM" reasoning stand-in: reads compressed findings, attaches causal context.
- `debate_room.py` — Advocate proposes a fix, Skeptic attacks it (matches a past failure, or exceeds a magnitude safety ceiling), room escalates to a safe fallback rather than looping.
- `exploration_ledger.py` — append-only memory of every proposed fix and its outcome.
- `hmac_auth.py` / `base_agent.py` — the two cryptographic protections: origin authentication of the whole payload, and per-value sealing after extraction.

---

## 5. "Which system failed, why, and where" — the observability/integrity layer

This is the part built specifically to answer: *when something fails and the factory immediately redirects to a fallback, how do we know afterward what actually happened?* Three complementary pieces, each answering a different question:

| Question | Module | How |
|---|---|---|
| **What happened, and why?** | `diagnostics.py` — `DiagnosticsStore` / `DiagnosticPort` | Every verdict and rejection becomes a trouble code: WHERE (department), WHAT (tag/metric/value), WHY (causal explanation + the actual debate transcript), HOW (recommended remedy + action taken). Persisted to disk; scannable by a brand-new process pointed at the same file. |
| **Can I prove the data trail wasn't broken or substituted?** | `custody_chain.py` — `CustodyChainLedger` | Independently *re-derives* three cryptographic links per node/epoch — HMAC signature, hash-commitment openings, archived-array resolvability — rather than trusting a log that merely asserts things were fine. A broken link is reported by name and specific reason (`ingestion_hmac` vs `sealed_value:X` vs `archived:<hash>`), because "the signature doesn't match" and "an archived array went missing" call for different responses. |
| **Is behavior quietly changing even though no single event looks alarming?** | `drift_analyzer.py` — `DriftAnalyzer` | Compares a recent window of a node's history against its own earlier baseline (compression ratio, escalation rate, polling cadence). Deliberately does **not** flag a node that has always been bad — that's a chronic failure, not drift — only a genuine *change* from baseline. |

`main.py` demonstrates all three together: a live run produces trouble codes and state-bus entries across 3 epochs; the final epoch is deliberately rigged with a much denser anomaly rate so `DriftAnalyzer` has something real to catch; and two custody records are deliberately corrupted after the fact (a tampered payload, a lost archive entry) to prove `CustodyChainLedger.verify()` actually catches breaks instead of always saying "fine."

---

## 6. Security

- **Origin authentication** (`hmac_auth.py`): HMAC-SHA256 over the whole payload, verified before any processing. Secret comes from `PINN_HMAC_SECRET`; falls back to a hardcoded demo key if unset — **not safe for any real deployment**, provision a real secret.
- **Per-value sealing** (`base_agent.py`): each extracted critical value gets a hash commitment (`SHA256(value || blinding_factor)`) with a fresh blinding factor, so a claim can't be silently altered between extraction and arbitration. Explicitly documented as a hash commitment, *not* a cryptographic Pedersen commitment — different security guarantees, and mislabeling it would be misleading.

---

## 7. Running it

```bash
cd generalized-industrial-pinn-mas/src
python3 main.py
```

Runs 3 synthetic epochs across 3 nodes (Feedstock Compounding, Continuous Extrusion, Thermal Vessel), prints the full debate transcript per anomaly, the Information State Bus's audit table per epoch, drift analysis, chain-of-custody verification (including two deliberate tamper demonstrations), and a final diagnostic scan from a freshly-instantiated tool pointed at the same on-disk store.

```bash
cd generalized-industrial-pinn-mas
for f in tests/test_*.py; do python3 "$f"; done   # plain scripts, most test files
python3 -m unittest tests.test_information_bus     # unittest-based
```

38 tests total across 8 test files, all passing, no third-party dependencies — stdlib only (`re`, `hashlib`, `hmac`, `sqlite3`, `asyncio`, `statistics`, `dataclasses`).

---

## 8. Known gaps

- **No real PINN.** `main.py`'s synthetic generator stands in for it. Nothing here trains or runs a physics model.
- **No real LLM calls.** Triage and debate are deterministic rule-based logic, matching the *position* a Line-LLM would occupy, not an actual model call.
- **No Boss-LLM agent**, no operator chat interface, no IFC visual layer, no Champion/Challenger retraining loop — all explicitly out of scope for this layer.
- **`PINN_HMAC_SECRET` has an insecure hardcoded fallback** — fine for this demo, not for anything real.
- Drift thresholds (`compression_drop_threshold_pct`, `escalation_rate_increase_threshold`) are reasonable defaults, not tuned against real operating data — there isn't any yet.
