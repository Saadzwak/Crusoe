# Industrial Cortex

**Predictive maintenance + operations optimization for under-instrumented factories, powered by Physics-Informed Neural Networks (PINNs) and an agent that acts.**

Built at RAISE Summit Hackathon 2026, Paris — Crusoe Track (Track 3).

## The problem

Unplanned downtime in manufacturing is brutally expensive (Toyota builds a car every 57 s — 10 minutes of line stoppage ≈ €200k lost). Classic predictive-maintenance solutions require heavy sensor instrumentation and produce alerts only engineers can interpret. Most factories are under-instrumented and won't pay to retrofit sensors everywhere.

## Our approach

A **PINN** embeds the physics (vibration, thermal, fatigue equations) directly into training. It is penalized when it contradicts physical law, so it works reliably with **few sensors and little failure history**, and can infer the state of unmeasured locations (measure 5 points, compute the other 95 through the equations).

One physical model, two uses:

1. **Failure prediction** — when the real machine deviates from the physically expected behavior, that deviation is the early degradation signal → "this bearing fails in ~40h".
2. **OPEX optimization** — the same model finds the operating point that maximizes output while minimizing cost, and acts preventively (cooling before overheating is cheaper than recovering a hot system).

## The agent (Crusoe layer)

An LLM on **Crusoe Managed Inference** continuously reads the physical twin's state (JSON every ~5 s from the local PINN engine) and translates it into concrete advisories a non-technical operator can understand, question, and **override** in the moment ("Press 3, bearing degrading, intervene before 2 pm"). The agent learns from overrides. Advisories are voiced via **Gradium TTS** — factory operators have their hands busy.

Two-tier model architecture: a large model (Nemotron 3 Ultra 550B) for important decisions, a fast one (DeepSeek V4 Flash) for the continuous loop.

**This is not a dashboard.** The 3D view is how the agent perceives; the product is an agent that predicts, advises, triggers, and learns from operator overrides.

## Architecture

```
industrial-cortex/
├── frontend/     Next.js + React Three Fiber — 3D production line, health states, advisory UI + override
├── backend/      FastAPI — agent loop + API
│   ├── agent/    Crusoe calls, advisory logic
│   └── api/      REST/WebSocket endpoints to the frontend
├── pinn/         PINN engine (PyTorch) — physical twin + cascade graph
│   ├── data/     run-to-failure datasets (gitignored if large)
│   └── models/
├── docs/         architecture, decisions, pitch notes
└── scripts/      setup, dataset download
```

PINN + cascade graph run locally; LLM reasoning runs on Crusoe.

## Validation data

Public run-to-failure bearing/machine datasets: XJTU-SY, FEMTO/PRONOSTIA (IEEE PHM 2012), IMS/NASA, KAIST 2024 (CC BY), CWRU, Ferrara. These run until actual failure, so the demo can show a prediction come true.

## Setup

```bash
cp .env.example .env   # fill in your own keys — never commit .env
# frontend
cd frontend && npm install && npm run dev
# backend
cd backend && pip install -r requirements.txt && uvicorn main:app --reload
```

## Team

- **Alberto** — idea lead, physics (nuclear background), pitch
- **PhD teammate** — PINN engine (12 months of PINN research)
- **Tim** — ML/physics
- **Teammate 4** — ML/physics
- **Eric** — full-stack, 3D frontend, demo, devops

## Hackathon compliance

- Public repo, **New Work Only**: all code written during the event (pre-event prep limited to this structure and configs).
- No API keys in the repo — see `.env.example`.
