# Project Architecture — Crusoe Track, RAISE Summit Hackathon 2026

*Summary document for team alignment before task allocation and final merge on GitHub.*

---

## 1. Vision in one sentence

A factory digital twin where each critical machine is monitored by a model that understands its physics, supervised by an AI agent that talks to the operator in plain language, justifies its decisions with real data, and always leaves the human in charge of the final call.

**Real organizational inspiration** (not marketing invention, verified):
- Toyota's **Jidoka/Andon** principle: a problem is detected (automatically or by a human), a signal triggers, someone intervenes — without stopping the whole line.
- Michelin's **"organisation responsabilisante"** (empowering organization, since 2012): the operator keeps real decision-making autonomy, the AI only proposes.
- A real, currently active R&D project (IRT SystemX + Air Liquide + Michelin, 2025-2028, 42 months) is working on something very close to our problem: generative AI + sensor data for industrial maintenance.

---

## 2. The three blocks of the architecture

| Block | Role | Real factory analogy |
|---|---|---|
| **PINN (physics)** | Understands the machine's real state from sensor data | The sensors + the physical intuition of an experienced technician |
| **Line-LLM** | One agent per machine/line, reasons over what the PINN reports, talks to the operator, justifies, answers questions | The line supervisor / team leader (Andon) |
| **Boss-LLM** | Receives structured summaries from the Line-LLMs, plant-wide view, escalates when needed | The plant manager |

**Information flow rule (important, avoids an engineering trap)**: Line-LLMs do **not** talk directly to each other (no point-to-point mesh — too fragile for a live demo). Each one writes its status to a **shared data store**, tagged by its machine/line of origin. Any Line-LLM, or the Boss-LLM, can read that store to know what's happening elsewhere. Same visible behavior for the operator and the judges, much simpler and more robust architecture.

**Management structure inspired by Michelin (matrix organization — a real, documented pattern)**: a department's boss isn't isolated in their own domain — they have liaisons with them who also understand the other departments (e.g. maintenance, production, + a third one to confirm), so that cross-department consequences of a problem get flagged quickly.

---

## 3. The PINN — chosen architecture

- **Multi-Head PINN (MH-PINN)**: a shared body (shared physical representation) + one output head per sensor/physics type (vibration, thermal, etc., depending on what the machine actually has). Pattern published by the founding PINN research team (Karniadakis et al.) — not improvised.
- Combined with a **recurrent structure (PI-LSTM)** to handle temporal continuity — several papers exist with exactly this pattern, including one directly on bearing fatigue.
- **Implementation**: built directly (e.g. in PyTorch), not tied to a specific vendor framework. If useful as a starting reference, the `javierfa98/PINNs-Examples` GitHub repo has working PyTorch + DeepXDE examples for the heat equation with a pretrained model included — optional, not mandatory.
- **Message security**: each PINN signs its messages with an HMAC code (a unique secret key per PINN) so the Line-LLM can verify origin and integrity. Standard, lightweight, a few lines in Python (`hmac` + `hashlib`).

---

## 4. Continuous learning — Champion/Challenger pattern

A real, documented MLOps pattern (natively supported by AWS SageMaker, Seldon Core, DataRobot), not an invention:

- **Champion** = the model currently serving live predictions, never stops.
- **Challenger** = a copy that learns in the background from the operator's corrections (whenever they reject an alert).
- Trigger: correction buffer → threshold reached (or manual trigger during the demo) → a few fine-tuning steps on that buffer + a sample of past data (to avoid "catastrophic forgetting," a real and documented pitfall of retraining).
- If the Challenger performs better on a validation set, it **replaces** the Champion.
- **For the demo**: try this locally/on CPU first — a reasonably sized PINN can likely fine-tune in a few seconds without a dedicated GPU. GPU (Colab or otherwise) is mainly for the initial full training, from which a weights file is exported and loaded into the backend.

---

## 5. The visual layer — IFC

- Scan the hackathon room with the **Make Plan** app (IFC export confirmed and verified) to symbolically represent the factory on Sunday.
- Displayed via `web-ifc-viewer` (ThatOpen/IFC.js) or the `davras5/ifc-viewer` template (already built for clicking/coloring elements).
- Each sensor/machine is mapped to an IFC element GUID → colored red on alert.
- **Important for the dev**: build the IFC loader to accept any file from the start (test with a placeholder IFC now), Sunday's real file will just be a swap, not new development.

---

## 6. What the operator can do (the part that answers the Crusoe brief word for word)

The operator can ask the agent a question ("I don't think this machine will fail tomorrow") and the agent must answer using real data, not vague assertions. Concretely: the Line-LLM has access to tools (`get_sensor_history`, `get_pinn_reconstruction`, `get_camera_frame`, `get_machine_spec`) and must always cite what it actually looked up in its answer.

---

## 7. Verified data sources (for the next step — training the PINNs)

- **CWRU** (Case Western Reserve University) — the world-reference bearing vibration dataset.
- **NASA IMS Bearing Dataset** — real degradation to failure, complements CWRU.
- **NASA Milling Dataset** — cutting tool wear, useful if a machine-tool angle is chosen.
- `github.com/VictorBauler/awesome-bearing-dataset` — to compare other options (Paderborn, FEMTO-ST, MFPT, XJTU-SY).
- **Next step, validated by Saad**: once this architecture is confirmed by the team, targeted search for Michelin-specific data (or generic if nothing specific exists), for 2-3 PINNs depending on time available.

---

## 8. Still to decide

- The third department in the matrix structure (the name given via voice dictation wasn't clear — to confirm).
- Exact number of MH-PINN heads (depends on which sensors are actually available in the data found).
- Number of PINNs to actually cover for the demo (1 solid one beats several half-built ones).

---

## 9. Task split (to formalize in the team tracking file)

- **Piece 1 — PINN**: Saad
- **Piece 2 — Line-LLM**: Saad's colleague
- **Contract to lock before coding separately**: exact JSON format sent by the PINN (data + HMAC code), so the final GitHub merge stays clean.
