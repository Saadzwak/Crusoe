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
                    └──────�