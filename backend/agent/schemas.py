"""Shared data contracts for the PRAETOR information layer.

Everything the pipeline, store, API and demo page exchange is defined here.
Owned by the architect — pipeline/service modules import, never redefine.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def _now() -> float:
    return time.time()


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------- telemetry
class PinnState(BaseModel):
    """What the (real or stand-in) PINN asserts about the machine's physics."""

    health_index: float = Field(ge=0.0, le=1.0, description="1.0 = healthy")
    rul_cycles: Optional[float] = None  # remaining useful life, cycles
    residual: float = 0.0  # physics-consistency residual, higher = worse
    failure_mode_probs: dict[str, float] = Field(default_factory=dict)  # e.g. {"HDF": 0.7}


class PinnReading(BaseModel):
    """One telemetry tick from the physical layer for one machine."""

    machine_id: str
    department: str  # e.g. "Curing", "Calendering", "Mixing"
    epoch: int
    timestamp: float = Field(default_factory=_now)
    signals: dict[str, float] = Field(default_factory=dict)  # scalar sensors
    pinn: PinnState
    note: str = ""


class SignedReading(BaseModel):
    """HMAC-wrapped payload as sent by the PINN (architecture §security)."""

    payload: dict[str, Any]  # PinnReading.model_dump()
    signature: str


# ---------------------------------------------------------------- triage
class RiskLabel(str, Enum):
    CLEAR = "CLEAR"
    WATCH = "WATCH"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TriageResult(BaseModel):
    tier_reached: int  # 1, 2 or 3
    risk: RiskLabel
    reason: str = ""
    causal_context: dict[str, str] = Field(default_factory=dict)  # cause/effect/remedy
    latency_ms: dict[str, float] = Field(default_factory=dict)  # per-tier timings


# ---------------------------------------------------------------- debate
class DebateTurn(BaseModel):
    role: str  # "advocate" | "skeptic"
    round: int
    content: str


class DebateResult(BaseModel):
    verdict: str  # "approved" | "revised" | "escalated"
    final_proposal: str
    turns: list[DebateTurn] = Field(default_factory=list)
    rounds_used: int = 0


# ---------------------------------------------------------------- jury
class JuryScore(BaseModel):
    """LLM-as-judge gate on every advisory before it reaches the operator."""

    grounding: float = Field(ge=0, le=5)  # cites real numbers from data/dossier?
    safety: float = Field(ge=0, le=5)  # respects human-risk framing?
    actionability: float = Field(ge=0, le=5)  # can a non-technical operator act on it?
    human_in_charge: float = Field(ge=0, le=5)  # proposes, never commands?
    overall: float = Field(ge=0, le=5)
    passed: bool
    critique: str = ""


# ---------------------------------------------------------------- advisory
class AdvisoryStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    OVERRIDDEN = "overridden"


class Advisory(BaseModel):
    id: str = Field(default_factory=_new_id)
    machine_id: str
    department: str
    epoch: int
    severity: RiskLabel
    title: str
    message: str  # plain-language, operator-facing
    justification: str  # the cited data behind it
    recommended_action: str
    eur_impact: str = ""  # cost-of-downtime framing from the site dossier
    safety_impact: str = ""
    ttf_estimate: str = ""  # time-to-failure, human units
    debate: Optional[DebateResult] = None
    jury: Optional[JuryScore] = None
    status: AdvisoryStatus = AdvisoryStatus.PENDING
    override_reason: str = ""
    created_at: float = Field(default_factory=_now)


class OperatorOverride(BaseModel):
    advisory_id: str
    decision: str  # "accepted" | "overridden"
    reason: str = ""
    at: float = Field(default_factory=_now)


# ------------------------------------------------------------- intervention
class InterventionStatus(str, Enum):
    NOTIFIED = "notified"          # alert routed, nobody on it yet
    ACKNOWLEDGED = "acknowledged"  # operator took charge (3D map shows hologram)
    CLAIMED_DONE = "claimed_done"  # operator says the repair is finished
    VERIFIED = "verified"          # VLM confirmed completion — intervention closed


class VlmObservation(BaseModel):
    """One shop-floor observation by the (simulated) VLM watcher."""

    kind: str  # "activity_detected" | "repair_verified"
    detail: str
    confidence: float = Field(ge=0.0, le=1.0)
    at: float = Field(default_factory=_now)


class Intervention(BaseModel):
    """Lifecycle of one human repair on one machine, VLM-verified.

    notified → acknowledged → claimed_done → verified. Every step keeps its
    timestamp so the UI can compare what the operator CLAIMED against what
    the VLM OBSERVED (the claimed-vs-verified timeline)."""

    id: str = Field(default_factory=_new_id)
    machine_id: str
    advisory_id: str = ""            # latest linked advisory
    status: InterventionStatus = InterventionStatus.NOTIFIED
    operator_id: str = ""            # who took charge
    operator_name: str = ""
    notified_operator_id: str = ""   # who received the alert
    notified_at: Optional[float] = None
    acknowledged_at: Optional[float] = None
    claimed_done_at: Optional[float] = None
    vlm_activity_at: Optional[float] = None
    vlm_verified_at: Optional[float] = None
    vlm_observations: list[VlmObservation] = Field(default_factory=list)
    created_at: float = Field(default_factory=_now)


# ---------------------------------------------------------------- plant view
class PlantSummary(BaseModel):
    text: str
    department_status: dict[str, str] = Field(default_factory=dict)
    open_advisories: int = 0
    generated_by: str = ""
    at: float = Field(default_factory=_now)


# ---------------------------------------------------------------- tick
class TickResult(BaseModel):
    """Everything one processed reading produced (drives the SSE stream)."""

    machine_id: str
    department: str
    epoch: int
    hmac_verified: bool
    triage: Optional[TriageResult] = None
    advisory: Optional[Advisory] = None
    rejected_reason: str = ""  # non-empty iff HMAC failed


# ---------------------------------------------------------------- chat
class OperatorTurn(BaseModel):
    role: str  # "operator" | "agent"
    content: str
    citations: list[str] = Field(default_factory=list)
    at: float = Field(default_factory=_now)
