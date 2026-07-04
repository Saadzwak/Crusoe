"""Boss-LLM — plant-wide summary read EXCLUSIVELY from the shared store.

Architecture rule (README / contract): line agents never talk to each other or
to the boss; everyone writes to the shared store, the boss reads it. This
module therefore takes (client, store) and touches nothing else.
"""
from __future__ import annotations

from .config import settings
from .crusoe_client import LLMClient
from .prompts import BOSS_SYSTEM, boss_user, format_overrides
from .schemas import AdvisoryStatus, PlantSummary


def _dept_status(snapshot: dict) -> dict[str, str]:
    """Duck-typed per-dept dicts -> short status strings for the summary."""
    out: dict[str, str] = {}
    for dept, info in (snapshot or {}).items():
        if isinstance(info, dict):
            risk = info.get("risk") or info.get("last_risk") or info.get("label") or "?"
            epoch = info.get("epoch") or info.get("last_epoch")
            out[dept] = f"{risk}" + (f" @ epoch {epoch}" if epoch is not None else "")
        else:
            out[dept] = str(info)[:60]
    return out


async def generate_plant_summary(client: LLMClient, store) -> PlantSummary:
    """Compose the store digest, one reasoning call, return PlantSummary."""
    try:
        snapshot = store.department_snapshot() or {}
        advisories = store.get_advisories(limit=50) or []
        overrides = store.get_overrides(limit=20) or []
    except Exception as e:  # noqa: BLE001 — degraded store must not kill the boss
        return PlantSummary(
            text=f"Plant summary unavailable: shared store unreadable ({type(e).__name__}).",
            generated_by="error",
        )

    dept_status = _dept_status(snapshot)
    open_advs = [a for a in advisories if getattr(a, "status", None) == AdvisoryStatus.PENDING]

    snapshot_text = "\n".join(f"- {d}: {s}" for d, s in dept_status.items()) or "- no ticks yet"
    advisories_text = "\n".join(
        f"- [{a.severity.value}] {a.machine_id} ({a.department}) {a.title} "
        f"(status {a.status.value})"
        for a in advisories[:10]
    ) or "- none"

    text = await client.complete(
        role="reasoning", hint="boss", max_tokens=320, temperature=0.3,
        messages=[{"role": "system", "content": BOSS_SYSTEM},
                  {"role": "user", "content": boss_user(
                      snapshot_text, advisories_text, format_overrides(overrides))}],
    )

    # Provenance label only — business logic never branches on mock (contract).
    generated_by = "mock" if getattr(client, "is_mock", False) else settings.model_reasoning
    return PlantSummary(
        text=text.strip(),
        department_status=dept_status,
        open_advisories=len(open_advs),
        generated_by=generated_by,
    )
