#!/usr/bin/env python3
"""Crusoe live sanity test — run from repo root ON YOUR MACHINE (not the sandbox):

    python scripts/crusoe_sanity.py

Loads .env via backend.agent.config, then:
  1. lists models available on the key and checks our three tier models,
  2. one short completion on the fast tier (DeepSeek V4 Flash),
  3. one on the reasoning tier (Nemotron Ultra 550B),
  4. one native tool-call round-trip on the reasoning tier (operator agent path).

Never prints the key.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.agent.config import settings  # noqa: E402  (loads .env)
from backend.agent.crusoe_client import get_client  # noqa: E402

TOOL_SPEC = [{
    "type": "function",
    "function": {
        "name": "get_sensor_history",
        "description": "Pull recent telemetry for one machine.",
        "parameters": {
            "type": "object",
            "properties": {"machine_id": {"type": "string"}},
            "required": ["machine_id"],
        },
    },
}]


async def main() -> int:
    print(f"base_url: {settings.base_url}")
    if settings.mock_mode:
        print("!! mock mode — no CRUSOE_API_KEY found in .env")
        return 1

    from openai import AsyncOpenAI

    raw = AsyncOpenAI(base_url=settings.base_url, api_key=settings.api_key)
    ok = True

    try:
        models = [m.id for m in (await raw.models.list()).data]
        print(f"\n{len(models)} models on this key")
        for target in (settings.model_fast, settings.model_reasoning, settings.model_omni):
            hit = target in models
            ok &= hit or target == settings.model_omni  # omni optional
            print(f"  {'OK ' if hit else '!! '}{target}")
    except Exception as e:  # noqa: BLE001
        print(f"models.list failed: {type(e).__name__}: {str(e)[:200]}")
        ok = False

    client = get_client()
    for role, label in (("fast", "fast / DeepSeek V4 Flash"),
                        ("reasoning", "reasoning / Nemotron Ultra 550B")):
        t0 = time.perf_counter()
        out = await client.complete(role, [
            {"role": "system", "content": "Factory ops agent. One short sentence."},
            {"role": "user", "content": "Press RC-07 vibration +38% over baseline, "
                                        "RUL ~40h. Advisory?"},
        ], max_tokens=60, hint="sanity")
        live = not out.startswith("[mock")
        ok &= live
        print(f"\n{label}: {time.perf_counter()-t0:.1f}s "
              f"[{'LIVE' if live else 'DEGRADED->MOCK'}]\n  {out[:160]}")

    # Native tool-calling round trip (what the operator agent uses).
    try:
        t0 = time.perf_counter()
        resp = await raw.chat.completions.create(
            model=settings.model_reasoning,
            messages=[{"role": "user",
                       "content": "Check the recent sensor history of machine RC-07."}],
            tools=TOOL_SPEC,
            max_tokens=200,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        calls = resp.choices[0].message.tool_calls or []
        print(f"\ntool-calling: {time.perf_counter()-t0:.1f}s — "
              f"{len(calls)} tool call(s)")
        for c in calls:
            print(f"  -> {c.function.name}({c.function.arguments})")
        if not calls:
            print("  !! model answered without calling the tool — operator agent "
                  "will still work (it re-prompts), but check CRUSOE.md tool support")
    except Exception as e:  # noqa: BLE001
        print(f"tool-calling probe failed: {type(e).__name__}: {str(e)[:200]}")
        ok = False

    print(f"\nRESULT: {'ALL LIVE CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
