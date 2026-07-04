#!/usr/bin/env python3
"""
Crusoe Managed Inference — latency benchmark.

Goal: pick the right model for each tier of the agent.
  - "brain"  : big model, important decisions (target: Nemotron Ultra)
  - "loop"   : fast model, continuous real-time loop (target: Kimi K2.6)

The script FIRST lists the models actually available on your key, then
benchmarks the ones you select. This way it works regardless of the exact
model-id strings on the Crusoe console.

Usage:
    export CRUSOE_API_KEY=...            # never hardcode the key
    pip install openai
    python scripts/crusoe_latency_test.py            # list models + benchmark defaults
    python scripts/crusoe_latency_test.py --list     # just list available models
    python scripts/crusoe_latency_test.py --models "id1,id2" --trials 5

Metrics reported per model (over N trials):
    TTFT   = time to first token (streaming) -> what the operator "feels"
    total  = full response latency
    tok/s  = output tokens / generation time
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

try:
    from openai import OpenAI
except ImportError:
    sys.exit("Missing dependency. Run: pip install openai")

BASE_URL = os.environ.get("CRUSOE_BASE_URL", "https://api.crusoe.ai/v1")

# Edit these once you see the real ids printed by --list.
# Substrings are matched against the available model list (case-insensitive),
# so "nemotron" will resolve to whatever the full id is.
DEFAULT_TARGETS = {
    "brain (decisions)": "nemotron",
    "loop (real-time)": "kimi",
}

# Representative advisory prompt — what the agent actually does in the demo.
SYSTEM = (
    "You are the operations agent for a factory. You read the physical twin "
    "state and give ONE short, concrete advisory a non-technical operator can "
    "act on. Be terse."
)
USER = (
    "State: Press 3, bearing B2 vibration RMS +38% over baseline, temp 71C "
    "rising 0.4C/min, PINN residual trending up. Estimated time-to-failure ~40h. "
    "Give the advisory."
)


def get_client() -> OpenAI:
    key = os.environ.get("CRUSOE_API_KEY")
    if not key:
        sys.exit("Set CRUSOE_API_KEY in your environment first (never commit it).")
    return OpenAI(base_url=BASE_URL, api_key=key)


def list_models(client: OpenAI) -> list[str]:
    ids = [m.id for m in client.models.list().data]
    return sorted(ids)


def resolve(target: str, available: list[str]) -> str | None:
    """Match a substring (or exact id) against available models."""
    if target in available:
        return target
    hits = [m for m in available if target.lower() in m.lower()]
    return hits[0] if hits else None


def bench_once(client: OpenAI, model: str) -> tuple[float, float, int]:
    """Return (ttft_s, total_s, output_tokens) for one streamed completion."""
    start = time.perf_counter()
    ttft = None
    tokens = 0
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": USER}],
        max_tokens=120,
        temperature=0.3,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            if ttft is None:
                ttft = time.perf_counter() - start
            tokens += 1
    total = time.perf_counter() - start
    return (ttft or total), total, tokens


def bench_model(client: OpenAI, label: str, model: str, trials: int) -> None:
    print(f"\n=== {label}  ->  {model} ===")
    ttfts, totals, rates = [], [], []
    for i in range(trials):
        try:
            ttft, total, toks = bench_once(client, model)
        except Exception as e:  # noqa: BLE001 - report and move on
            print(f"  trial {i+1}: ERROR {type(e).__name__}: {e}")
            continue
        gen = max(total - ttft, 1e-6)
        rate = toks / gen
        ttfts.append(ttft)
        totals.append(total)
        rates.append(rate)
        print(f"  trial {i+1}: TTFT {ttft*1000:6.0f} ms | total {total*1000:6.0f} ms "
              f"| {toks:3d} tok | {rate:5.1f} tok/s")
    if ttfts:
        print(f"  --> median TTFT {statistics.median(ttfts)*1000:.0f} ms | "
              f"median total {statistics.median(totals)*1000:.0f} ms | "
              f"median {statistics.median(rates):.1f} tok/s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="only list available models")
    ap.add_argument("--models", default="", help="comma-separated model ids/substrings")
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    client = get_client()

    print(f"Base URL: {BASE_URL}")
    print("Fetching available models...")
    available = list_models(client)
    print(f"\n{len(available)} models available:")
    for m in available:
        print(f"  - {m}")

    if args.list:
        return

    if args.models:
        targets = {t.strip(): t.strip() for t in args.models.split(",") if t.strip()}
    else:
        targets = DEFAULT_TARGETS

    print("\n" + "=" * 60)
    print("BENCHMARK")
    print("=" * 60)
    for label, target in targets.items():
        model = resolve(target, available)
        if not model:
            print(f"\n=== {label} ===\n  '{target}' not found in available models. "
                  f"Run with --list and pick an exact id.")
            continue
        bench_model(client, label, model, args.trials)


if __name__ == "__main__":
    main()
