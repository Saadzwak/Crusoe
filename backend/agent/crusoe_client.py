"""Crusoe Managed Inference client — one wrapper for every LLM call in PRAETOR.

Design goals (hackathon-grade resilience):
- OpenAI-compatible AsyncOpenAI against Crusoe (docs/CRUSOE.md).
- role → model indirection: "fast" (DeepSeek V4 Flash), "reasoning" (Nemotron
  Ultra 550B), "omni" (Nemotron 3 Nano Omni), or any explicit model id.
- complete_json(): thinking disabled + tolerant JSON extraction + one repair
  retry — reasoning models leak <think> blocks otherwise.
- 412 "no available servers" retried briefly; on final failure we DEGRADE TO
  MOCK instead of crashing the demo loop.
- MockClient: deterministic, schema-generic. Runs the whole pipeline offline
  (MOCK_LLM=1 or no key). Same interface, zero code change to go live.

NEVER print or log the API key.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncIterator, Optional, Type, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel

from .config import settings, thinking_off_extra_body

T = TypeVar("T", bound=BaseModel)

Messages = list[dict[str, Any]]

_ROLE_TO_MODEL = {
    "fast": lambda: settings.model_fast,
    "reasoning": lambda: settings.model_reasoning,
    "omni": lambda: settings.model_omni,
}


def resolve_model(role: str) -> str:
    fn = _ROLE_TO_MODEL.get(role)
    return fn() if fn else role  # explicit model id passes through


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


def extract_json_block(text: str) -> str:
    """Best-effort: first '{' to last '}' after stripping think blocks/fences."""
    t = strip_thinking(text)
    t = re.sub(r"```(?:json)?", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        return t[start : end + 1]
    return t.strip()


# ===================================================================== base
class LLMClient:
    """Interface every module codes against. hint labels the call site so the
    mock can produce a sensible answer ("tier2_classify", "advocate", ...)."""

    is_mock = False

    async def complete(
        self,
        role: str,
        messages: Messages,
        *,
        max_tokens: int = 512,
        temperature: float = 0.3,
        hint: str = "",
        timeout: float = 90.0,
        thinking: bool = False,
    ) -> str:
        raise NotImplementedError

    async def complete_json(
        self,
        role: str,
        messages: Messages,
        schema: Type[T],
        *,
        max_tokens: int = 900,
        temperature: float = 0.2,
        hint: str = "",
        retries: int = 1,
        timeout: float = 90.0,
    ) -> T:
        raise NotImplementedError

    async def stream(
        self,
        role: str,
        messages: Messages,
        *,
        max_tokens: int = 512,
        temperature: float = 0.3,
        hint: str = "",
        timeout: float = 90.0,
    ) -> AsyncIterator[str]:
        raise NotImplementedError
        yield  # pragma: no cover


# ===================================================================== mock
def _mid(lo: Optional[float], hi: Optional[float], default: float) -> float:
    if lo is not None and hi is not None:
        return round(lo + (hi - lo) * 0.82, 2)
    return default


def _mock_value(name: str, annotation: Any, field: Any, hint: str, prompt: str) -> Any:
    """Generic, always-valid value for a pydantic field (mock mode)."""
    origin = get_origin(annotation)
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        return _mock_value(name, args[0], field, hint, prompt) if args else None
    if origin in (list, set, tuple):
        return []
    if origin is dict:
        return {}
    if annotation is bool:
        return True
    if annotation in (int, float):
        lo = hi = None
        for meta in getattr(field, "metadata", []) or []:
            lo = getattr(meta, "ge", lo)
            hi = getattr(meta, "le", hi)
        return _mid(lo, hi, 1.0 if annotation is float else 1)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _mock_model(annotation, hint, prompt).model_dump()
    # Enum-ish (incl. RiskLabel): choose by prompt keywords, else first member
    if isinstance(annotation, type) and hasattr(annotation, "__members__"):
        members = list(annotation.__members__.values())
        pl = prompt.lower()
        for kw in ("critical", "high", "watch", "clear"):
            for m in members:
                if kw in str(m.value).lower() and kw in pl:
                    return m
        return members[-1] if "verdict" in name else members[0]
    # strings
    canned = {
        "title": "Bearing degradation trend on curing press",
        "verdict": "approved",
        "critique": "[mock] Grounded in provided values; action is operator-safe.",
        "reason": "[mock] Signal exceeds baseline envelope.",
    }
    if name in canned:
        return canned[name]
    return f"[mock:{hint or name}] deterministic stand-in response."


def _mock_model(schema: Type[T], hint: str, prompt: str) -> T:
    values: dict[str, Any] = {}
    for name, field in schema.model_fields.items():
        if not field.is_required():
            continue
        values[name] = _mock_value(name, field.annotation, field, hint, prompt)
    return schema.model_validate(values)


class MockClient(LLMClient):
    """Deterministic offline stand-in. Keyword-aware where the demo needs it."""

    is_mock = True

    @staticmethod
    def _last_user(messages: Messages) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                c = m.get("content")
                return c if isinstance(c, str) else json.dumps(c)
        return ""

    async def complete(self, role, messages, *, max_tokens=512, temperature=0.3,
                       hint="", timeout=90.0, thinking=False) -> str:
        await asyncio.sleep(0.05)
        prompt = self._last_user(messages).lower()
        if hint == "tier2_classify":
            if any(k in prompt for k in ("critical", "0.9", "failure", "rul_cycles': 1", "imminent")):
                return "CRITICAL"
            if any(k in prompt for k in ("high", "drift", "rising", "+3", "anomal")):
                return "HIGH"
            return "CLEAR"
        if hint == "boss":
            return ("[mock] Plant status: Curing shows a rising thermal trend on press RC-07 "
                    "(watch); Calendering and Mixing nominal. One open advisory awaiting the "
                    "operator's call. No cross-department cascade detected this window.")
        if hint == "operator":
            return ("[mock] Based on the last readings, vibration RMS on RC-07 is 38% over its "
                    "baseline and the PINN residual is trending up [sensor_history RC-07]. The "
                    "site dossier prices a curing stop at the bottleneck rate [site_dossier p.4]. "
                    "I recommend planning the swap in the next maintenance window — your call.\n"
                    "Gaps: no camera frame or acoustic reading for this press in the store yet.")
        return f"[mock:{hint or role}] {prompt[:80]}"

    async def complete_json(self, role, messages, schema, *, max_tokens=900,
                            temperature=0.2, hint="", retries=1, timeout=90.0):
        await asyncio.sleep(0.05)
        return _mock_model(schema, hint, self._last_user(messages))

    async def stream(self, role, messages, *, max_tokens=512, temperature=0.3,
                     hint="", timeout=90.0) -> AsyncIterator[str]:
        text = await self.complete(role, messages, hint=hint)
        for i in range(0, len(text), 12):
            await asyncio.sleep(0.01)
            yield text[i : i + 12]


# ===================================================================== live
class CrusoeClient(LLMClient):
    def __init__(self) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            base_url=settings.base_url, api_key=settings.api_key, max_retries=0
        )
        self._fallback = MockClient()

    async def _create(self, role: str, messages: Messages, *, max_tokens: int,
                      temperature: float, timeout: float, thinking: bool,
                      stream: bool = False) -> Any:
        model = resolve_model(role)
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            stream=stream,
        )
        if not thinking:
            extra = thinking_off_extra_body(model)
            if extra:
                kwargs["extra_body"] = extra
        last_err: Optional[Exception] = None
        for attempt, pause in enumerate((0.0, 3.0, 8.0)):
            if pause:
                await asyncio.sleep(pause)
            try:
                return await self._client.chat.completions.create(**kwargs)
            except Exception as e:  # noqa: BLE001 — includes 412 capacity errors
                last_err = e
                status = getattr(e, "status_code", None)
                # Retry only transient statuses (412 capacity / 429 / 5xx) and
                # network errors (status None). 400/401/404 fail fast.
                if status is not None and status not in (412, 429, 500, 502, 503):
                    break
        raise last_err  # type: ignore[misc]

    async def complete(self, role, messages, *, max_tokens=512, temperature=0.3,
                       hint="", timeout=90.0, thinking=False) -> str:
        try:
            resp = await self._create(role, messages, max_tokens=max_tokens,
                                      temperature=temperature, timeout=timeout,
                                      thinking=thinking)
            return strip_thinking(resp.choices[0].message.content or "")
        except Exception as e:  # degrade, don't die on stage
            print(f"[crusoe_client] live call failed ({type(e).__name__}); degrading to mock "
                  f"for hint={hint!r}")
            return await self._fallback.complete(role, messages, hint=hint)

    async def complete_json(self, role, messages, schema, *, max_tokens=900,
                            temperature=0.2, hint="", retries=1, timeout=90.0):
        sys_extra = (
            "Respond with ONE valid JSON object only — no prose, no markdown fences. "
            f"It must validate against this JSON schema:\n{json.dumps(schema.model_json_schema())}"
        )
        msgs = [messages[0]] if messages and messages[0].get("role") == "system" else []
        if msgs:
            msgs = [dict(msgs[0], content=str(msgs[0]["content"]) + "\n\n" + sys_extra)]
            msgs += messages[1:]
        else:
            msgs = [{"role": "system", "content": sys_extra}, *messages]

        last_err: Optional[Exception] = None
        raw = ""
        for _ in range(retries + 1):
            try:
                raw = await self.complete(role, msgs, max_tokens=max_tokens,
                                          temperature=temperature, hint=hint,
                                          timeout=timeout)
                return schema.model_validate_json(extract_json_block(raw))
            except Exception as e:  # noqa: BLE001
                last_err = e
                msgs = msgs + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"Invalid JSON ({e}). Return ONLY the corrected JSON object."},
                ]
        print(f"[crusoe_client] JSON parse failed after retries for hint={hint!r}; mock fallback.")
        return await self._fallback.complete_json(role, messages, schema, hint=hint)

    async def stream(self, role, messages, *, max_tokens=512, temperature=0.3,
                     hint="", timeout=90.0) -> AsyncIterator[str]:
        try:
            resp = await self._create(role, messages, max_tokens=max_tokens,
                                      temperature=temperature, timeout=timeout,
                                      thinking=False, stream=True)
            inside_think = False
            async for chunk in resp:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if not delta:
                    continue
                # crude guard: skip leaked think blocks in streams
                if "<think>" in delta:
                    inside_think = True
                    continue
                if "</think>" in delta:
                    inside_think = False
                    continue
                if not inside_think:
                    yield delta
        except Exception as e:  # noqa: BLE001
            print(f"[crusoe_client] stream failed ({type(e).__name__}); mock fallback.")
            async for tok in self._fallback.stream(role, messages, hint=hint):
                yield tok


# ================================================================ singleton
_client: Optional[LLMClient] = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = MockClient() if settings.mock_mode else CrusoeClient()
        mode = "MOCK (offline deterministic)" if _client.is_mock else "LIVE Crusoe"
        print(f"[crusoe_client] mode: {mode}, base_url={settings.base_url}")
    return _client
