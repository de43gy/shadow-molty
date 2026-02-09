from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import httpx
import openai

from src.config import settings

logger = logging.getLogger(__name__)

# Preferred free models on OpenRouter, ordered by quality.
# Used when OPENROUTER_MODEL is not set explicitly.
_PREFERRED_FREE_MODELS = [
    "google/gemini-2.0-flash-exp:free",
    "google/gemini-2.5-flash-preview:free",
    "google/gemini-2.0-flash-thinking-exp:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
]

_FALLBACK_MODEL = _PREFERRED_FREE_MODELS[0]


def _resolve_openrouter_model() -> str:
    """Query OpenRouter API to find the best available free model."""
    try:
        resp = httpx.get(
            "https://openrouter.ai/api/v1/models",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])

        free_ids: set[str] = set()
        for m in data:
            pricing = m.get("pricing", {})
            if pricing.get("prompt") == "0" and pricing.get("completion") == "0":
                free_ids.add(m["id"])

        # Try preferred models first
        for model_id in _PREFERRED_FREE_MODELS:
            if model_id in free_ids:
                logger.info("Auto-selected OpenRouter model: %s", model_id)
                return model_id

        # Fallback: any model with :free suffix
        free_tagged = sorted(m for m in free_ids if ":free" in m)
        if free_tagged:
            logger.info("Auto-selected OpenRouter model (fallback): %s", free_tagged[0])
            return free_tagged[0]

        if free_ids:
            pick = sorted(free_ids)[0]
            logger.info("Auto-selected OpenRouter model (fallback): %s", pick)
            return pick

    except Exception as e:
        logger.warning("Failed to resolve OpenRouter model: %s", e)

    logger.info("Using hardcoded OpenRouter fallback: %s", _FALLBACK_MODEL)
    return _FALLBACK_MODEL


def _make_provider_stats() -> dict:
    return {
        "requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "errors": 0,
        "actions": defaultdict(lambda: {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}),
    }


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


@dataclass
class _Provider:
    name: str
    client: openai.AsyncOpenAI
    model: str


class _Completions:
    """Mimics client.chat.completions.create() with fallback."""

    def __init__(self, providers: list[_Provider], usage: dict, storage=None):
        self._providers = providers
        self._usage = usage
        self._storage = storage

    async def create(self, **kwargs):
        action = kwargs.pop("_action", "unknown")
        last_error = None
        for p in self._providers:
            key = f"{p.name}:{p.model}"
            try:
                kwargs["model"] = p.model
                resp = await p.client.chat.completions.create(**kwargs)
                await self._record(p.name, p.model, action, resp)
                return resp
            except Exception as e:
                logger.warning("LLM provider %s failed: %s", p.name, e)
                self._usage[key]["errors"] += 1
                last_error = e
        raise last_error

    async def _record(self, provider: str, model: str, action: str, response) -> None:
        usage = getattr(response, "usage", None)
        if not usage:
            return
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        # In-memory
        key = f"{provider}:{model}"
        stats = self._usage[key]
        stats["requests"] += 1
        stats["prompt_tokens"] += pt
        stats["completion_tokens"] += ct
        stats["total_tokens"] += pt + ct
        a = stats["actions"][action]
        a["requests"] += 1
        a["prompt_tokens"] += pt
        a["completion_tokens"] += ct
        # SQLite
        if self._storage:
            try:
                await self._storage.save_llm_usage(provider, model, action, pt, ct)
            except Exception:
                logger.debug("Failed to persist LLM usage to DB", exc_info=True)


class _Chat:
    def __init__(self, providers: list[_Provider], usage: dict, storage=None):
        self.completions = _Completions(providers, usage, storage)


class FallbackLLMClient:
    """Drop-in replacement for openai.AsyncOpenAI with multi-provider fallback."""

    def __init__(self, providers: list[_Provider], storage=None):
        self._usage: dict = defaultdict(_make_provider_stats)
        self.chat = _Chat(providers, self._usage, storage)

    def get_usage_report(self) -> str:
        if not self._usage:
            return "No LLM usage recorded yet."
        lines = ["LLM Usage (since restart)\n"]
        for key, stats in self._usage.items():
            provider, model = key.split(":", 1)
            pt = stats["prompt_tokens"]
            ct = stats["completion_tokens"]
            lines.append(
                f"{provider} ({model}):\n"
                f"  {stats['requests']} req, {_fmt_tokens(stats['total_tokens'])} tok "
                f"({_fmt_tokens(pt)} in + {_fmt_tokens(ct)} out)"
            )
            if stats["errors"]:
                lines[-1] += f", {stats['errors']} errors"
            # Sort actions by request count desc
            sorted_actions = sorted(
                stats["actions"].items(), key=lambda x: x[1]["requests"], reverse=True,
            )
            for action, a in sorted_actions:
                total = a["prompt_tokens"] + a["completion_tokens"]
                lines.append(f"  {action}: {a['requests']} req, {_fmt_tokens(total)} tok")
            lines.append("")
        return "\n".join(lines).strip()


def create_llm_client(storage=None) -> FallbackLLMClient:
    """Factory: build client from settings, skip providers without keys."""
    providers: list[_Provider] = []

    if settings.google_api_key:
        providers.append(_Provider(
            name="google",
            client=openai.AsyncOpenAI(
                api_key=settings.google_api_key,
                base_url=settings.google_base_url,
            ),
            model=settings.google_model or "gemini-2.0-flash",
        ))

    if settings.openrouter_api_key:
        model = settings.openrouter_model or _resolve_openrouter_model()
        providers.append(_Provider(
            name="openrouter",
            client=openai.AsyncOpenAI(
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
            ),
            model=model,
        ))

    if settings.llm_api_key:
        providers.append(_Provider(
            name="legacy",
            client=openai.AsyncOpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
            ),
            model=settings.llm_model,
        ))

    if not providers:
        raise RuntimeError(
            "No LLM providers configured — set GOOGLE_API_KEY, OPENROUTER_API_KEY, or LLM_API_KEY"
        )

    logger.info("LLM providers: %s", [(p.name, p.model) for p in providers])
    return FallbackLLMClient(providers, storage=storage)
