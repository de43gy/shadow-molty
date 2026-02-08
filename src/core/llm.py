from __future__ import annotations

import logging
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


@dataclass
class _Provider:
    name: str
    client: openai.AsyncOpenAI
    model: str


class _Completions:
    """Mimics client.chat.completions.create() with fallback."""

    def __init__(self, providers: list[_Provider]):
        self._providers = providers

    async def create(self, **kwargs):
        last_error = None
        for p in self._providers:
            try:
                kwargs["model"] = p.model
                return await p.client.chat.completions.create(**kwargs)
            except Exception as e:
                logger.warning("LLM provider %s failed: %s", p.name, e)
                last_error = e
        raise last_error


class _Chat:
    def __init__(self, providers: list[_Provider]):
        self.completions = _Completions(providers)


class FallbackLLMClient:
    """Drop-in replacement for openai.AsyncOpenAI with multi-provider fallback."""

    def __init__(self, providers: list[_Provider]):
        self.chat = _Chat(providers)


def create_llm_client() -> FallbackLLMClient:
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
    return FallbackLLMClient(providers)
