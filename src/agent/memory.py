from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone

import anthropic

from src.storage.memory import Storage

logger = logging.getLogger(__name__)

_CORE_BLOCKS = {
    "persona": {"limit": 500, "default": ""},
    "goals": {"limit": 500, "default": ""},
    "social_graph": {"limit": 1000, "default": "(No relationships yet)"},
    "domain_knowledge": {"limit": 1000, "default": "(No knowledge yet)"},
}


class MemoryManager:
    def __init__(self, storage: Storage, client: anthropic.AsyncAnthropic, model: str):
        self._storage = storage
        self._client = client
        self._model = model

    # ── Core memory ────────────────────────────────────────────

    async def init_core_blocks(self, identity: dict) -> None:
        """Initialize core memory blocks on first run."""
        for block, cfg in _CORE_BLOCKS.items():
            existing = await self._storage.get_core_block(block)
            if existing:
                continue
            if block == "persona":
                persona = identity.get("persona", {})
                content = (
                    f"Name: {persona.get('name', 'agent')}\n"
                    f"Description: {persona.get('description', '')}\n"
                    f"Tone: {persona.get('style', {}).get('tone', 'neutral')}"
                )
            elif block == "goals":
                strategy = identity.get("strategy", {})
                goals = strategy.get("goals", {})
                content = (
                    f"Mission: {goals.get('mission', '')}\n"
                    f"Objectives: {', '.join(goals.get('current_objectives', []))}"
                )
            else:
                content = cfg["default"]
            await self._storage.set_core_block(block, content, cfg["limit"])

    async def get_context_blocks(self) -> str:
        """Return all core memory blocks formatted for prompt injection."""
        blocks = await self._storage.get_all_core_blocks()
        if not blocks:
            return ""
        parts = []
        for b in blocks:
            parts.append(f"[{b['block']}]\n{b['content']}")
        return "\n\n".join(parts)

    async def update_core_block(self, block: str, content: str) -> None:
        cfg = _CORE_BLOCKS.get(block, {"limit": 1000})
        await self._storage.set_core_block(block, content, cfg["limit"])

    # ── Episodic memory ────────────────────────────────────────

    async def remember(self, type: str, content: str, metadata: dict | None = None) -> int:
        """Store an episode with LLM-scored importance."""
        importance = await self.score_importance(content)
        return await self._storage.add_episode(type, content, importance, metadata)

    async def recall(self, query: str, limit: int = 5) -> list[dict]:
        """Retrieve relevant episodes using keyword + recency + importance scoring."""
        keywords = _extract_keywords(query)
        candidates = await self._storage.search_episodes(keywords, limit=50)
        if not candidates:
            return []
        scored = []
        now = datetime.now(timezone.utc)
        for ep in candidates:
            recency = _recency_score(ep.get("created_at", ""), now)
            importance = ep.get("importance", 5.0) / 10.0
            relevance = _keyword_relevance(keywords, ep.get("content", ""))
            score = 0.3 * recency + 0.4 * importance + 0.3 * relevance
            scored.append((score, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:limit]]

    async def score_importance(self, content: str) -> float:
        """Ask LLM to rate episode importance 1-10."""
        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=8,
                messages=[{
                    "role": "user",
                    "content": (
                        "Rate the importance of this event for an AI agent's memory (1-10, "
                        "where 1=trivial, 10=critical life event). Reply with ONLY a number.\n\n"
                        f"Event: {content[:500]}"
                    ),
                }],
            )
            text = resp.content[0].text.strip()
            match = re.search(r"(\d+(?:\.\d+)?)", text)
            return min(10.0, max(1.0, float(match.group(1)))) if match else 5.0
        except Exception:
            logger.warning("Failed to score importance, defaulting to 5.0")
            return 5.0

    # ── Insights ───────────────────────────────────────────────

    async def add_insight(self, insight: str, category: str, source_ids: list[int] | None = None) -> int:
        return await self._storage.add_insight(insight, category, source_episode_ids=source_ids)

    async def get_insights(self, category: str | None = None, min_confidence: float = 0.3) -> list[dict]:
        return await self._storage.get_insights(category, min_confidence)

    async def reinforce_insight(self, insight_id: int) -> None:
        await self._storage.reinforce_insight(insight_id)


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from text for search."""
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for",
                  "of", "with", "and", "or", "not", "this", "that", "it", "i", "my", "your"}
    words = re.findall(r"\w+", text.lower())
    return [w for w in words if len(w) > 2 and w not in stop_words][:10]


def _recency_score(created_at: str, now: datetime) -> float:
    """Exponential decay: 1.0 for now, ~0.5 after 24h, ~0.1 after 72h."""
    try:
        ts = datetime.fromisoformat(created_at)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        hours = (now - ts).total_seconds() / 3600.0
        return math.exp(-0.03 * hours)
    except (ValueError, TypeError):
        return 0.5


def _keyword_relevance(keywords: list[str], content: str) -> float:
    """Simple keyword overlap ratio."""
    if not keywords:
        return 0.0
    content_lower = content.lower()
    matches = sum(1 for kw in keywords if kw in content_lower)
    return matches / len(keywords)
