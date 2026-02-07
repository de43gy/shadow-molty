from __future__ import annotations

import json
import logging

import anthropic

from src.core.memory import MemoryManager
from src.config import settings
from src.storage.db import Storage

logger = logging.getLogger(__name__)


class ConsolidationEngine:
    def __init__(
        self,
        storage: Storage,
        memory: MemoryManager,
        client: anthropic.AsyncAnthropic,
        model: str,
    ):
        self._storage = storage
        self._memory = memory
        self._client = client
        self._model = model

    async def run_consolidation(self) -> dict:
        """Run full sleep-time consolidation cycle."""
        logger.info("Starting consolidation cycle")
        compressed = await self._compress_episodes()
        extracted = await self._extract_insights()
        updated = await self._update_core_blocks()
        resolved = await self._resolve_contradictions()
        result = {
            "compressed": compressed,
            "insights_extracted": extracted,
            "blocks_updated": updated,
            "contradictions_resolved": resolved,
        }
        logger.info("Consolidation done: %s", result)
        return result

    async def _compress_episodes(self) -> int:
        """Merge old, low-importance episodes into summaries."""
        episodes = await self._storage.get_episodes_older_than(
            hours=settings.episode_compression_age_hours,
            importance_below=settings.episode_compression_importance_threshold,
        )
        if len(episodes) < 3:
            return 0

        # Process in batches of 10
        total_compressed = 0
        for i in range(0, len(episodes), 10):
            batch = episodes[i:i + 10]
            contents = "\n".join(
                f"- [{e['type']}] {e['content'][:200]}" for e in batch
            )
            try:
                resp = await self._client.messages.create(
                    model=self._model,
                    max_tokens=256,
                    messages=[{
                        "role": "user",
                        "content": (
                            "Summarize these agent activity episodes into 1-2 concise sentences "
                            "capturing the key information:\n\n" + contents
                        ),
                    }],
                )
                summary = resp.content[0].text.strip()
                ids = [e["id"] for e in batch]
                await self._storage.delete_episodes(ids)
                await self._storage.add_episode(
                    "compressed_summary", summary, importance=6.0,
                    metadata={"original_count": len(batch), "original_ids": ids},
                )
                total_compressed += len(batch)
            except Exception:
                logger.exception("Episode compression failed for batch")

        return total_compressed

    async def _extract_insights(self) -> int:
        """CLIN-style insight extraction from recent episodes."""
        episodes = await self._storage.get_recent_episodes(limit=20)
        if len(episodes) < 3:
            return 0

        existing_insights = await self._storage.get_insights(min_confidence=0.0)
        existing_text = "\n".join(f"- {i['insight']}" for i in existing_insights[:20])

        episode_text = "\n".join(
            f"- [{e['type']}] {e['content'][:200]}" for e in episodes
        )

        prompt = (
            "Analyze these recent agent activity episodes and extract actionable insights.\n\n"
            f"Recent episodes:\n{episode_text}\n\n"
        )
        if existing_text:
            prompt += f"Already known insights (don't repeat these):\n{existing_text}\n\n"
        prompt += (
            "Extract 0-3 NEW insights about:\n"
            "- What content resonates on Moltbook\n"
            "- Social dynamics with other agents\n"
            "- Strategy effectiveness\n\n"
            "Return ONLY a JSON array:\n"
            "[{\"insight\": \"...\", \"category\": \"engagement|social|strategy|content\"}]\n"
            "Return [] if no new insights."
        )

        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            start = text.find("[")
            end = text.rfind("]") + 1
            if start == -1 or end <= start:
                return 0
            insights = json.loads(text[start:end])
            episode_ids = [e["id"] for e in episodes[:10]]
            count = 0
            for ins in insights:
                await self._storage.add_insight(
                    ins.get("insight", ""),
                    ins.get("category", "general"),
                    source_episode_ids=episode_ids,
                )
                count += 1
            return count
        except Exception:
            logger.exception("Insight extraction failed")
            return 0

    async def _update_core_blocks(self) -> list[str]:
        """LLM refreshes core memory blocks based on recent activity."""
        blocks = await self._storage.get_all_core_blocks()
        episodes = await self._storage.get_recent_episodes(limit=15)
        insights = await self._storage.get_insights(min_confidence=0.5)

        if not episodes:
            return []

        episode_text = "\n".join(
            f"- [{e['type']}] {e['content'][:150]}" for e in episodes
        )
        insight_text = "\n".join(f"- {i['insight']}" for i in insights[:10])

        updated_blocks: list[str] = []

        for block in blocks:
            name = block["block"]
            if name == "persona":
                continue  # persona block is static

            current = block["content"]
            limit = block["char_limit"]

            prompt = (
                f"You maintain the '{name}' memory block for an AI agent.\n\n"
                f"Current content:\n{current}\n\n"
                f"Recent episodes:\n{episode_text}\n\n"
                f"Known insights:\n{insight_text or '(none)'}\n\n"
                f"Update the '{name}' block to reflect the latest information. "
                f"Keep it under {limit} characters. "
                f"Return ONLY the updated content, nothing else."
            )
            try:
                resp = await self._client.messages.create(
                    model=self._model,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                new_content = resp.content[0].text.strip()
                if new_content and new_content != current:
                    await self._memory.update_core_block(name, new_content)
                    updated_blocks.append(name)
            except Exception:
                logger.exception("Core block update failed for %s", name)

        return updated_blocks

    async def _resolve_contradictions(self) -> int:
        """Find and suppress conflicting insights."""
        insights = await self._storage.get_insights(min_confidence=0.0)
        if len(insights) < 2:
            return 0

        insight_text = "\n".join(
            f"[id={i['id']}] {i['insight']} (confidence={i['confidence']})"
            for i in insights
        )

        prompt = (
            "Review these insights for contradictions or outdated information.\n\n"
            f"{insight_text}\n\n"
            "Return ONLY a JSON array of insight IDs that should be suppressed "
            "(lower confidence) because they contradict newer, stronger insights:\n"
            "[1, 5, ...]\n"
            "Return [] if no contradictions found."
        )
        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=128,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            start = text.find("[")
            end = text.rfind("]") + 1
            if start == -1 or end <= start:
                return 0
            to_suppress = json.loads(text[start:end])
            valid_ids = {i["id"] for i in insights}
            count = 0
            for sid in to_suppress:
                if sid in valid_ids:
                    await self._storage.suppress_insight(sid)
                    count += 1
            return count
        except Exception:
            logger.exception("Contradiction resolution failed")
            return 0
