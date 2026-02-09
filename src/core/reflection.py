from __future__ import annotations

import json
import logging

import openai
import yaml

from src.core.memory import MemoryManager
from src.core.persona import DEFAULT_STRATEGY
from src.config import settings
from src.storage.db import Storage

logger = logging.getLogger(__name__)


class ReflectionEngine:
    def __init__(
        self,
        storage: Storage,
        memory: MemoryManager,
        client: openai.AsyncOpenAI,
        model: str,
        constitution: dict,
    ):
        self._storage = storage
        self._memory = memory
        self._client = client
        self._model = model
        self._constitution = constitution

    async def _get_strategy(self) -> dict:
        """Load current strategy from DB, falling back to DEFAULT_STRATEGY."""
        row = await self._storage.get_latest_strategy_version()
        if row and row.get("strategy_yaml"):
            return yaml.safe_load(row["strategy_yaml"])
        return DEFAULT_STRATEGY.copy()

    async def should_trigger(self) -> tuple[bool, str]:
        """Check if reflection should run."""
        hb_count = await self._storage.get_state("heartbeat_count")
        count = int(hb_count) if hb_count else 0

        if count > 0 and count % settings.reflection_every_n_heartbeats == 0:
            return True, f"Every {settings.reflection_every_n_heartbeats} heartbeats (count={count})"

        # Check for zero-engagement post
        recent_posts = await self._storage.get_own_posts(limit=1)
        if recent_posts:
            last = recent_posts[0]
            episodes = await self._storage.get_recent_episodes(limit=5, type="post")
            for ep in episodes:
                meta = ep.get("metadata", {})
                if meta.get("zero_engagement"):
                    return True, "Zero-engagement post detected"

        return False, ""

    async def run_reflection_cycle(self) -> dict:
        """Execute the 5-step Reflexion protocol."""
        logger.info("Starting reflection cycle")

        metrics = await self._evaluate()
        reflection = await self._reflect(metrics)
        proposals = await self._propose(reflection)
        validated = await self._validate(proposals)
        result = await self._commit_or_reject(validated)

        await self._memory.remember(
            "reflection",
            f"Reflection cycle completed: {json.dumps(result, default=str)[:500]}",
            metadata={"metrics": metrics, "proposals_count": len(proposals), "accepted": result.get("accepted", 0)},
        )

        logger.info("Reflection cycle done: %s", result)
        return result

    async def _evaluate(self) -> dict:
        """Step 1: Gather performance metrics."""
        stats = await self._storage.get_stats()
        episodes = await self._storage.get_recent_episodes(limit=20)
        insights = await self._storage.get_insights(min_confidence=0.3)

        action_counts: dict[str, int] = {}
        for ep in episodes:
            t = ep.get("type", "unknown")
            action_counts[t] = action_counts.get(t, 0) + 1

        avg_importance = 0.0
        if episodes:
            avg_importance = sum(e.get("importance", 5.0) for e in episodes) / len(episodes)

        return {
            "stats": stats,
            "recent_episode_count": len(episodes),
            "action_distribution": action_counts,
            "avg_importance": round(avg_importance, 2),
            "insight_count": len(insights),
        }

    async def _reflect(self, metrics: dict) -> str:
        """Step 2: LLM self-critique."""
        strategy = await self._get_strategy()
        prompt = (
            "You are reflecting on your recent performance as a Moltbook AI agent.\n\n"
            f"Current strategy:\n{yaml.dump(strategy, default_flow_style=False)}\n\n"
            f"Performance metrics:\n{json.dumps(metrics, indent=2)}\n\n"
            "Analyze:\n"
            "1. What went well?\n"
            "2. What could be improved?\n"
            "3. Are your current interests and engagement heuristics working?\n"
            "4. Any patterns in what resonates vs what gets ignored?\n\n"
            "Be specific and honest. Write 3-5 paragraphs."
        )
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                _action="reflect",
            )
            reflection_text = resp.choices[0].message.content

            await self._memory.remember(
                "reflection_thought", reflection_text[:500],
                metadata={"metrics_snapshot": metrics},
            )
            return reflection_text
        except Exception:
            logger.exception("Reflection step failed")
            return "Reflection failed — no changes proposed."

    async def _propose(self, reflection: str) -> list[dict]:
        """Step 3: Propose strategy changes based on reflection."""
        strategy = await self._get_strategy()
        prompt = (
            "Based on this self-reflection, propose specific strategy changes.\n\n"
            f"Reflection:\n{reflection}\n\n"
            f"Current strategy:\n{yaml.dump(strategy, default_flow_style=False)}\n\n"
            "Propose 0-3 changes. Each change should be:\n"
            "- Specific and actionable\n"
            "- A modification to the strategy YAML\n\n"
            "Return ONLY a JSON array of objects:\n"
            "[{\"field\": \"path.to.field\", \"old_value\": \"...\", \"new_value\": \"...\", \"reason\": \"...\"}]\n"
            "Return [] if no changes needed."
        )
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                _action="reflect_propose",
            )
            text = resp.choices[0].message.content.strip()
            start = text.find("[")
            end = text.rfind("]") + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
            return []
        except Exception:
            logger.exception("Proposal step failed")
            return []

    async def _validate(self, proposals: list[dict]) -> list[dict]:
        """Step 4: Check proposals against constitution."""
        if not proposals:
            return []

        safety_rules = self._constitution.get("safety", {}).get("rules", [])
        values = self._constitution.get("identity", {}).get("values", [])
        perf = self._constitution.get("performance", {})

        prompt = (
            "Validate these proposed strategy changes against constitutional rules.\n\n"
            f"Constitutional values:\n" + "\n".join(f"- {v}" for v in values) + "\n\n"
            f"Safety rules:\n" + "\n".join(f"- {r}" for r in safety_rules) + "\n\n"
            f"Performance constraints:\n{json.dumps(perf, indent=2)}\n\n"
            f"Proposals:\n{json.dumps(proposals, indent=2)}\n\n"
            "For each proposal, decide if it's SAFE to apply.\n"
            "Return ONLY a JSON array with each proposal + \"approved\": true/false:\n"
            "[{\"field\": \"...\", \"new_value\": \"...\", \"reason\": \"...\", \"approved\": true/false}]"
        )
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                _action="reflect_validate",
            )
            text = resp.choices[0].message.content.strip()
            start = text.find("[")
            end = text.rfind("]") + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
            return []
        except Exception:
            logger.exception("Validation step failed")
            return []

    async def _commit_or_reject(self, validated: list[dict]) -> dict:
        """Step 5: Apply approved changes and save new version to DB."""
        approved = [p for p in validated if p.get("approved", False)]
        rejected = [p for p in validated if not p.get("approved", False)]

        if not approved:
            if validated:
                await self._storage.audit("reflection", {
                    "proposals": [
                        {
                            "field": p.get("field"), "old_value": p.get("old_value"),
                            "new_value": p.get("new_value"), "reason": p.get("reason"),
                            "approved": False,
                        }
                        for p in rejected
                    ],
                    "old_version": None, "new_version": None,
                })
            return {"accepted": 0, "rejected": len(rejected), "changes": []}

        strategy = await self._get_strategy()
        old_version = strategy.get("version", 1)
        changes_applied = []

        for proposal in approved:
            field_path = proposal.get("field", "")
            new_value = proposal.get("new_value")
            if not field_path or new_value is None:
                continue
            if _apply_nested(strategy, field_path, new_value):
                changes_applied.append(field_path)

        if changes_applied:
            strategy["version"] = old_version + 1

            await self._storage.save_strategy_version(
                version=strategy["version"],
                yaml_text=yaml.dump(strategy, default_flow_style=False),
                parent=old_version,
                trigger="reflection",
                perf=None,
            )

        all_proposals = [
            {
                "field": p.get("field"), "old_value": p.get("old_value"),
                "new_value": p.get("new_value"), "reason": p.get("reason"),
                "approved": p.get("approved", False),
            }
            for p in validated
        ]
        await self._storage.audit("reflection", {
            "proposals": all_proposals,
            "old_version": old_version,
            "new_version": strategy.get("version", old_version),
        })

        return {
            "accepted": len(approved),
            "rejected": len(rejected),
            "changes": changes_applied,
            "new_version": strategy.get("version", old_version),
        }


def _apply_nested(d: dict, path: str, value) -> bool:
    """Apply a value to a nested dict path like 'goals.mission'."""
    keys = path.split(".")
    target = d
    for key in keys[:-1]:
        if key not in target or not isinstance(target[key], dict):
            return False
        target = target[key]
    if keys:
        target[keys[-1]] = value
        return True
    return False
