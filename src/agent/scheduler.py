from __future__ import annotations

import json
import logging
import random

from aiogram import Bot
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.agent.brain import Brain
from src.agent.memory import MemoryManager
from src.agent.reflection import ReflectionEngine
from src.agent.safety import StabilityIndex, validate_action
from src.config import settings
from src.moltbook.client import MoltbookClient
from src.storage.memory import Storage

logger = logging.getLogger(__name__)

HEARTBEAT_JOB_ID = "heartbeat"
CONSOLIDATION_JOB_ID = "consolidation"


def create_scheduler(
    storage: Storage,
    brain: Brain,
    moltbook: MoltbookClient,
    bot: Bot,
    owner_id: int,
    memory: MemoryManager | None = None,
    reflection: ReflectionEngine | None = None,
    consolidation_engine=None,
) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler()

    initial_delay = random.randint(settings.heartbeat_min_sec, settings.heartbeat_max_sec)
    scheduler.add_job(
        _heartbeat,
        trigger=IntervalTrigger(seconds=initial_delay),
        id=HEARTBEAT_JOB_ID,
        kwargs={
            "scheduler": scheduler,
            "storage": storage,
            "brain": brain,
            "moltbook": moltbook,
            "bot": bot,
            "owner_id": owner_id,
            "memory": memory,
            "reflection": reflection,
        },
    )
    logger.info("Scheduler created (first heartbeat in %ds)", initial_delay)

    # Consolidation job
    if consolidation_engine:
        scheduler.add_job(
            _consolidation_tick,
            trigger=IntervalTrigger(minutes=settings.consolidation_interval_min),
            id=CONSOLIDATION_JOB_ID,
            kwargs={
                "storage": storage,
                "consolidation_engine": consolidation_engine,
            },
        )
        logger.info("Consolidation job scheduled (every %d min)", settings.consolidation_interval_min)

    return scheduler


async def _heartbeat(
    scheduler: AsyncIOScheduler,
    storage: Storage,
    brain: Brain,
    moltbook: MoltbookClient,
    bot: Bot,
    owner_id: int,
    memory: MemoryManager | None = None,
    reflection: ReflectionEngine | None = None,
) -> None:
    # Reschedule with new random interval for next run
    next_delay = random.randint(settings.heartbeat_min_sec, settings.heartbeat_max_sec)
    scheduler.reschedule_job(HEARTBEAT_JOB_ID, trigger=IntervalTrigger(seconds=next_delay))
    logger.info("Next heartbeat in %ds", next_delay)

    if not moltbook.registered:
        logger.info("Heartbeat skipped (not registered)")
        return

    paused = await storage.get_state("paused")
    if paused == "1":
        logger.info("Heartbeat skipped (paused)")
        return

    # Set heartbeat lock
    await storage.set_state("heartbeat_running", "1")

    try:
        # Increment heartbeat counter
        hb_count = await storage.get_state("heartbeat_count")
        count = int(hb_count) + 1 if hb_count else 1
        await storage.set_state("heartbeat_count", str(count))

        stats = await storage.get_stats()
        feed = await moltbook.get_feed(sort="new", limit=15)

        for post in feed:
            await storage.mark_seen(post.id)

        decision = await brain.decide_action(feed, stats)
        action = decision.get("action", "skip")
        params = decision.get("params") or {}

        logger.info("Heartbeat decision: %s %s", action, params)

        # Task Shield: validate action against goals
        constitution = brain.identity.get("constitution", {})
        strategy = brain.identity.get("strategy", {})
        goals = strategy.get("goals", {})

        safe, reason = await validate_action(
            decision, goals, constitution, brain._client, brain._model
        )
        if not safe:
            logger.warning("Action blocked by Task Shield: %s", reason)
            if memory:
                await memory.remember(
                    "safety_block",
                    f"Action '{action}' blocked: {reason}",
                    metadata={"action": action, "reason": reason},
                )
            action = "skip"

        # Execute action
        if action == "post":
            await _do_post(storage, brain, moltbook, bot, owner_id, feed, memory)
        elif action == "comment":
            await _do_comment(storage, brain, moltbook, bot, owner_id, feed, params, memory)
        elif action == "upvote":
            post_id = params.get("post_id")
            if post_id:
                await moltbook.upvote_post(post_id)
                logger.info("Upvoted post %s", post_id)
                if memory:
                    await memory.remember("upvote", f"Upvoted post {post_id}")
        else:
            logger.info("Heartbeat: skip")
            if memory:
                await memory.remember("skip", "Skipped this heartbeat cycle", metadata={"count": count})

        # Stability Index check
        asi = StabilityIndex(storage)
        stability = await asi.compute()
        if stability.get("alert"):
            logger.warning("Stability alert! ASI=%.3f %s", stability["overall"], stability["components"])
            try:
                await bot.send_message(
                    owner_id,
                    f"Stability alert: ASI={stability['overall']:.2f}\n"
                    f"Components: {json.dumps(stability['components'])}",
                )
            except Exception:
                logger.exception("Failed to send stability alert")

        # Reflection trigger
        if reflection:
            should, trigger_reason = await reflection.should_trigger()
            if should:
                logger.info("Reflection triggered: %s", trigger_reason)
                result = await reflection.run_reflection_cycle()
                if result.get("changes"):
                    brain.reload_prompt()
                    try:
                        await bot.send_message(
                            owner_id,
                            f"Reflection complete: {result['accepted']} changes applied, "
                            f"{result['rejected']} rejected.\nChanges: {result['changes']}",
                        )
                    except Exception:
                        logger.exception("Failed to notify about reflection")

    except Exception:
        logger.exception("Heartbeat failed")
    finally:
        await storage.set_state("heartbeat_running", "0")


async def _do_post(
    storage: Storage,
    brain: Brain,
    moltbook: MoltbookClient,
    bot: Bot,
    owner_id: int,
    feed: list,
    memory: MemoryManager | None = None,
) -> None:
    feed_summary = [f"{p.submolt}: {p.title}" for p in feed[:10]]
    own_posts = await storage.get_own_posts(limit=10)
    recent_own = [p["title"] for p in own_posts]

    post_data = await brain.generate_post(feed_summary, recent_own)
    if not post_data:
        logger.warning("Brain returned empty post data")
        return

    post = await moltbook.create_post(
        submolt=post_data["submolt"],
        title=post_data["title"],
        content=post_data["content"],
    )
    await storage.save_own_post(post)
    await storage.add_digest_item("post", {"id": post.id, "title": post.title, "submolt": post.submolt})
    logger.info("Created post: %s (id=%s)", post.title, post.id)

    if memory:
        await memory.remember(
            "post",
            f"Posted '{post.title}' in s/{post.submolt}: {post.content[:200]}",
            metadata={"post_id": post.id, "submolt": post.submolt},
        )

    try:
        await bot.send_message(owner_id, f"New post in s/{post.submolt}: {post.title}")
    except Exception:
        logger.exception("Failed to notify owner about new post")


async def _do_comment(
    storage: Storage,
    brain: Brain,
    moltbook: MoltbookClient,
    bot: Bot,
    owner_id: int,
    feed: list,
    params: dict,
    memory: MemoryManager | None = None,
) -> None:
    post_id = params.get("post_id")
    if not post_id:
        logger.warning("comment action without post_id")
        return

    target = None
    for p in feed:
        if p.id == post_id:
            target = p
            break

    if not target:
        logger.warning("Post %s not found in feed", post_id)
        return

    existing_comments = await moltbook.get_comments(post_id)
    text = await brain.generate_comment(target, existing_comments)
    if not text:
        logger.warning("Brain returned empty comment")
        return

    comment = await moltbook.create_comment(post_id, text)
    await storage.save_own_comment(comment)
    await storage.add_digest_item("comment", {"post_id": post_id, "post_title": target.title, "content": text[:100]})
    await storage.mark_seen(post_id, interacted=True)
    logger.info("Commented on post %s", post_id)

    if memory:
        await memory.remember(
            "comment",
            f"Commented on '{target.title}' by {target.author}: {text[:200]}",
            metadata={"post_id": post_id, "author": target.author},
        )

    try:
        await bot.send_message(owner_id, f"Commented on: {target.title}")
    except Exception:
        logger.exception("Failed to notify owner about new comment")


async def _consolidation_tick(
    storage: Storage,
    consolidation_engine,
) -> None:
    """Run consolidation if not paused and heartbeat isn't running."""
    paused = await storage.get_state("paused")
    if paused == "1":
        return

    hb_running = await storage.get_state("heartbeat_running")
    if hb_running == "1":
        logger.info("Consolidation skipped (heartbeat running)")
        return

    try:
        await consolidation_engine.run_consolidation()
    except Exception:
        logger.exception("Consolidation tick failed")
