from __future__ import annotations

import logging
import random

from aiogram import Bot
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.agent.brain import Brain
from src.config import settings
from src.moltbook.client import MoltbookClient
from src.storage.memory import Storage

logger = logging.getLogger(__name__)

HEARTBEAT_JOB_ID = "heartbeat"


def create_scheduler(
    storage: Storage,
    brain: Brain,
    moltbook: MoltbookClient,
    bot: Bot,
    owner_id: int,
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
        },
    )
    logger.info("Scheduler created (first heartbeat in %ds)", initial_delay)
    return scheduler


async def _heartbeat(
    scheduler: AsyncIOScheduler,
    storage: Storage,
    brain: Brain,
    moltbook: MoltbookClient,
    bot: Bot,
    owner_id: int,
) -> None:
    # Reschedule with new random interval for next run
    next_delay = random.randint(settings.heartbeat_min_sec, settings.heartbeat_max_sec)
    scheduler.reschedule_job(HEARTBEAT_JOB_ID, trigger=IntervalTrigger(seconds=next_delay))
    logger.info("Next heartbeat in %ds", next_delay)

    paused = await storage.get_state("paused")
    if paused == "1":
        logger.info("Heartbeat skipped (paused)")
        return

    try:
        stats = await storage.get_stats()
        feed = await moltbook.get_feed(sort="new", limit=15)

        for post in feed:
            await storage.mark_seen(post.id)

        decision = await brain.decide_action(feed, stats)
        action = decision.get("action", "skip")
        params = decision.get("params") or {}

        logger.info("Heartbeat decision: %s %s", action, params)

        if action == "post":
            await _do_post(storage, brain, moltbook, bot, owner_id, feed)
        elif action == "comment":
            await _do_comment(storage, brain, moltbook, bot, owner_id, feed, params)
        elif action == "upvote":
            post_id = params.get("post_id")
            if post_id:
                await moltbook.upvote_post(post_id)
                logger.info("Upvoted post %s", post_id)
        else:
            logger.info("Heartbeat: skip")

    except Exception:
        logger.exception("Heartbeat failed")


async def _do_post(
    storage: Storage,
    brain: Brain,
    moltbook: MoltbookClient,
    bot: Bot,
    owner_id: int,
    feed: list,
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

    try:
        await bot.send_message(owner_id, f"Commented on: {target.title}")
    except Exception:
        logger.exception("Failed to notify owner about new comment")
