from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from aiogram import Bot

from src.agent.brain import Brain
from src.agent.memory import MemoryManager
from src.agent.safety import validate_action
from src.config import settings
from src.moltbook.client import MoltbookClient
from src.storage.memory import Storage

logger = logging.getLogger(__name__)


async def run_worker(
    storage: Storage,
    brain: Brain,
    bot: Bot,
    owner_id: int,
    moltbook: MoltbookClient | None = None,
    memory: MemoryManager | None = None,
    reflection_engine=None,
    poll_interval: int = 5,
) -> None:
    """Async loop that processes pending tasks from the SQLite queue."""
    logger.info("Worker started (poll every %ds)", poll_interval)
    try:
        while True:
            tasks = await storage.get_pending_tasks()
            for task in tasks:
                task_id = task["id"]
                task_type = task["type"]
                payload = task["payload"]
                logger.info("Processing task #%d type=%s", task_id, task_type)

                try:
                    if task_type == "ask":
                        answer = await brain.answer_question(payload["question"])
                        await bot.send_message(owner_id, f"Task #{task_id} answer:\n\n{answer}")
                        await storage.complete_task(task_id, {"answer": answer})
                    elif task_type == "reflect":
                        if reflection_engine:
                            result = await reflection_engine.run_reflection_cycle()
                            if result.get("changes"):
                                brain.reload_prompt()
                            msg = (
                                f"Reflection complete: {result.get('accepted', 0)} changes applied, "
                                f"{result.get('rejected', 0)} rejected."
                            )
                            if result.get("changes"):
                                msg += f"\nChanges: {result['changes']}"
                            await bot.send_message(owner_id, msg)
                            await storage.complete_task(task_id, result)
                        else:
                            await storage.fail_task(task_id, "Reflection engine not initialized")
                    elif task_type == "heartbeat":
                        if moltbook and moltbook.registered:
                            result = await _manual_heartbeat(
                                storage, brain, moltbook, bot, owner_id, memory
                            )
                            await storage.complete_task(task_id, result)
                        else:
                            await storage.fail_task(task_id, "Not registered on Moltbook")
                            await bot.send_message(owner_id, "Heartbeat failed: not registered.")
                    else:
                        logger.warning("Unknown task type: %s", task_type)
                        await storage.fail_task(task_id, f"Unknown task type: {task_type}")
                except Exception as exc:
                    logger.exception("Task #%d failed", task_id)
                    await storage.fail_task(task_id, str(exc))
                    try:
                        await bot.send_message(owner_id, f"Task #{task_id} failed: {exc}")
                    except Exception:
                        logger.exception("Failed to notify owner about task failure")

            await asyncio.sleep(poll_interval)
    except asyncio.CancelledError:
        logger.info("Worker shutting down")


def _seconds_since(iso_timestamp: str) -> float:
    """Return seconds elapsed since an ISO timestamp."""
    ts = datetime.fromisoformat(iso_timestamp)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - ts).total_seconds()


async def _check_rate_limit(storage: Storage, action: str) -> str | None:
    """Check if action is allowed. Returns error message or None if OK."""
    if action == "post":
        posts = await storage.get_own_posts(limit=1)
        if posts:
            elapsed = _seconds_since(posts[0]["created_at"])
            remaining = settings.post_cooldown_sec - elapsed
            if remaining > 0:
                return f"Post cooldown: {int(remaining)}s remaining"

    elif action == "comment":
        count = await storage.get_today_comment_count()
        if count >= settings.max_comments_per_day:
            return f"Daily comment limit reached ({count}/{settings.max_comments_per_day})"

    return None


async def _manual_heartbeat(
    storage: Storage,
    brain: Brain,
    moltbook: MoltbookClient,
    bot: Bot,
    owner_id: int,
    memory: MemoryManager | None,
) -> dict:
    """Execute a manual heartbeat with rate limit checks and detailed report."""
    stats = await storage.get_stats()
    feed = await moltbook.get_feed(sort="new", limit=15)

    for post in feed:
        await storage.mark_seen(post.id)

    # Build feed summary for report
    feed_lines = [
        f"  {p.author}: {p.title} (s/{p.submolt}, {p.upvotes} up, {p.comment_count} comments)"
        for p in feed[:8]
    ]

    decision = await brain.decide_action(feed, stats)
    action = decision.get("action", "skip")
    params = decision.get("params") or {}

    # Rate limit check
    limit_err = await _check_rate_limit(storage, action)
    if limit_err:
        report = (
            f"HEARTBEAT REPORT\n\n"
            f"Feed ({len(feed)} posts):\n" + "\n".join(feed_lines) + "\n\n"
            f"Decision: {action}\n"
            f"Blocked: {limit_err}"
        )
        await bot.send_message(owner_id, report)
        return {"action": "skip", "reason": limit_err}

    # Task Shield
    constitution = brain.identity.get("constitution", {})
    goals = brain.identity.get("strategy", {}).get("goals", {})
    safe, reason = await validate_action(
        decision, goals, constitution, brain._client, brain._model
    )
    if not safe:
        report = (
            f"HEARTBEAT REPORT\n\n"
            f"Feed ({len(feed)} posts):\n" + "\n".join(feed_lines) + "\n\n"
            f"Decision: {action}\n"
            f"Blocked by safety: {reason}"
        )
        await bot.send_message(owner_id, report)
        return {"action": "blocked", "reason": reason}

    # Execute and build report
    action_detail = ""

    if action == "post":
        feed_summary = [f"{p.submolt}: {p.title}" for p in feed[:10]]
        own_posts = await storage.get_own_posts(limit=10)
        recent_own = [p["title"] for p in own_posts]
        post_data = await brain.generate_post(feed_summary, recent_own)
        if post_data:
            post = await moltbook.create_post(
                submolt=post_data["submolt"],
                title=post_data["title"],
                content=post_data["content"],
            )
            await storage.save_own_post(post)
            await storage.add_digest_item("post", {"id": post.id, "title": post.title, "submolt": post.submolt})
            action_detail = (
                f"Posted in s/{post.submolt}:\n"
                f"  Title: {post.title}\n"
                f"  Content: {post.content[:300]}"
            )
            if memory:
                await memory.remember(
                    "post", f"Posted '{post.title}' in s/{post.submolt}: {post.content[:200]}",
                    metadata={"post_id": post.id, "submolt": post.submolt, "manual": True},
                )
        else:
            action_detail = "Failed to generate post."

    elif action == "comment":
        post_id = params.get("post_id")
        target = next((p for p in feed if p.id == post_id), None) if post_id else None
        if target:
            existing_comments = await moltbook.get_comments(post_id)
            text = await brain.generate_comment(target, existing_comments)
            if text:
                comment = await moltbook.create_comment(post_id, text)
                await storage.save_own_comment(comment)
                await storage.add_digest_item("comment", {"post_id": post_id, "post_title": target.title, "content": text[:100]})
                await storage.mark_seen(post_id, interacted=True)
                action_detail = (
                    f"Commented on '{target.title}' by {target.author}:\n"
                    f"  {text[:300]}"
                )
                if memory:
                    await memory.remember(
                        "comment", f"Commented on '{target.title}' by {target.author}: {text[:200]}",
                        metadata={"post_id": post_id, "author": target.author, "manual": True},
                    )
            else:
                action_detail = "Failed to generate comment."
        else:
            action_detail = f"Target post {post_id} not found in feed."

    elif action == "upvote":
        post_id = params.get("post_id")
        target = next((p for p in feed if p.id == post_id), None) if post_id else None
        if post_id:
            await moltbook.upvote_post(post_id)
            if target:
                action_detail = (
                    f"Upvoted '{target.title}' by {target.author} "
                    f"(s/{target.submolt}, {target.upvotes} up)"
                )
            else:
                action_detail = f"Upvoted post {post_id}"
            if memory:
                await memory.remember(
                    "upvote",
                    f"Upvoted '{target.title}' by {target.author}" if target else f"Upvoted post {post_id}",
                    metadata={"post_id": post_id, "manual": True},
                )
        else:
            action_detail = "No post_id for upvote."

    else:
        action_detail = "Decided to skip this cycle."

    # Send consolidated report
    report = (
        f"HEARTBEAT REPORT\n\n"
        f"Feed ({len(feed)} posts):\n" + "\n".join(feed_lines) + "\n\n"
        f"Decision: {action}\n\n"
        f"{action_detail}"
    )
    await bot.send_message(owner_id, report)

    return {"action": action, "params": params}
