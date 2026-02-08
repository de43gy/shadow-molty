from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.core.brain import Brain
from src.core.memory import MemoryManager
from src.core.reflection import ReflectionEngine
from src.core.safety import StabilityIndex, validate_action
from src.config import settings
from src.moltbook.client import MoltbookClient
from src.moltbook.models import Comment, Post
from src.storage.db import Storage

logger = logging.getLogger(__name__)

HEARTBEAT_JOB_ID = "heartbeat"
CONSOLIDATION_JOB_ID = "consolidation"


def create_scheduler(
    storage: Storage,
    brain: Brain,
    moltbook: MoltbookClient,
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


def _build_thread_context(target: Comment, all_comments: list[Comment]) -> list[Comment]:
    """Walk parent_id chain to build conversation thread leading to target."""
    by_id = {c.id: c for c in all_comments}
    thread: list[Comment] = []
    current = target
    while current.parent_id and current.parent_id in by_id:
        parent = by_id[current.parent_id]
        thread.append(parent)
        current = parent
    thread.reverse()
    return thread


async def _check_own_post_replies(
    storage: Storage,
    brain: Brain,
    moltbook: MoltbookClient,
    memory: MemoryManager | None = None,
    max_replies: int = 2,
) -> int:
    """Check for new comments on own posts and reply. Returns number of replies sent."""
    agent_name = await storage.get_state("agent_name")
    if not agent_name:
        return 0

    cutoff = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    own_posts = await storage.get_own_posts(limit=10)
    recent_posts = [p for p in own_posts if (p.get("created_at") or "") >= cutoff]

    if not recent_posts:
        return 0

    pending_replies: list[tuple[Post, Comment, list[Comment]]] = []

    for post_row in recent_posts:
        post_id = post_row["id"]
        try:
            comments = await moltbook.get_comments(post_id, sort="new")
        except Exception:
            logger.warning("Failed to fetch comments for post %s", post_id)
            continue

        seen_ids = await storage.get_seen_comment_ids(post_id)

        # Build a Post object for brain
        post = Post(
            id=post_id,
            author=agent_name,
            submolt=post_row.get("submolt", ""),
            title=post_row.get("title", ""),
            content=post_row.get("content", ""),
        )

        for comment in comments:
            # Skip own comments
            if comment.author.lower() == agent_name.lower():
                if comment.id not in seen_ids:
                    await storage.mark_comment_seen(comment.id, post_id, replied=True)
                continue

            if comment.id in seen_ids:
                continue

            # Queue reply for: top-level comments or direct replies to our comments
            is_top_level = not comment.parent_id
            is_reply_to_us = False
            if comment.parent_id:
                parent = next((c for c in comments if c.id == comment.parent_id), None)
                if parent and parent.author.lower() == agent_name.lower():
                    is_reply_to_us = True

            if is_top_level or is_reply_to_us:
                # Don't mark seen yet — defer until actually replied
                thread_ctx = _build_thread_context(comment, comments)
                pending_replies.append((post, comment, thread_ctx))
            else:
                # Not reply-eligible — mark seen and move on
                await storage.mark_comment_seen(comment.id, post_id, replied=False)

    # Process replies FIFO (oldest first), capped at max_replies
    pending_replies.sort(key=lambda x: x[1].created_at or datetime.min.replace(tzinfo=timezone.utc))
    replies_sent = 0

    for post, comment, thread_ctx in pending_replies[:max_replies]:
        try:
            text = await brain.generate_reply(post, comment, thread_ctx)
            if not text:
                logger.warning("Brain returned empty reply for comment %s", comment.id)
                await storage.mark_comment_seen(comment.id, post.id, replied=False)
                continue

            reply = await moltbook.create_comment(post.id, text, parent_id=comment.id)
            await storage.save_own_comment(reply)
            await storage.mark_comment_seen(comment.id, post.id, replied=True)
            replies_sent += 1

            try:
                await moltbook.upvote_comment(comment.id)
            except Exception:
                logger.debug("Failed to upvote comment %s", comment.id)

            logger.info("Replied to %s on post %s", comment.author, post.id)

            if memory:
                await memory.remember(
                    "reply",
                    f"Replied to {comment.author}'s comment on '{post.title}': {text[:200]}",
                    metadata={"post_id": post.id, "comment_id": comment.id, "author": comment.author},
                )

            await storage.emit_event("reply_sent", {
                "post_title": post.title, "comment_author": comment.author,
                "reply_text": text[:300],
            })

        except RuntimeError:
            logger.info("Comment rate limit reached, stopping replies")
            break
        except Exception:
            logger.exception("Failed to reply to comment %s", comment.id)

    if replies_sent:
        logger.info("Replied to %d comments on own posts", replies_sent)
    return replies_sent


async def _check_dms(
    storage: Storage,
    brain: Brain,
    moltbook: MoltbookClient,
    memory: MemoryManager | None = None,
) -> None:
    """Check DMs: auto-approve requests, reply to new messages."""
    try:
        check = await moltbook.dm_check()
    except Exception:
        logger.warning("DM check failed (endpoint may not exist yet)")
        return

    has_activity = check.get("has_activity", False)
    if not has_activity:
        logger.debug("DM check: no activity")
        return

    agent_name = await storage.get_state("agent_name") or ""

    # Auto-approve pending requests
    try:
        requests = await moltbook.dm_get_requests()
        for req in requests:
            conv_id = req.get("conversation_id") or req.get("id", "")
            from_agent = req.get("from", {})
            if isinstance(from_agent, dict):
                from_name = from_agent.get("name", "unknown")
            else:
                from_name = str(from_agent)

            try:
                await moltbook.dm_approve(conv_id)
                await storage.upsert_dm_conversation(conv_id, from_name)
                logger.info("Auto-approved DM request from %s", from_name)
                await storage.emit_event("dm_approved", {"other_agent": from_name})
            except Exception:
                logger.exception("Failed to approve DM from %s", from_name)
    except Exception:
        logger.warning("Failed to fetch DM requests")

    # Process conversations with unread messages
    try:
        conversations = await moltbook.dm_get_conversations()
    except Exception:
        logger.warning("Failed to fetch DM conversations")
        return

    for conv in conversations:
        conv_id = conv.get("conversation_id") or conv.get("id", "")
        unread = conv.get("unread_count", 0)
        if not unread:
            continue

        with_agent = conv.get("with_agent", {})
        if isinstance(with_agent, dict):
            other_name = with_agent.get("name", "unknown")
        else:
            other_name = str(with_agent)

        await storage.upsert_dm_conversation(conv_id, other_name)

        # Check if flagged for human
        db_conv = await storage.get_dm_conversation(conv_id)
        if db_conv and db_conv.get("needs_human"):
            await storage.emit_event("dm_needs_human", {
                "other_agent": other_name, "conversation_id": conv_id,
                "unread_count": unread,
            })
            continue

        # Fetch messages
        try:
            messages = await moltbook.dm_get_messages(conv_id)
        except Exception:
            logger.warning("Failed to fetch messages for DM %s", conv_id)
            continue

        if not messages:
            continue

        # Find new messages (after last_seen watermark)
        last_seen_id = (db_conv or {}).get("last_seen_message_id")
        if last_seen_id:
            found_watermark = False
            new_messages = []
            for msg in messages:
                if found_watermark:
                    new_messages.append(msg)
                elif msg.get("id") == last_seen_id:
                    found_watermark = True
            if not found_watermark:
                new_messages = messages
        else:
            new_messages = messages

        # Skip if no new messages from others
        incoming = [m for m in new_messages if m.get("sender", {}).get("name", m.get("sender", "")) != agent_name]
        if not incoming:
            # Update watermark anyway
            if messages:
                await storage.update_dm_last_seen(conv_id, messages[-1].get("id", ""))
            continue

        # Generate reply
        try:
            # Normalize message format for brain
            normalized = []
            for msg in messages[-10:]:
                sender = msg.get("sender", {})
                if isinstance(sender, dict):
                    sender_name = sender.get("name", "unknown")
                else:
                    sender_name = str(sender)
                normalized.append({"sender": sender_name, "content": msg.get("content", "")})

            result = await brain.generate_dm_reply(other_name, normalized)

            if not result.get("content"):
                logger.warning("Brain returned empty DM reply for %s", other_name)
                continue

            if result.get("needs_human_input"):
                await storage.set_dm_needs_human(conv_id, True)
                await storage.emit_event("dm_needs_human", {
                    "other_agent": other_name, "conversation_id": conv_id,
                    "last_message": incoming[-1].get("content", "")[:200],
                    "unread_count": unread,
                })
            else:
                await moltbook.dm_send(conv_id, result["content"])
                logger.info("Sent DM reply to %s", other_name)
                await storage.emit_event("dm_replied", {
                    "other_agent": other_name,
                    "reply_text": result["content"][:300],
                })

                if memory:
                    await memory.remember(
                        "dm",
                        f"DM with {other_name}: {result['content'][:200]}",
                        metadata={"conversation_id": conv_id, "other_agent": other_name},
                    )

        except Exception:
            logger.exception("Failed to process DM conversation %s", conv_id)

        # Update watermark
        if messages:
            await storage.update_dm_last_seen(conv_id, messages[-1].get("id", ""))


async def _heartbeat(
    scheduler: AsyncIOScheduler,
    storage: Storage,
    brain: Brain,
    moltbook: MoltbookClient,
    memory: MemoryManager | None = None,
    reflection: ReflectionEngine | None = None,
) -> None:
    # Reschedule with new random interval for next run
    next_delay = random.randint(settings.heartbeat_min_sec, settings.heartbeat_max_sec)
    scheduler.reschedule_job(HEARTBEAT_JOB_ID, trigger=IntervalTrigger(seconds=next_delay))
    logger.info("Next heartbeat in %ds", next_delay)

    if not moltbook.registered:
        logger.info("Heartbeat skipped (not registered)")
        await storage.emit_event("heartbeat_skip", {"reason": "not registered"})
        return

    paused = await storage.get_state("paused")
    if paused == "1":
        logger.info("Heartbeat skipped (paused)")
        await storage.emit_event("heartbeat_skip", {"reason": "paused"})
        return

    # Set heartbeat lock
    await storage.set_state("heartbeat_running", "1")

    try:
        # Increment heartbeat counter
        hb_count = await storage.get_state("heartbeat_count")
        count = int(hb_count) + 1 if hb_count else 1
        await storage.set_state("heartbeat_count", str(count))

        # Phase 1: Obligations — reply to comments on own posts, check DMs
        try:
            replies_sent = await _check_own_post_replies(
                storage, brain, moltbook, memory
            )
        except Exception:
            logger.exception("_check_own_post_replies failed")
            replies_sent = 0

        try:
            await _check_dms(storage, brain, moltbook, memory)
        except Exception:
            logger.exception("_check_dms failed")

        # Phase 2: Autonomous action
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
            await _do_post(storage, brain, moltbook, feed, memory)
        elif action == "comment":
            await _do_comment(storage, brain, moltbook, feed, params, memory)
        elif action == "upvote":
            post_id = params.get("post_id")
            if post_id:
                await moltbook.upvote_post(post_id)
                logger.info("Upvoted post %s", post_id)
                target = next((p for p in feed if p.id == post_id), None)
                await storage.emit_event("upvoted", {
                    "post_id": post_id,
                    "post_title": target.title if target else "",
                    "post_author": target.author if target else "",
                })
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
            await storage.emit_event("stability_alert", {
                "overall": stability["overall"],
                "components": stability["components"],
            })

        # Reflection trigger
        if reflection:
            should, trigger_reason = await reflection.should_trigger()
            if should:
                logger.info("Reflection triggered: %s", trigger_reason)
                result = await reflection.run_reflection_cycle()
                if result.get("changes"):
                    new_strategy = await storage.get_strategy()
                    brain.reload_prompt(strategy=new_strategy)
                    await storage.emit_event("reflection_done", {
                        "accepted": result.get("accepted", 0),
                        "rejected": result.get("rejected", 0),
                        "changes": result["changes"],
                    })

    except Exception:
        logger.exception("Heartbeat failed")
    finally:
        await storage.set_state("heartbeat_running", "0")


async def _do_post(
    storage: Storage,
    brain: Brain,
    moltbook: MoltbookClient,
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

    await storage.emit_event("post_created", {
        "submolt": post.submolt, "title": post.title,
        "content": (post.content or "")[:300], "post_id": post.id,
    })


async def _do_comment(
    storage: Storage,
    brain: Brain,
    moltbook: MoltbookClient,
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

    try:
        await moltbook.upvote_post(post_id)
    except Exception:
        logger.debug("Failed to upvote post %s after commenting", post_id)

    logger.info("Commented on post %s", post_id)

    if memory:
        await memory.remember(
            "comment",
            f"Commented on '{target.title}' by {target.author}: {text[:200]}",
            metadata={"post_id": post_id, "author": target.author},
        )

    await storage.emit_event("comment_created", {
        "post_title": target.title, "post_author": target.author,
        "comment_text": text[:300],
    })


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
