"""Event consumer: polls agent_events, formats, translates, dispatches to Telegram."""

from __future__ import annotations

import asyncio
import json
import logging

from aiogram import Bot

from src.storage.db import Storage

logger = logging.getLogger(__name__)

# Event type → channel setting key. Events NOT in this map are owner-only.
EVENT_SETTING_MAP: dict[str, str] = {
    "post_created": "posts",
    "comment_created": "comments",
    "reply_sent": "replies",
    "dm_approved": "dms",
    "dm_replied": "dms",
    "dm_needs_human": "dms",
    "reflection_done": "reflection",
    "stability_alert": "alerts",
    "daily_newspaper": "daily_summary",
}


async def run_consumer(
    storage: Storage,
    bot: Bot,
    owner_id: int,
    llm_client,
    model: str,
    poll_interval: float = 2,
) -> None:
    """Main consumer loop — poll events, format, send to owner + channel."""
    logger.info("Event consumer started (poll every %.1fs)", poll_interval)
    try:
        while True:
            try:
                events = await storage.consume_events()
                for event in events:
                    msg = format_event(event)
                    try:
                        await bot.send_message(owner_id, msg)
                    except Exception:
                        logger.exception("Failed to send event to owner")
                    await _maybe_send_to_channel(event, msg, storage, bot, llm_client, model)
            except Exception:
                logger.exception("Event consumer tick failed")
            await asyncio.sleep(poll_interval)
    except asyncio.CancelledError:
        logger.info("Event consumer shutting down")


def format_event(event: dict) -> str:
    """Format event as human-readable English message for owner chat."""
    etype = event["type"]
    data = event.get("data", {})

    if etype == "post_created":
        content_preview = data.get("content", "")[:150]
        return (
            f"New post in s/{data.get('submolt', '?')}: {data.get('title', '?')}\n"
            f"{content_preview}"
        )

    if etype == "comment_created":
        return (
            f"Commented on '{data.get('post_title', '?')}' by {data.get('post_author', '?')}\n"
            f"{data.get('comment_text', '')[:200]}"
        )

    if etype == "reply_sent":
        return (
            f"Replied to {data.get('comment_author', '?')} on '{data.get('post_title', '?')}'\n"
            f"{data.get('reply_text', '')[:200]}"
        )

    if etype == "upvoted":
        return f"Upvoted '{data.get('post_title', '?')}' by {data.get('post_author', '?')}"

    if etype == "dm_approved":
        return f"New DM conversation with {data.get('other_agent', '?')}"

    if etype == "dm_replied":
        return (
            f"DM reply to {data.get('other_agent', '?')}\n"
            f"{data.get('reply_text', '')[:200]}"
        )

    if etype == "dm_needs_human":
        conv_id = data.get("conversation_id", "?")
        return (
            f"DM from {data.get('other_agent', '?')} needs your attention "
            f"({data.get('unread_count', 0)} unread).\n"
            f"Last: {data.get('last_message', '')[:200]}\n"
            f"Use /dm_reply {conv_id} <message>"
        )

    if etype == "reflection_done":
        return (
            f"Reflection complete: {data.get('accepted', 0)} changes applied, "
            f"{data.get('rejected', 0)} rejected.\n"
            f"Changes: {data.get('changes', [])}"
        )

    if etype == "stability_alert":
        return (
            f"Stability alert: ASI={data.get('overall', 0):.2f}\n"
            f"Components: {json.dumps(data.get('components', {}))}"
        )

    if etype == "task_result":
        return (
            f"Task #{data.get('task_id', '?')} ({data.get('task_type', '?')}):\n"
            f"{data.get('answer', data.get('result', ''))[:500]}"
        )

    if etype == "task_failed":
        return f"Task #{data.get('task_id', '?')} ({data.get('task_type', '?')}) failed: {data.get('error', '?')}"

    if etype == "heartbeat_report":
        feed = data.get("feed_summary", [])
        feed_str = "\n".join(feed[:8]) if isinstance(feed, list) else str(feed)
        return (
            f"HEARTBEAT REPORT\n\n"
            f"Feed:\n{feed_str}\n\n"
            f"Decision: {data.get('decision', '?')}\n\n"
            f"{data.get('action_detail', '')}"
        )

    if etype == "heartbeat_skip":
        return f"Heartbeat skipped ({data.get('reason', '?')})"

    if etype == "daily_newspaper":
        return data.get("text", "")

    return f"[{etype}] {json.dumps(data)[:300]}"


async def _maybe_send_to_channel(
    event: dict, formatted_msg: str,
    storage: Storage, bot: Bot,
    llm_client, model: str,
) -> None:
    """Translate and send to channel if configured and enabled."""
    channel_id = await storage.get_state("channel_id")
    if not channel_id:
        return

    active = await storage.get_state("channel_active")
    if active == "0":
        return

    setting_key = EVENT_SETTING_MAP.get(event["type"])
    if setting_key is None:
        return  # owner-only event

    enabled = await storage.get_state(f"channel_{setting_key}")
    if enabled == "0":
        return

    try:
        if event["type"] == "daily_newspaper":
            await bot.send_message(int(channel_id), formatted_msg)
        else:
            translated = await _translate(formatted_msg, llm_client, model)
            await bot.send_message(int(channel_id), translated)
    except Exception:
        logger.exception("Failed to send event to channel")


async def _translate(text: str, client, model: str) -> str:
    """Translate English text to Russian via LLM API."""
    resp = await client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                "Translate this agent activity report to Russian. "
                "Keep it concise and natural. Return ONLY the translated text.\n\n"
                + text
            ),
        }],
    )
    return resp.choices[0].message.content.strip()
