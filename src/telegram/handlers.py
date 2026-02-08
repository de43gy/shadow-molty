from __future__ import annotations

import logging

import httpx
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from src.core.persona import generate_identity
from src.moltbook.client import MoltbookClient, NameTakenError
from src.storage.db import Storage

logger = logging.getLogger(__name__)

HELP_TEXT = (
    "Available commands:\n"
    "/register — register agent on Moltbook\n"
    "/status — agent status & stats\n"
    "/search <query> — search Moltbook\n"
    "/ask <question> — queue a question for the LLM\n"
    "/post <submolt> <title> | <content> — create a post\n"
    "/watch <name> — follow an agent\n"
    "/unwatch <name> — unfollow an agent\n"
    "/digest — get unreported activity digest\n"
    "/dms — list active DM conversations\n"
    "/dm_reply <id> <message> — reply to a DM\n"
    "/reflect — trigger a reflection cycle\n"
    "/heartbeat — trigger a manual heartbeat\n"
    "/channel — channel posting settings\n"
    "/pause — pause autonomous behavior\n"
    "/resume — resume autonomous behavior"
)


def register_handlers(router: Router) -> None:
    router.message.register(cmd_start, Command("start"))
    router.message.register(cmd_register, Command("register"))
    router.message.register(cmd_status, Command("status"))
    router.message.register(cmd_search, Command("search"))
    router.message.register(cmd_ask, Command("ask"))
    router.message.register(cmd_post, Command("post"))
    router.message.register(cmd_watch, Command("watch"))
    router.message.register(cmd_unwatch, Command("unwatch"))
    router.message.register(cmd_digest, Command("digest"))
    router.message.register(cmd_dms, Command("dms"))
    router.message.register(cmd_dm_reply, Command("dm_reply"))
    router.message.register(cmd_reflect, Command("reflect"))
    router.message.register(cmd_heartbeat, Command("heartbeat"))
    router.message.register(cmd_channel, Command("channel"))
    router.message.register(cmd_pause, Command("pause"))
    router.message.register(cmd_resume, Command("resume"))


async def cmd_start(message: Message) -> None:
    await message.answer(f"Shadow-Molty control panel.\n\n{HELP_TEXT}")


async def cmd_register(message: Message, storage: Storage, moltbook: MoltbookClient) -> None:
    try:
        existing_key = await storage.get_state("moltbook_api_key")
        if existing_key or moltbook.registered:
            await message.answer("Already registered.")
            return

        await message.answer("Generating identity...")
        identity = await generate_identity()
        name = identity["name"]
        description = identity["description"]

        max_attempts = 5
        taken_names: list[str] = []
        for attempt in range(max_attempts):
            await message.answer(f"Attempt {attempt + 1}/{max_attempts}: registering as '{name}'...")
            try:
                result = await moltbook.register(name, description)
                break
            except NameTakenError:
                taken_names.append(name)
                if attempt + 1 < max_attempts:
                    identity = await generate_identity(taken_names)
                    name = identity["name"]
                    description = identity["description"]
                else:
                    await message.answer(f"Failed after {max_attempts} attempts — all names taken.")
                    return

        await storage.set_state("moltbook_api_key", result.api_key)
        await storage.set_state("agent_name", result.name)
        await storage.set_state("agent_description", description)
        await moltbook.set_api_key(result.api_key)

        await message.answer(
            f"Agent '{result.name}' registered!\n"
            f"API key saved.\n\n"
            f"Now claim your agent:\n"
            f"1. Open: {result.claim_url}\n"
            f"2. Post a tweet with code: {result.verification_code}\n"
            f"3. Once verified, the agent will start posting autonomously.\n\n"
            f"Profile: {result.profile_url}"
        )
    except Exception as e:
        logger.exception("cmd_register failed")
        await message.answer(f"Registration failed: {e}")


async def cmd_status(message: Message, storage: Storage, moltbook: MoltbookClient) -> None:
    try:
        me = await moltbook.get_me()
        stats = await storage.get_stats()
        paused_str = "PAUSED" if stats["paused"] else "active"
        text = (
            f"Agent: {me.name}\n"
            f"Karma: {me.karma}\n"
            f"State: {paused_str}\n"
            f"Posts: {stats['total_posts']}\n"
            f"Comments today: {stats['comments_today']}\n"
            f"Seen posts: {stats['seen_posts']}\n"
            f"Pending tasks: {stats['pending_tasks']}\n"
            f"Watched agents: {stats['watched_agents']}"
        )
        await message.answer(text)
    except Exception as e:
        logger.exception("cmd_status failed")
        await message.answer(f"Error: {e}")


async def cmd_search(message: Message, moltbook: MoltbookClient) -> None:
    try:
        query = (message.text or "").removeprefix("/search").strip()
        if not query:
            await message.answer("Usage: /search <query>")
            return

        results = await moltbook.search(query, limit=5)
        posts = results.get("posts", [])
        comments = results.get("comments", [])

        if not posts and not comments:
            await message.answer("Nothing found.")
            return

        lines: list[str] = []
        for p in posts[:5]:
            title = p.get("title", p.get("id", "?"))
            author = p.get("author", "?")
            lines.append(f"[post] {title}  — by {author}")
        for c in comments[:5]:
            snippet = (c.get("content", ""))[:80]
            author = c.get("author", "?")
            lines.append(f"[comment] {author}: {snippet}")

        await message.answer("\n".join(lines))
    except Exception as e:
        logger.exception("cmd_search failed")
        await message.answer(f"Error: {e}")


async def cmd_ask(message: Message, storage: Storage) -> None:
    try:
        question = (message.text or "").removeprefix("/ask").strip()
        if not question:
            await message.answer("Usage: /ask <question>")
            return

        task_id = await storage.add_task("ask", {"question": question})
        await message.answer(f"Queued task #{task_id}: {question}")
    except Exception as e:
        logger.exception("cmd_ask failed")
        await message.answer(f"Error: {e}")


async def cmd_post(message: Message, storage: Storage, moltbook: MoltbookClient) -> None:
    try:
        raw = (message.text or "").removeprefix("/post").strip()
        # Expected: <submolt> <title> | <content>
        if "|" not in raw:
            await message.answer("Usage: /post <submolt> <title> | <content>")
            return

        head, content = raw.split("|", 1)
        content = content.strip()
        parts = head.strip().split(None, 1)
        if len(parts) < 2:
            await message.answer("Usage: /post <submolt> <title> | <content>")
            return

        submolt, title = parts[0], parts[1].strip()
        post = await moltbook.create_post(submolt, title, content)
        await storage.save_own_post(post)
        await message.answer(f"Posted: {post.title} (id={post.id})")
    except Exception as e:
        logger.exception("cmd_post failed")
        await message.answer(f"Error: {e}")


async def cmd_watch(message: Message, storage: Storage, moltbook: MoltbookClient) -> None:
    try:
        name = (message.text or "").removeprefix("/watch").strip()
        if not name:
            await message.answer("Usage: /watch <name>")
            return

        await storage.watch_agent(name)
        await moltbook.follow(name)
        await message.answer(f"Now watching {name}")
    except Exception as e:
        logger.exception("cmd_watch failed")
        await message.answer(f"Error: {e}")


async def cmd_unwatch(message: Message, storage: Storage, moltbook: MoltbookClient) -> None:
    try:
        name = (message.text or "").removeprefix("/unwatch").strip()
        if not name:
            await message.answer("Usage: /unwatch <name>")
            return

        await storage.unwatch_agent(name)
        await moltbook.unfollow(name)
        await message.answer(f"Stopped watching {name}")
    except Exception as e:
        logger.exception("cmd_unwatch failed")
        await message.answer(f"Error: {e}")


async def cmd_digest(message: Message, storage: Storage) -> None:
    try:
        items = await storage.get_unreported_digest()
        if not items:
            await message.answer("No new digest items.")
            return

        lines: list[str] = []
        ids: list[int] = []
        for item in items:
            ids.append(item["id"])
            data = item["data"]
            item_type = item["type"]
            summary = str(data)[:120]
            lines.append(f"[{item_type}] {summary}")

        await message.answer("\n".join(lines))
        await storage.mark_digest_reported(ids)
    except Exception as e:
        logger.exception("cmd_digest failed")
        await message.answer(f"Error: {e}")


async def cmd_dms(message: Message, moltbook: MoltbookClient) -> None:
    try:
        conversations = await moltbook.dm_get_conversations()
        if not conversations:
            await message.answer("No DM conversations.")
            return

        lines: list[str] = []
        for conv in conversations:
            conv_id = conv.get("conversation_id") or conv.get("id", "?")
            with_agent = conv.get("with_agent", {})
            if isinstance(with_agent, dict):
                name = with_agent.get("name", "?")
            else:
                name = str(with_agent)
            unread = conv.get("unread_count", 0)
            unread_str = f" ({unread} unread)" if unread else ""
            lines.append(f"- {name}{unread_str}\n  ID: {conv_id}")

        await message.answer("DM conversations:\n" + "\n".join(lines))
    except httpx.HTTPStatusError as e:
        if e.response.status_code >= 500:
            await message.answer("Moltbook DM API is currently unavailable (server error).")
        else:
            await message.answer(f"Error: {e}")
    except Exception as e:
        logger.exception("cmd_dms failed")
        await message.answer(f"Error: {e}")


async def cmd_dm_reply(message: Message, storage: Storage, moltbook: MoltbookClient) -> None:
    try:
        raw = (message.text or "").removeprefix("/dm_reply").strip()
        parts = raw.split(None, 1)
        if len(parts) < 2:
            await message.answer("Usage: /dm_reply <conversation_id> <message>")
            return

        conv_id, text = parts[0], parts[1]
        await moltbook.dm_send(conv_id, text)
        await storage.set_dm_needs_human(conv_id, False)
        await message.answer(f"DM sent to conversation {conv_id}")
    except Exception as e:
        logger.exception("cmd_dm_reply failed")
        await message.answer(f"Error: {e}")


async def cmd_reflect(message: Message, storage: Storage) -> None:
    try:
        task_id = await storage.add_task("reflect", {})
        await message.answer(f"Reflection queued as task #{task_id}")
    except Exception as e:
        logger.exception("cmd_reflect failed")
        await message.answer(f"Error: {e}")


async def cmd_heartbeat(message: Message, storage: Storage) -> None:
    try:
        task_id = await storage.add_task("heartbeat", {})
        await message.answer(f"Manual heartbeat queued as task #{task_id}")
    except Exception as e:
        logger.exception("cmd_heartbeat failed")
        await message.answer(f"Error: {e}")


_CHANNEL_SETTINGS = ("posts", "comments", "replies", "dms", "reflection", "alerts", "daily_summary")


async def _channel_status(storage: Storage) -> str:
    channel_id = await storage.get_state("channel_id")
    if not channel_id:
        return "No channel configured. Add bot as admin to a channel to auto-detect."
    active = await storage.get_state("channel_active")
    lines = [f"Channel: {channel_id}", f"Active: {'yes' if active != '0' else 'no (paused)'}"]
    for key in _CHANNEL_SETTINGS:
        val = await storage.get_state(f"channel_{key}")
        lines.append(f"  {key}: {'on' if val != '0' else 'off'}")
    lines.append("\nCommands: /channel pause | resume | toggle <setting>")
    return "\n".join(lines)


async def cmd_channel(message: Message, storage: Storage) -> None:
    try:
        raw = (message.text or "").removeprefix("/channel").strip()
        args = raw.split()

        if not args:
            await message.answer(await _channel_status(storage))
            return

        cmd = args[0].lower()

        if cmd == "pause":
            await storage.set_state("channel_active", "0")
            await message.answer(await _channel_status(storage))
        elif cmd == "resume":
            await storage.set_state("channel_active", "1")
            await message.answer(await _channel_status(storage))
        elif cmd == "toggle" and len(args) >= 2:
            key = args[1].lower()
            if key not in _CHANNEL_SETTINGS:
                await message.answer(f"Unknown setting. Available: {', '.join(_CHANNEL_SETTINGS)}")
                return
            current = await storage.get_state(f"channel_{key}")
            new_val = "0" if current != "0" else "1"
            await storage.set_state(f"channel_{key}", new_val)
            await message.answer(await _channel_status(storage))
        else:
            await message.answer("Usage: /channel [pause|resume|toggle <setting>]")

    except Exception as e:
        logger.exception("cmd_channel failed")
        await message.answer(f"Error: {e}")


async def cmd_pause(message: Message, storage: Storage) -> None:
    try:
        await storage.set_state("paused", "1")
        await message.answer("Agent paused.")
    except Exception as e:
        logger.exception("cmd_pause failed")
        await message.answer(f"Error: {e}")


async def cmd_resume(message: Message, storage: Storage) -> None:
    try:
        await storage.set_state("paused", "0")
        await message.answer("Agent resumed.")
    except Exception as e:
        logger.exception("cmd_resume failed")
        await message.answer(f"Error: {e}")
