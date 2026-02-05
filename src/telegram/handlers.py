from __future__ import annotations

import logging

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from src.moltbook.client import MoltbookClient
from src.storage.memory import Storage

logger = logging.getLogger(__name__)

HELP_TEXT = (
    "Available commands:\n"
    "/status — agent status & stats\n"
    "/search <query> — search Moltbook\n"
    "/ask <question> — queue a question for the LLM\n"
    "/post <submolt> <title> | <content> — create a post\n"
    "/watch <name> — follow an agent\n"
    "/unwatch <name> — unfollow an agent\n"
    "/digest — get unreported activity digest\n"
    "/pause — pause autonomous behavior\n"
    "/resume — resume autonomous behavior"
)


def register_handlers(router: Router) -> None:
    router.message.register(cmd_start, Command("start"))
    router.message.register(cmd_status, Command("status"))
    router.message.register(cmd_search, Command("search"))
    router.message.register(cmd_ask, Command("ask"))
    router.message.register(cmd_post, Command("post"))
    router.message.register(cmd_watch, Command("watch"))
    router.message.register(cmd_unwatch, Command("unwatch"))
    router.message.register(cmd_digest, Command("digest"))
    router.message.register(cmd_pause, Command("pause"))
    router.message.register(cmd_resume, Command("resume"))


async def cmd_start(message: Message) -> None:
    await message.answer(f"Shadow-Molty control panel.\n\n{HELP_TEXT}")


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
