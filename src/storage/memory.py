from __future__ import annotations

import json
from datetime import datetime, timezone

import aiosqlite

from src.moltbook.models import Comment, Post

_SCHEMA = """
CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS own_posts (
    id TEXT PRIMARY KEY,
    submolt TEXT,
    title TEXT,
    content TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS own_comments (
    id TEXT PRIMARY KEY,
    post_id TEXT,
    content TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS seen_posts (
    post_id TEXT PRIMARY KEY,
    interacted INTEGER DEFAULT 0,
    seen_at TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT,
    payload TEXT,
    status TEXT DEFAULT 'pending',
    result TEXT,
    created_at TEXT,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS watched_agents (
    name TEXT PRIMARY KEY,
    added_at TEXT
);

CREATE TABLE IF NOT EXISTS digest_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT,
    data TEXT,
    reported INTEGER DEFAULT 0,
    created_at TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts(value: datetime | None) -> str:
    if value is None:
        return _now()
    return value.isoformat()


class Storage:
    def __init__(self, db_path: str = "data/agent.db") -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        assert self._db is not None, "Storage not initialized — call init() first"
        return self._db

    # ── State KV ──────────────────────────────────────────────

    async def get_state(self, key: str) -> str | None:
        cur = await self.db.execute("SELECT value FROM state WHERE key = ?", (key,))
        row = await cur.fetchone()
        return row["value"] if row else None

    async def set_state(self, key: str, value: str) -> None:
        await self.db.execute(
            "INSERT INTO state (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (key, value, _now()),
        )
        await self.db.commit()

    # ── Own posts / comments ──────────────────────────────────

    async def save_own_post(self, post: Post) -> None:
        await self.db.execute(
            "INSERT OR REPLACE INTO own_posts (id, submolt, title, content, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (post.id, post.submolt, post.title, post.content, _ts(post.created_at)),
        )
        await self.db.commit()

    async def save_own_comment(self, comment: Comment) -> None:
        await self.db.execute(
            "INSERT OR REPLACE INTO own_comments (id, post_id, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (comment.id, comment.post_id, comment.content, _ts(comment.created_at)),
        )
        await self.db.commit()

    async def get_own_posts(self, limit: int = 50) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM own_posts ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def get_today_comment_count(self) -> int:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cur = await self.db.execute(
            "SELECT COUNT(*) as cnt FROM own_comments WHERE created_at LIKE ?",
            (f"{today}%",),
        )
        row = await cur.fetchone()
        return row["cnt"]

    # ── Seen posts ────────────────────────────────────────────

    async def mark_seen(self, post_id: str, interacted: bool = False) -> None:
        await self.db.execute(
            "INSERT INTO seen_posts (post_id, interacted, seen_at) VALUES (?, ?, ?) "
            "ON CONFLICT(post_id) DO UPDATE SET interacted = MAX(seen_posts.interacted, excluded.interacted)",
            (post_id, int(interacted), _now()),
        )
        await self.db.commit()

    async def is_seen(self, post_id: str) -> bool:
        cur = await self.db.execute(
            "SELECT 1 FROM seen_posts WHERE post_id = ?", (post_id,)
        )
        return await cur.fetchone() is not None

    # ── Tasks ─────────────────────────────────────────────────

    async def add_task(self, type: str, payload: dict) -> int:
        cur = await self.db.execute(
            "INSERT INTO tasks (type, payload, status, created_at) VALUES (?, ?, 'pending', ?)",
            (type, json.dumps(payload), _now()),
        )
        await self.db.commit()
        return cur.lastrowid  # type: ignore[return-value]

    async def get_pending_tasks(self) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM tasks WHERE status = 'pending' ORDER BY id"
        )
        rows = await cur.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d["payload"]) if d["payload"] else {}
            result.append(d)
        return result

    async def complete_task(self, task_id: int, result: dict) -> None:
        await self.db.execute(
            "UPDATE tasks SET status = 'done', result = ?, completed_at = ? WHERE id = ?",
            (json.dumps(result), _now(), task_id),
        )
        await self.db.commit()

    async def fail_task(self, task_id: int, error: str) -> None:
        await self.db.execute(
            "UPDATE tasks SET status = 'failed', result = ?, completed_at = ? WHERE id = ?",
            (json.dumps({"error": error}), _now(), task_id),
        )
        await self.db.commit()

    # ── Watched agents ────────────────────────────────────────

    async def watch_agent(self, name: str) -> None:
        await self.db.execute(
            "INSERT OR IGNORE INTO watched_agents (name, added_at) VALUES (?, ?)",
            (name, _now()),
        )
        await self.db.commit()

    async def unwatch_agent(self, name: str) -> None:
        await self.db.execute("DELETE FROM watched_agents WHERE name = ?", (name,))
        await self.db.commit()

    async def get_watched_agents(self) -> list[str]:
        cur = await self.db.execute(
            "SELECT name FROM watched_agents ORDER BY added_at"
        )
        rows = await cur.fetchall()
        return [r["name"] for r in rows]

    # ── Digest ────────────────────────────────────────────────

    async def add_digest_item(self, type: str, data: dict) -> None:
        await self.db.execute(
            "INSERT INTO digest_items (type, data, reported, created_at) VALUES (?, ?, 0, ?)",
            (type, json.dumps(data), _now()),
        )
        await self.db.commit()

    async def get_unreported_digest(self) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM digest_items WHERE reported = 0 ORDER BY id"
        )
        rows = await cur.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["data"] = json.loads(d["data"]) if d["data"] else {}
            result.append(d)
        return result

    async def mark_digest_reported(self, ids: list[int]) -> None:
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        await self.db.execute(
            f"UPDATE digest_items SET reported = 1 WHERE id IN ({placeholders})",
            ids,
        )
        await self.db.commit()

    # ── Stats ─────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        posts_cur = await self.db.execute("SELECT COUNT(*) as cnt FROM own_posts")
        posts_row = await posts_cur.fetchone()

        comments_today = await self.get_today_comment_count()

        seen_cur = await self.db.execute("SELECT COUNT(*) as cnt FROM seen_posts")
        seen_row = await seen_cur.fetchone()

        pending_cur = await self.db.execute(
            "SELECT COUNT(*) as cnt FROM tasks WHERE status = 'pending'"
        )
        pending_row = await pending_cur.fetchone()

        watched_cur = await self.db.execute(
            "SELECT COUNT(*) as cnt FROM watched_agents"
        )
        watched_row = await watched_cur.fetchone()

        paused = await self.get_state("paused")

        return {
            "total_posts": posts_row["cnt"],
            "comments_today": comments_today,
            "seen_posts": seen_row["cnt"],
            "pending_tasks": pending_row["cnt"],
            "watched_agents": watched_row["cnt"],
            "paused": paused == "1" if paused else False,
        }
