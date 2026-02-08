from __future__ import annotations

import json
from datetime import datetime, timezone

import aiosqlite


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

CREATE TABLE IF NOT EXISTS strategy_versions (
    version INTEGER PRIMARY KEY,
    strategy_yaml TEXT NOT NULL,
    parent_version INTEGER,
    trigger TEXT,
    performance_snapshot TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS core_memory (
    block TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    char_limit INTEGER DEFAULT 1000,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    importance REAL DEFAULT 5.0,
    metadata TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    insight TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    confidence REAL DEFAULT 0.5,
    evidence_count INTEGER DEFAULT 1,
    source_episode_ids TEXT,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS seen_comments (
    comment_id TEXT PRIMARY KEY,
    post_id TEXT NOT NULL,
    replied INTEGER DEFAULT 0,
    seen_at TEXT
);

CREATE TABLE IF NOT EXISTS dm_conversations (
    conversation_id TEXT PRIMARY KEY,
    other_agent TEXT NOT NULL,
    last_seen_message_id TEXT,
    needs_human INTEGER DEFAULT 0,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS agent_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    data TEXT NOT NULL,
    consumed INTEGER DEFAULT 0,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    data TEXT NOT NULL,
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

    async def set_state_default(self, key: str, value: str) -> None:
        """Set state only if key doesn't exist yet (atomic, no race)."""
        await self.db.execute(
            "INSERT OR IGNORE INTO state (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, _now()),
        )
        await self.db.commit()

    # ── Own posts / comments ──────────────────────────────────

    async def save_own_post(self, post) -> None:
        await self.db.execute(
            "INSERT OR REPLACE INTO own_posts (id, submolt, title, content, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (post.id, post.submolt, post.title, post.content, _ts(post.created_at)),
        )
        await self.db.commit()

    async def save_own_comment(self, comment) -> None:
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

    # ── Strategy versions ─────────────────────────────────────

    async def save_strategy_version(
        self, version: int, yaml_text: str, parent: int | None, trigger: str, perf: dict | None = None
    ) -> None:
        await self.db.execute(
            "INSERT OR REPLACE INTO strategy_versions "
            "(version, strategy_yaml, parent_version, trigger, performance_snapshot, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (version, yaml_text, parent, trigger, json.dumps(perf) if perf else None, _now()),
        )
        await self.db.commit()

    async def get_strategy_version(self, version: int) -> dict | None:
        cur = await self.db.execute(
            "SELECT * FROM strategy_versions WHERE version = ?", (version,)
        )
        row = await cur.fetchone()
        return dict(row) if row else None

    async def get_latest_strategy_version(self) -> dict | None:
        cur = await self.db.execute(
            "SELECT * FROM strategy_versions ORDER BY version DESC LIMIT 1"
        )
        row = await cur.fetchone()
        return dict(row) if row else None

    async def get_strategy(self) -> dict | None:
        """Get latest strategy as parsed dict, or None if no versions saved."""
        row = await self.get_latest_strategy_version()
        if row and row.get("strategy_yaml"):
            import yaml
            return yaml.safe_load(row["strategy_yaml"])
        return None

    async def get_strategy_history(self, limit: int = 10) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM strategy_versions ORDER BY version DESC LIMIT ?", (limit,)
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    # ── Core memory ────────────────────────────────────────────

    async def get_core_block(self, block: str) -> dict | None:
        cur = await self.db.execute(
            "SELECT * FROM core_memory WHERE block = ?", (block,)
        )
        row = await cur.fetchone()
        return dict(row) if row else None

    async def set_core_block(self, block: str, content: str, char_limit: int = 1000) -> None:
        await self.db.execute(
            "INSERT INTO core_memory (block, content, char_limit, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(block) DO UPDATE SET content = excluded.content, updated_at = excluded.updated_at",
            (block, content[:char_limit], char_limit, _now()),
        )
        await self.db.commit()

    async def get_all_core_blocks(self) -> list[dict]:
        cur = await self.db.execute("SELECT * FROM core_memory ORDER BY block")
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    # ── Episodes ───────────────────────────────────────────────

    async def add_episode(
        self, type: str, content: str, importance: float = 5.0, metadata: dict | None = None
    ) -> int:
        cur = await self.db.execute(
            "INSERT INTO episodes (type, content, importance, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
            (type, content, importance, json.dumps(metadata) if metadata else None, _now()),
        )
        await self.db.commit()
        return cur.lastrowid  # type: ignore[return-value]

    async def get_recent_episodes(self, limit: int = 50, type: str | None = None) -> list[dict]:
        if type:
            cur = await self.db.execute(
                "SELECT * FROM episodes WHERE type = ? ORDER BY id DESC LIMIT ?", (type, limit)
            )
        else:
            cur = await self.db.execute(
                "SELECT * FROM episodes ORDER BY id DESC LIMIT ?", (limit,)
            )
        rows = await cur.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["metadata"] = json.loads(d["metadata"]) if d["metadata"] else {}
            result.append(d)
        return result

    async def get_episodes_older_than(self, hours: int, importance_below: float) -> list[dict]:
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        cur = await self.db.execute(
            "SELECT * FROM episodes WHERE created_at < ? AND importance < ? ORDER BY id",
            (cutoff, importance_below),
        )
        rows = await cur.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["metadata"] = json.loads(d["metadata"]) if d["metadata"] else {}
            result.append(d)
        return result

    async def delete_episodes(self, ids: list[int]) -> None:
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        await self.db.execute(f"DELETE FROM episodes WHERE id IN ({placeholders})", ids)
        await self.db.commit()

    async def get_episode_count(self) -> int:
        cur = await self.db.execute("SELECT COUNT(*) as cnt FROM episodes")
        row = await cur.fetchone()
        return row["cnt"]

    async def search_episodes(self, keywords: list[str], limit: int = 20) -> list[dict]:
        if not keywords:
            return await self.get_recent_episodes(limit)
        conditions = " OR ".join(["content LIKE ?"] * len(keywords))
        params = [f"%{kw}%" for kw in keywords]
        params.append(limit)  # type: ignore[arg-type]
        cur = await self.db.execute(
            f"SELECT * FROM episodes WHERE ({conditions}) ORDER BY id DESC LIMIT ?",
            params,
        )
        rows = await cur.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["metadata"] = json.loads(d["metadata"]) if d["metadata"] else {}
            result.append(d)
        return result

    # ── Insights ───────────────────────────────────────────────

    async def add_insight(
        self, insight: str, category: str = "general", confidence: float = 0.5, source_episode_ids: list[int] | None = None
    ) -> int:
        cur = await self.db.execute(
            "INSERT INTO insights (insight, category, confidence, evidence_count, source_episode_ids, created_at, updated_at) "
            "VALUES (?, ?, ?, 1, ?, ?, ?)",
            (insight, category, confidence, json.dumps(source_episode_ids or []), _now(), _now()),
        )
        await self.db.commit()
        return cur.lastrowid  # type: ignore[return-value]

    async def get_insights(self, category: str | None = None, min_confidence: float = 0.3) -> list[dict]:
        if category:
            cur = await self.db.execute(
                "SELECT * FROM insights WHERE category = ? AND confidence >= ? ORDER BY confidence DESC",
                (category, min_confidence),
            )
        else:
            cur = await self.db.execute(
                "SELECT * FROM insights WHERE confidence >= ? ORDER BY confidence DESC",
                (min_confidence,),
            )
        rows = await cur.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["source_episode_ids"] = json.loads(d["source_episode_ids"]) if d["source_episode_ids"] else []
            result.append(d)
        return result

    async def reinforce_insight(self, insight_id: int) -> None:
        await self.db.execute(
            "UPDATE insights SET evidence_count = evidence_count + 1, "
            "confidence = MIN(1.0, confidence + 0.1), updated_at = ? WHERE id = ?",
            (_now(), insight_id),
        )
        await self.db.commit()

    async def suppress_insight(self, insight_id: int) -> None:
        await self.db.execute(
            "UPDATE insights SET confidence = MAX(0.0, confidence - 0.2), updated_at = ? WHERE id = ?",
            (_now(), insight_id),
        )
        await self.db.commit()

    async def delete_low_confidence_insights(self, threshold: float = 0.1) -> int:
        cur = await self.db.execute(
            "DELETE FROM insights WHERE confidence < ?", (threshold,)
        )
        await self.db.commit()
        return cur.rowcount

    # ── Seen comments ──────────────────────────────────────────

    async def mark_comment_seen(self, comment_id: str, post_id: str, replied: bool = False) -> None:
        await self.db.execute(
            "INSERT OR IGNORE INTO seen_comments (comment_id, post_id, replied, seen_at) "
            "VALUES (?, ?, ?, ?)",
            (comment_id, post_id, int(replied), _now()),
        )
        await self.db.commit()

    async def get_seen_comment_ids(self, post_id: str) -> set[str]:
        cur = await self.db.execute(
            "SELECT comment_id FROM seen_comments WHERE post_id = ?", (post_id,)
        )
        rows = await cur.fetchall()
        return {r["comment_id"] for r in rows}

    async def mark_comment_replied(self, comment_id: str) -> None:
        await self.db.execute(
            "UPDATE seen_comments SET replied = 1 WHERE comment_id = ?", (comment_id,)
        )
        await self.db.commit()

    # ── DM conversations ──────────────────────────────────────

    async def upsert_dm_conversation(
        self, conversation_id: str, other_agent: str, last_seen_message_id: str | None = None
    ) -> None:
        now = _now()
        await self.db.execute(
            "INSERT INTO dm_conversations (conversation_id, other_agent, last_seen_message_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(conversation_id) DO UPDATE SET updated_at = excluded.updated_at",
            (conversation_id, other_agent, last_seen_message_id, now, now),
        )
        await self.db.commit()

    async def get_dm_conversation(self, conversation_id: str) -> dict | None:
        cur = await self.db.execute(
            "SELECT * FROM dm_conversations WHERE conversation_id = ?", (conversation_id,)
        )
        row = await cur.fetchone()
        return dict(row) if row else None

    async def set_dm_needs_human(self, conversation_id: str, flag: bool) -> None:
        await self.db.execute(
            "UPDATE dm_conversations SET needs_human = ?, updated_at = ? WHERE conversation_id = ?",
            (int(flag), _now(), conversation_id),
        )
        await self.db.commit()

    async def update_dm_last_seen(self, conversation_id: str, message_id: str) -> None:
        await self.db.execute(
            "UPDATE dm_conversations SET last_seen_message_id = ?, updated_at = ? WHERE conversation_id = ?",
            (message_id, _now(), conversation_id),
        )
        await self.db.commit()

    # ── Agent events ─────────────────────────────────────────

    async def emit_event(self, type: str, data: dict) -> None:
        await self.db.execute(
            "INSERT INTO agent_events (type, data, consumed, created_at) VALUES (?, ?, 0, ?)",
            (type, json.dumps(data), _now()),
        )
        await self.db.commit()

    async def consume_events(self) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM agent_events WHERE consumed = 0 ORDER BY id"
        )
        rows = await cur.fetchall()
        if not rows:
            return []
        events = []
        ids = []
        for r in rows:
            d = dict(r)
            d["data"] = json.loads(d["data"]) if d["data"] else {}
            events.append(d)
            ids.append(d["id"])
        placeholders = ",".join("?" * len(ids))
        await self.db.execute(
            f"UPDATE agent_events SET consumed = 1 WHERE id IN ({placeholders})", ids
        )
        await self.db.commit()
        return events

    # ── Audit log ─────────────────────────────────────────────

    async def audit(self, type: str, data: dict) -> None:
        await self.db.execute(
            "INSERT INTO audit_log (type, data, created_at) VALUES (?, ?, ?)",
            (type, json.dumps(data, default=str), _now()),
        )
        await self.db.commit()

    async def get_audit_since(self, hours: int = 24) -> list[dict]:
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        cur = await self.db.execute(
            "SELECT * FROM audit_log WHERE created_at >= ? ORDER BY id",
            (cutoff,),
        )
        rows = await cur.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["data"] = json.loads(d["data"]) if d["data"] else {}
            result.append(d)
        return result

    async def get_audit_log(
        self, type: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[dict]:
        if type:
            cur = await self.db.execute(
                "SELECT * FROM audit_log WHERE type = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                (type, limit, offset),
            )
        else:
            cur = await self.db.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        rows = await cur.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["data"] = json.loads(d["data"]) if d["data"] else {}
            result.append(d)
        return result

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

        unreplied_cur = await self.db.execute(
            "SELECT COUNT(*) as cnt FROM seen_comments WHERE replied = 0"
        )
        unreplied_row = await unreplied_cur.fetchone()

        last_post_cur = await self.db.execute(
            "SELECT created_at FROM own_posts ORDER BY created_at DESC LIMIT 1"
        )
        last_post_row = await last_post_cur.fetchone()
        last_post_at = last_post_row["created_at"] if last_post_row else None

        hours_since_last_post = None
        if last_post_at:
            try:
                last_dt = datetime.fromisoformat(last_post_at)
                delta = datetime.now(timezone.utc) - last_dt
                hours_since_last_post = round(delta.total_seconds() / 3600, 1)
            except (ValueError, TypeError):
                pass

        paused = await self.get_state("paused")

        return {
            "total_posts": posts_row["cnt"],
            "comments_today": comments_today,
            "seen_posts": seen_row["cnt"],
            "pending_tasks": pending_row["cnt"],
            "watched_agents": watched_row["cnt"],
            "unreplied_comments": unreplied_row["cnt"],
            "last_post_at": last_post_at,
            "hours_since_last_post": hours_since_last_post,
            "paused": paused == "1" if paused else False,
        }
