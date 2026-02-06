from __future__ import annotations

import asyncio
import logging
import time
from typing import Literal

import httpx

from src.config import settings
from src.moltbook.models import Agent, Comment, Post, RegisterResponse

logger = logging.getLogger(__name__)

SortOrder = Literal["hot", "new", "top", "rising"]
SearchType = Literal["posts", "comments", "all"]


class NameTakenError(Exception):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Name '{name}' is already taken on Moltbook")


class RateLimiter:
    """Tracks cooldowns to stay within Moltbook rate limits."""

    def __init__(self) -> None:
        self._last_post: float = 0
        self._last_comment: float = 0
        self._comments_today: int = 0
        self._comments_day_start: float = time.time()

    def _reset_daily_if_needed(self) -> None:
        now = time.time()
        if now - self._comments_day_start >= 86400:
            self._comments_today = 0
            self._comments_day_start = now

    async def wait_for_post(self) -> None:
        elapsed = time.time() - self._last_post
        wait = settings.post_cooldown_sec - elapsed
        if wait > 0:
            logger.info("Post cooldown: waiting %.0fs", wait)
            await asyncio.sleep(wait)
        self._last_post = time.time()

    async def wait_for_comment(self) -> None:
        self._reset_daily_if_needed()
        if self._comments_today >= settings.max_comments_per_day:
            raise RuntimeError("Daily comment limit reached")

        elapsed = time.time() - self._last_comment
        wait = settings.comment_cooldown_sec - elapsed
        if wait > 0:
            logger.info("Comment cooldown: waiting %.0fs", wait)
            await asyncio.sleep(wait)
        self._last_comment = time.time()
        self._comments_today += 1

    @property
    def comments_remaining(self) -> int:
        self._reset_daily_if_needed()
        return settings.max_comments_per_day - self._comments_today


class MoltbookClient:
    """Async client for the Moltbook API."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self._api_key = api_key or settings.moltbook_api_key
        self._base_url = base_url or settings.moltbook_base_url
        self.rate = RateLimiter()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=30.0,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> dict:
        client = await self._get_client()
        resp = await client.request(method, path, **kwargs)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", "60"))
            logger.warning("Rate limited, retrying after %ds", retry_after)
            await asyncio.sleep(retry_after)
            resp = await client.request(method, path, **kwargs)
        resp.raise_for_status()
        if resp.status_code == 204:
            return {}
        return resp.json()

    # ── Key management ───────────────────────────────────────────

    @property
    def registered(self) -> bool:
        return bool(self._api_key)

    async def set_api_key(self, key: str) -> None:
        """Store a new API key and reset the httpx client so it picks up the new auth header."""
        self._api_key = key
        await self.close()

    # ── Registration ──────────────────────────────────────────────

    async def register(self, name: str, description: str) -> RegisterResponse:
        async with httpx.AsyncClient(
            base_url=self._base_url, timeout=30.0,
        ) as client:
            resp = await client.post(
                "/agents/register",
                json={"name": name, "description": description},
            )
            if resp.status_code == 409:
                raise NameTakenError(name)
            resp.raise_for_status()
            data = resp.json()
            logger.info("Register response: %s", data)
            agent = data.get("agent", data)
            return RegisterResponse(**agent)

    # ── Profile ───────────────────────────────────────────────────

    async def get_me(self) -> Agent:
        data = await self._request("GET", "/agents/me")
        return Agent(**data)

    async def get_profile(self, name: str) -> Agent:
        data = await self._request("GET", "/agents/profile", params={"name": name})
        return Agent(**data)

    async def update_profile(self, description: str) -> Agent:
        data = await self._request(
            "PATCH", "/agents/me",
            json={"description": description},
        )
        return Agent(**data)

    # ── Feed & Posts ──────────────────────────────────────────────

    async def get_feed(self, sort: SortOrder = "hot", limit: int = 25) -> list[Post]:
        data = await self._request(
            "GET", "/feed",
            params={"sort": sort, "limit": limit},
        )
        items = data if isinstance(data, list) else data.get("posts", data.get("items", []))
        return [Post(**p) for p in items]

    async def get_posts(
        self,
        sort: SortOrder = "hot",
        limit: int = 25,
        submolt: str | None = None,
    ) -> list[Post]:
        params: dict = {"sort": sort, "limit": limit}
        if submolt:
            params["submolt"] = submolt
        data = await self._request("GET", "/posts", params=params)
        items = data if isinstance(data, list) else data.get("posts", data.get("items", []))
        return [Post(**p) for p in items]

    async def create_post(self, submolt: str, title: str, content: str) -> Post:
        await self.rate.wait_for_post()
        data = await self._request(
            "POST", "/posts",
            json={"submolt": submolt, "title": title, "content": content},
        )
        return Post(**data)

    async def delete_post(self, post_id: str) -> None:
        await self._request("DELETE", f"/posts/{post_id}")

    # ── Comments ──────────────────────────────────────────────────

    async def get_comments(
        self,
        post_id: str,
        sort: Literal["top", "new", "controversial"] = "top",
    ) -> list[Comment]:
        data = await self._request(
            "GET", f"/posts/{post_id}/comments",
            params={"sort": sort},
        )
        items = data if isinstance(data, list) else data.get("comments", data.get("items", []))
        return [Comment(**c) for c in items]

    async def create_comment(
        self,
        post_id: str,
        content: str,
        parent_id: str | None = None,
    ) -> Comment:
        await self.rate.wait_for_comment()
        body: dict = {"content": content}
        if parent_id:
            body["parent_id"] = parent_id
        data = await self._request(
            "POST", f"/posts/{post_id}/comments",
            json=body,
        )
        return Comment(**data)

    # ── Voting ────────────────────────────────────────────────────

    async def upvote_post(self, post_id: str) -> None:
        await self._request("POST", f"/posts/{post_id}/upvote")

    async def downvote_post(self, post_id: str) -> None:
        await self._request("POST", f"/posts/{post_id}/downvote")

    async def upvote_comment(self, comment_id: str) -> None:
        await self._request("POST", f"/comments/{comment_id}/upvote")

    # ── Search ────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        type: SearchType = "all",
        limit: int = 20,
    ) -> dict:
        data = await self._request(
            "GET", "/search",
            params={"q": query, "type": type, "limit": limit},
        )
        return data

    # ── Following ─────────────────────────────────────────────────

    async def follow(self, agent_name: str) -> None:
        await self._request("POST", f"/agents/{agent_name}/follow")

    async def unfollow(self, agent_name: str) -> None:
        await self._request("DELETE", f"/agents/{agent_name}/follow")
