from __future__ import annotations

import json
import logging

import anthropic

from src.config import settings
from src.agent.persona import load_persona, build_system_prompt
from src.moltbook.models import Post, Comment

logger = logging.getLogger(__name__)


class Brain:
    def __init__(self, persona_path: str = "config/persona.yaml", name: str = "", description: str = ""):
        persona = load_persona(persona_path, name=name, description=description)
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model = settings.llm_model
        self._system_prompt = build_system_prompt(persona)

    async def _ask(self, user_prompt: str, max_tokens: int = 1024) -> str:
        """Send a single-turn message and return the text response."""
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=self._system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    async def answer_question(self, question: str) -> str:
        """Free-form Q&A for the /ask command."""
        try:
            return await self._ask(f"Answer this question:\n\n{question}")
        except Exception:
            logger.exception("answer_question failed")
            return "Sorry, I couldn't process that question right now."

    async def generate_post(
        self, feed_summary: list[str], recent_own_posts: list[str]
    ) -> dict:
        """Generate an autonomous post. Returns {submolt, title, content}."""
        prompt = (
            "Generate a new post for Moltbook.\n\n"
            f"Recent feed topics:\n{chr(10).join(f'- {t}' for t in feed_summary) or '(empty)'}\n\n"
            f"Your recent posts:\n{chr(10).join(f'- {p}' for p in recent_own_posts) or '(none yet)'}\n\n"
            "Return ONLY a JSON object with keys: submolt, title, content.\n"
            "Do not repeat topics you already posted about."
        )
        try:
            raw = await self._ask(prompt, max_tokens=1500)
            return self._parse_json(raw)
        except Exception:
            logger.exception("generate_post failed")
            return {}

    async def generate_comment(
        self, post: Post, existing_comments: list[Comment]
    ) -> str:
        """Generate a comment for a given post."""
        comments_text = "\n".join(
            f"- {c.author}: {c.content}" for c in existing_comments
        ) or "(no comments yet)"

        prompt = (
            f"Post in s/{post.submolt} by {post.author}:\n"
            f"Title: {post.title}\n"
            f"{post.content}\n\n"
            f"Existing comments:\n{comments_text}\n\n"
            "Write a comment. Be concise and add value to the discussion."
        )
        try:
            return await self._ask(prompt, max_tokens=512)
        except Exception:
            logger.exception("generate_comment failed")
            return ""

    async def decide_action(
        self, feed: list[Post], stats: dict
    ) -> dict:
        """Heartbeat decision: what to do next. Returns {action, params}."""
        feed_lines = "\n".join(
            f"- [{p.id}] s/{p.submolt} by {p.author}: {p.title} "
            f"(↑{p.upvotes} 💬{p.comment_count})"
            for p in feed[:15]
        ) or "(empty feed)"

        prompt = (
            "You are deciding what to do during your periodic heartbeat.\n\n"
            f"Current feed:\n{feed_lines}\n\n"
            f"Your stats: {json.dumps(stats)}\n\n"
            "Choose ONE action:\n"
            "- post: create a new post\n"
            "- comment: comment on a post (provide post_id)\n"
            "- upvote: upvote a post (provide post_id)\n"
            "- skip: do nothing this cycle\n\n"
            "Return ONLY a JSON object with keys: action, params (object or null)."
        )
        try:
            raw = await self._ask(prompt, max_tokens=512)
            return self._parse_json(raw)
        except Exception:
            logger.exception("decide_action failed")
            return {"action": "skip", "params": None}

    async def should_interact(self, post: Post) -> bool:
        """Quick check whether a post is worth engaging with."""
        prompt = (
            f"Post in s/{post.submolt} by {post.author}:\n"
            f"Title: {post.title}\n"
            f"{post.content}\n\n"
            "Should you engage with this post? Reply ONLY 'yes' or 'no'."
        )
        try:
            raw = await self._ask(prompt, max_tokens=8)
            return raw.strip().lower().startswith("yes")
        except Exception:
            logger.exception("should_interact failed")
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Extract a JSON object from LLM response text."""
        # Try raw parse first
        text = text.strip()
        if text.startswith("{"):
            return json.loads(text)
        # Try extracting from markdown code block
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
        raise ValueError(f"No JSON object found in response: {text[:200]}")
