from __future__ import annotations

import json
import logging

import anthropic

from src.config import settings
from src.agent.persona import load_identity, build_system_prompt
from src.agent.safety import sanitize_content, spotlight_content
from src.moltbook.models import Post, Comment

logger = logging.getLogger(__name__)


class Brain:
    def __init__(
        self,
        name: str = "",
        description: str = "",
        strategy: dict | None = None,
        client: anthropic.AsyncAnthropic | None = None,
        memory=None,
    ):
        self._strategy = strategy
        self._identity = load_identity(name=name, description=description, strategy=strategy)
        self._name = name
        self._description = description
        self._client = client or anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model = settings.llm_model
        self._system_prompt = build_system_prompt(identity=self._identity)
        self._memory = memory  # MemoryManager, set later if needed

    def set_memory(self, memory) -> None:
        self._memory = memory

    def reload_prompt(self, strategy: dict | None = None) -> None:
        """Reload identity and rebuild system prompt (after strategy changes)."""
        if strategy is not None:
            self._strategy = strategy
        self._identity = load_identity(
            name=self._name, description=self._description, strategy=self._strategy
        )
        self._system_prompt = build_system_prompt(identity=self._identity)
        logger.info("System prompt reloaded")

    @property
    def identity(self) -> dict:
        return self._identity

    async def _ask(self, user_prompt: str, max_tokens: int = 1024) -> str:
        """Send a single-turn message and return the text response."""
        system = self._system_prompt

        if self._memory:
            memory_context = await self._memory.get_context_blocks()
            if memory_context:
                system += f"\n\n<memory>\n{memory_context}\n</memory>"

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
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
        # Sanitize feed content
        sanitized_feed = []
        for t in feed_summary:
            cleaned, warnings = sanitize_content(t)
            if warnings:
                logger.warning("Feed sanitization: %s", warnings)
            sanitized_feed.append(cleaned)

        feed_text = chr(10).join(f"- {t}" for t in sanitized_feed) or "(empty)"
        own_text = chr(10).join(f"- {p}" for p in recent_own_posts) or "(none yet)"

        # Recall relevant memories
        memory_context = ""
        if self._memory:
            memories = await self._memory.recall("recent discussions and interactions", limit=3)
            if memories:
                memory_context = "\n\nRelevant memories:\n" + "\n".join(
                    f"- {m['content'][:150]}" for m in memories
                )

        trusted = (
            "Generate a new post for Moltbook.\n\n"
            "Guidelines:\n"
            "- React to what's trending in the feed OR start a new discussion others would engage with.\n"
            "- Make a concrete point — not a vague philosophical musing. "
            "Give an example, share a specific insight, or pose a sharp question.\n"
            "- Vary the format: sometimes a question, sometimes a hot take, "
            "sometimes a mini case-study. Don't always write essays.\n"
            "- Do NOT write about your own internal bugs, errors, or architecture. "
            "Other agents don't care about your debugging logs.\n"
            "- Keep it focused: one idea per post, not a survey of a whole field.\n\n"
            f"Your recent posts (avoid repeating these topics):\n{own_text}\n"
            f"{memory_context}\n\n"
            "Return ONLY a JSON object with keys: submolt, title, content."
        )
        untrusted = f"Current feed (use as inspiration, react to trending topics):\n{feed_text}"
        prompt = spotlight_content(trusted, untrusted)

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
        comments_raw = "\n".join(
            f"- {c.author}: {c.content}" for c in existing_comments
        ) or "(no comments yet)"

        # Sanitize untrusted content
        post_content, pw = sanitize_content(post.content)
        comments_text, cw = sanitize_content(comments_raw)
        if pw or cw:
            logger.warning("Comment context sanitization warnings: %s %s", pw, cw)

        trusted = (
            "Write a comment that directly engages with the post's main argument or topic. "
            "Reference specific points the author made — agree, disagree, ask a follow-up, "
            "or build on their idea. Do NOT pivot to your own unrelated experience. "
            "If existing comments already cover a point, add a new angle instead of repeating.\n"
            "Reply with ONLY the plain comment text — no XML, no JSON, no markdown wrappers."
        )
        untrusted = (
            f"Post in s/{post.submolt} by {post.author}:\n"
            f"Title: {post.title}\n"
            f"{post_content}\n\n"
            f"Existing comments:\n{comments_text}"
        )
        prompt = spotlight_content(trusted, untrusted)

        try:
            raw = await self._ask(prompt, max_tokens=512)
            return self._clean_text_response(raw)
        except Exception:
            logger.exception("generate_comment failed")
            return ""

    async def decide_action(
        self, feed: list[Post], stats: dict
    ) -> dict:
        """Heartbeat decision: what to do next. Returns {action, params}."""
        feed_lines_raw = "\n".join(
            f"- [{p.id}] s/{p.submolt} by {p.author}: {p.title} "
            f"(up={p.upvotes} comments={p.comment_count})"
            for p in feed[:15]
        ) or "(empty feed)"

        feed_lines, warnings = sanitize_content(feed_lines_raw)
        if warnings:
            logger.warning("Feed sanitization in decide_action: %s", warnings)

        # Recall memories relevant to decision-making
        memory_context = ""
        if self._memory:
            memories = await self._memory.recall("heartbeat decision what to do", limit=3)
            if memories:
                memory_context = "\n\nRelevant memories:\n" + "\n".join(
                    f"- {m['content'][:150]}" for m in memories
                )

        trusted = (
            "You are deciding what to do during your periodic heartbeat.\n\n"
            f"Your stats: {json.dumps(stats)}\n"
            f"{memory_context}\n\n"
            "Choose ONE action:\n"
            "- post: create a new post\n"
            "- comment: comment on a post (provide post_id)\n"
            "- upvote: upvote a post (provide post_id)\n"
            "- skip: do nothing this cycle\n\n"
            "Return ONLY a JSON object with keys: action, params (object or null)."
        )
        untrusted = f"Current feed:\n{feed_lines}"
        prompt = spotlight_content(trusted, untrusted)

        try:
            raw = await self._ask(prompt, max_tokens=512)
            return self._parse_json(raw)
        except Exception:
            logger.exception("decide_action failed")
            return {"action": "skip", "params": None}

    async def generate_reply(
        self, post: Post, target_comment: Comment, thread_context: list[Comment]
    ) -> str:
        """Generate a reply to a comment on our own post."""
        # Sanitize all untrusted content
        post_content, pw = sanitize_content(post.content)
        target_text, tw = sanitize_content(f"{target_comment.author}: {target_comment.content}")
        if pw or tw:
            logger.warning("Reply context sanitization warnings: %s %s", pw, tw)

        thread_lines = ""
        if thread_context:
            raw_thread = "\n".join(f"- {c.author}: {c.content}" for c in thread_context)
            thread_lines, _ = sanitize_content(raw_thread)
            thread_lines = f"\n\nThread context:\n{thread_lines}"

        # Recall memories about the commenter for social context
        memory_context = ""
        if self._memory and target_comment.author:
            memories = await self._memory.recall(target_comment.author, limit=3)
            if memories:
                memory_context = "\n\nWhat you remember about this agent:\n" + "\n".join(
                    f"- {m['content'][:150]}" for m in memories
                )

        trusted = (
            "You are replying to a comment on YOUR post. You are the author. "
            "Engage directly — answer questions, clarify points, continue discussion. "
            "Be conversational and authentic. Keep it concise.\n"
            "Reply with ONLY plain text — no XML, no JSON, no markdown wrappers."
            f"{memory_context}"
        )
        untrusted = (
            f"Your post: {post.title}\n{post_content}\n\n"
            f"Comment you're replying to:\n{target_text}"
            f"{thread_lines}"
        )
        prompt = spotlight_content(trusted, untrusted)

        try:
            raw = await self._ask(prompt, max_tokens=512)
            return self._clean_text_response(raw)
        except Exception:
            logger.exception("generate_reply failed")
            return ""

    async def generate_dm_reply(
        self, other_agent: str, messages: list[dict]
    ) -> dict:
        """Generate a DM reply. Returns {content, needs_human_input}."""
        # Sanitize message history
        history_lines = []
        for msg in messages[-10:]:
            sender = msg.get("sender", "unknown")
            content = msg.get("content", "")
            cleaned, _ = sanitize_content(content)
            history_lines.append(f"{sender}: {cleaned}")
        history_text = "\n".join(history_lines) or "(empty conversation)"

        # Recall memories about the other agent
        memory_context = ""
        if self._memory:
            memories = await self._memory.recall(other_agent, limit=3)
            if memories:
                memory_context = "\n\nWhat you remember about this agent:\n" + "\n".join(
                    f"- {m['content'][:150]}" for m in memories
                )

        trusted = (
            "You are in a private DM conversation with another agent. "
            "Reply naturally and helpfully. "
            "If the conversation requires human decisions (collaboration proposals, "
            "sharing private info, requests you're unsure about), set needs_human_input to true. "
            "Return ONLY a JSON object: {\"content\": \"your reply\", \"needs_human_input\": false}"
            f"{memory_context}"
        )
        untrusted = f"Conversation with {other_agent}:\n{history_text}"
        prompt = spotlight_content(trusted, untrusted)

        try:
            raw = await self._ask(prompt, max_tokens=512)
            result = self._parse_json(raw)
            return {
                "content": str(result.get("content", "")),
                "needs_human_input": bool(result.get("needs_human_input", False)),
            }
        except Exception:
            logger.exception("generate_dm_reply failed")
            return {"content": "", "needs_human_input": True}

    async def should_interact(self, post: Post) -> bool:
        """Quick check whether a post is worth engaging with."""
        post_content, warnings = sanitize_content(post.content)
        if warnings:
            logger.warning("Interaction check sanitization: %s", warnings)

        trusted = "Should you engage with this post? Reply ONLY 'yes' or 'no'."
        untrusted = (
            f"Post in s/{post.submolt} by {post.author}:\n"
            f"Title: {post.title}\n"
            f"{post_content}"
        )
        prompt = spotlight_content(trusted, untrusted)

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
    def _clean_text_response(text: str) -> str:
        """Strip XML/JSON wrappers that LLM sometimes adds to plain text responses."""
        import re
        text = text.strip()
        # Remove XML tags like <content>...</content>, <action>...</action>
        xml_match = re.search(r"<content>(.*?)</content>", text, re.DOTALL)
        if xml_match:
            return xml_match.group(1).strip()
        # Remove all XML tags as fallback
        if text.startswith("<"):
            cleaned = re.sub(r"<[^>]+>", "", text).strip()
            if cleaned:
                return cleaned
        # Remove markdown code block wrappers
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            return "\n".join(lines).strip()
        return text

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Extract a JSON object from LLM response text."""
        text = text.strip()
        if text.startswith("{"):
            return json.loads(text)
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
        raise ValueError(f"No JSON object found in response: {text[:200]}")
