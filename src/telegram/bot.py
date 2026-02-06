from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware, Bot, Dispatcher, Router
from aiogram.types import BotCommand, Message

from src.config import settings
from src.moltbook.client import MoltbookClient
from src.storage.memory import Storage
from src.telegram.handlers import register_handlers

logger = logging.getLogger(__name__)


class OwnerOnlyMiddleware(BaseMiddleware):
    """Silently drops messages from non-owner users."""

    def __init__(self, owner_id: int) -> None:
        self._owner_id = owner_id

    async def __call__(
        self,
        handler: Callable[[Message, dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: dict[str, Any],
    ) -> Any:
        if event.from_user and event.from_user.id == self._owner_id:
            return await handler(event, data)
        logger.debug("Ignored message from user %s", event.from_user and event.from_user.id)
        return None


def create_bot(storage: Storage, moltbook: MoltbookClient) -> tuple[Dispatcher, Bot]:
    bot = Bot(token=settings.telegram_bot_token)
    dp = Dispatcher()
    router = Router()

    router.message.middleware(OwnerOnlyMiddleware(settings.telegram_owner_id))
    register_handlers(router)

    dp.include_router(router)
    dp["storage"] = storage
    dp["moltbook"] = moltbook

    dp.startup.register(_set_commands(bot))

    return dp, bot


def _set_commands(bot: Bot):
    async def on_startup(*args, **kwargs):
        await bot.set_my_commands([
            BotCommand(command="status", description="Agent status & stats"),
            BotCommand(command="search", description="Search Moltbook"),
            BotCommand(command="ask", description="Ask the LLM a question"),
            BotCommand(command="post", description="Create a post"),
            BotCommand(command="watch", description="Follow an agent"),
            BotCommand(command="unwatch", description="Unfollow an agent"),
            BotCommand(command="digest", description="Activity digest"),
            BotCommand(command="dms", description="List DM conversations"),
            BotCommand(command="dm_reply", description="Reply to a DM"),
            BotCommand(command="reflect", description="Trigger reflection"),
            BotCommand(command="heartbeat", description="Manual heartbeat"),
            BotCommand(command="pause", description="Pause autonomous behavior"),
            BotCommand(command="resume", description="Resume autonomous behavior"),
        ])
    return on_startup
