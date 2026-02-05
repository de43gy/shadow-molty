from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware, Bot, Dispatcher, Router
from aiogram.types import Message

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

    return dp, bot
