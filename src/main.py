import asyncio
import logging

from src.agent import Brain
from src.agent.scheduler import create_scheduler
from src.agent.worker import run_worker
from src.config import settings
from src.moltbook.client import MoltbookClient
from src.storage import Storage
from src.telegram import create_bot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    storage = Storage()
    await storage.init()

    api_key = settings.moltbook_api_key or await storage.get_state("moltbook_api_key") or ""
    moltbook = MoltbookClient(api_key=api_key)

    agent_name = settings.agent_name or await storage.get_state("agent_name") or ""
    agent_desc = settings.agent_description or await storage.get_state("agent_description") or ""
    brain = Brain(name=agent_name, description=agent_desc)
    dp, bot = create_bot(storage, moltbook)

    scheduler = create_scheduler(storage, brain, moltbook, bot, settings.telegram_owner_id)
    if moltbook.registered:
        scheduler.start()
        logger.info("Scheduler started (API key present)")
    else:
        scheduler.start()
        logger.info("No Moltbook API key — waiting for /register. Heartbeat will skip until registered.")

    worker_task = asyncio.create_task(
        run_worker(storage, brain, bot, settings.telegram_owner_id)
    )

    try:
        await dp.start_polling(bot)
    finally:
        worker_task.cancel()
        await worker_task
        scheduler.shutdown(wait=False)
        await moltbook.close()
        await storage.close()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
