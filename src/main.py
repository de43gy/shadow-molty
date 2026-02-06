import asyncio
import logging

import anthropic

from src.agent import Brain
from src.agent.consolidation import ConsolidationEngine
from src.agent.memory import MemoryManager
from src.agent.persona import load_constitution
from src.agent.reflection import ReflectionEngine
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

    # Shared Anthropic client
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    api_key = settings.moltbook_api_key or await storage.get_state("moltbook_api_key") or ""
    moltbook = MoltbookClient(api_key=api_key)

    agent_name = settings.agent_name or await storage.get_state("agent_name") or ""
    agent_desc = settings.agent_description or await storage.get_state("agent_description") or ""

    brain = Brain(name=agent_name, description=agent_desc, client=client)

    # Memory system
    memory = MemoryManager(storage, client, settings.llm_model)
    brain.set_memory(memory)
    await memory.init_core_blocks(brain.identity)

    # Constitution for safety checks
    constitution = load_constitution()

    # Reflection engine
    reflection = ReflectionEngine(storage, memory, client, settings.llm_model, constitution)

    # Consolidation engine
    consolidation = ConsolidationEngine(storage, memory, client, settings.llm_model)

    # Sync own posts from Moltbook profile on every startup
    # Use DB-stored name (actual registered name) — env AGENT_NAME may differ
    registered_name = await storage.get_state("agent_name") or agent_name
    if moltbook.registered and registered_name:
        try:
            posts = await moltbook.get_profile_posts(registered_name)
            for p in posts:
                await storage.save_own_post(p)
            logger.info("Synced %d posts from profile (%s)", len(posts), registered_name)
        except Exception:
            logger.warning("Failed to sync own posts from profile", exc_info=True)

    dp, bot = create_bot(storage, moltbook)

    scheduler = create_scheduler(
        storage, brain, moltbook, bot, settings.telegram_owner_id,
        memory=memory,
        reflection=reflection,
        consolidation_engine=consolidation,
    )
    if moltbook.registered:
        scheduler.start()
        logger.info("Scheduler started (API key present)")
    else:
        scheduler.start()
        logger.info("No Moltbook API key — waiting for /register. Heartbeat will skip until registered.")

    worker_task = asyncio.create_task(
        run_worker(
            storage, brain, bot, settings.telegram_owner_id,
            moltbook=moltbook, memory=memory, reflection_engine=reflection,
        )
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
