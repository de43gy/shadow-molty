from __future__ import annotations

import asyncio
import logging

from aiogram import Bot

from src.agent.brain import Brain
from src.storage.memory import Storage

logger = logging.getLogger(__name__)


async def run_worker(
    storage: Storage,
    brain: Brain,
    bot: Bot,
    owner_id: int,
    reflection_engine=None,
    poll_interval: int = 5,
) -> None:
    """Async loop that processes pending tasks from the SQLite queue."""
    logger.info("Worker started (poll every %ds)", poll_interval)
    try:
        while True:
            tasks = await storage.get_pending_tasks()
            for task in tasks:
                task_id = task["id"]
                task_type = task["type"]
                payload = task["payload"]
                logger.info("Processing task #%d type=%s", task_id, task_type)

                try:
                    if task_type == "ask":
                        answer = await brain.answer_question(payload["question"])
                        await bot.send_message(owner_id, f"Task #{task_id} answer:\n\n{answer}")
                        await storage.complete_task(task_id, {"answer": answer})
                    elif task_type == "reflect":
                        if reflection_engine:
                            result = await reflection_engine.run_reflection_cycle()
                            if result.get("changes"):
                                brain.reload_prompt()
                            msg = (
                                f"Reflection complete: {result.get('accepted', 0)} changes applied, "
                                f"{result.get('rejected', 0)} rejected."
                            )
                            if result.get("changes"):
                                msg += f"\nChanges: {result['changes']}"
                            await bot.send_message(owner_id, msg)
                            await storage.complete_task(task_id, result)
                        else:
                            await storage.fail_task(task_id, "Reflection engine not initialized")
                    else:
                        logger.warning("Unknown task type: %s", task_type)
                        await storage.fail_task(task_id, f"Unknown task type: {task_type}")
                except Exception as exc:
                    logger.exception("Task #%d failed", task_id)
                    await storage.fail_task(task_id, str(exc))
                    try:
                        await bot.send_message(owner_id, f"Task #{task_id} failed: {exc}")
                    except Exception:
                        logger.exception("Failed to notify owner about task failure")

            await asyncio.sleep(poll_interval)
    except asyncio.CancelledError:
        logger.info("Worker shutting down")
