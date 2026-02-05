import asyncio
import logging

from src.moltbook.client import MoltbookClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    client = MoltbookClient()
    try:
        me = await client.get_me()
        logger.info("Connected as %s (karma: %d)", me.name, me.karma)

        posts = await client.get_feed(sort="hot", limit=5)
        for p in posts:
            logger.info("[%s] %s — %s", p.submolt, p.title, p.author)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
