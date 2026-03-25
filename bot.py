from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher

from logic import rag_service, router
from settings import get_secret


async def on_startup() -> None:
    await rag_service.initialize()
    await rag_service.validate_llm()


async def main() -> None:
    token = get_secret("BOT_TOKEN", "")
    if not token:
        raise RuntimeError("BOT_TOKEN is not set")

    logging.basicConfig(level=logging.INFO)
    bot = Bot(token=token)
    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    dispatcher.startup.register(on_startup)
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
