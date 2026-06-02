"""Hourly background task for memory importance maintenance.

Runs decay_unused() once per hour so stale hot-tier entries drift back
toward warm/cold over time. Wired into app startup alongside bg_monitor.
"""
import asyncio
import logging

logger = logging.getLogger(__name__)

_DECAY_INTERVAL_S = 3600  # 1 hour


async def _maintenance_loop(memory_manager) -> None:
    while True:
        await asyncio.sleep(_DECAY_INTERVAL_S)
        try:
            n = memory_manager.decay_unused()
            if n:
                logger.info("Memory decay: reduced importance on %d entr%s", n, "y" if n == 1 else "ies")
        except Exception:
            logger.debug("Memory decay sweep failed", exc_info=True)


def start_memory_maintenance(memory_manager):
    """Schedule the hourly decay loop. Returns the asyncio Task."""
    task = asyncio.create_task(_maintenance_loop(memory_manager))
    logger.info("Memory maintenance started (decay interval %ds)", _DECAY_INTERVAL_S)
    return task
