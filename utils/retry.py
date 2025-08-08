import time
import random
import asyncio


def retry_with_backoff(
    func, max_retries=3, initial_delay=2, backoff_factor=2, jitter=0.5, logger=None, *args, **kwargs
):
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if logger:
                logger.info(f"Attempt {attempt} failed with error: {e}")
            if attempt == max_retries:
                if logger:
                    logger.error(f"All {max_retries} attempts failed. Raising exception.")
                raise
            sleep_time = delay + random.uniform(0, jitter)
            if logger:
                logger.info(f"Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
            delay *= backoff_factor


async def async_retry_with_backoff(
    async_func,
    max_retries=3,
    initial_delay=2,
    backoff_factor=2,
    jitter=0.5,
    logger=None,
    *args,
    **kwargs,
):
    """
    Async version of retry_with_backoff for async functions.
    """
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return await async_func(*args, **kwargs)
        except Exception as e:
            if logger:
                logger.info(f"Attempt {attempt} failed with error: {e}")
            if attempt == max_retries:
                if logger:
                    logger.error(f"All {max_retries} attempts failed. Raising exception.")
                raise
            sleep_time = delay + random.uniform(0, jitter)
            if logger:
                logger.info(f"Retrying in {sleep_time:.2f} seconds...")
            await asyncio.sleep(sleep_time)
            delay *= backoff_factor
