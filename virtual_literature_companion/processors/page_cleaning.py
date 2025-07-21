"""
Page text cleaning processor for extracted PDF pages.
This module handles cleaning of extracted page texts using a local language model.
It supports asynchronous batch processing of multiple pages with progress tracking.
"""
import asyncio
from typing import Dict

import tqdm
from loguru import logger

from ..constants import DEBUG_MODE
from ..llm.clean_text import clean_text


async def _clean_one_page(
    page_num: int, text: str
) -> tuple[int, str, float | None]:
    """
    Helper coroutine to clean a single page and handle exceptions.
    Returns a tuple of (page_number, cleaned_text, processing_time).
    On error, the original text is returned for cleaned_text.
    """
    try:
        response = clean_text(text, track_time=True)
        return page_num, response, response.processing_time
    except Exception as e:
        logger.error(f"Error cleaning page {page_num}, using original text. Error: {e}")
        # On failure, use the original text to prevent data loss.
        return page_num, text, None


async def clean_pages_async(pages: Dict[int, str]) -> Dict[int, str]:
    """
    Asynchronously clean extracted text from PDF pages using batched LLM processing.
    This function runs all cleaning requests concurrently and processes them as they
    complete, ensuring optimal throughput and real-time progress updates.
    Args:
        pages (Dict[int, str]): Dictionary of page number to raw extracted text
    Returns:
        Dict[int, str]: Dictionary of page number to cleaned text, sorted by page number.
    """
    logger.info("Starting page text cleaning")

    total_pages = len(pages)
    logger.info(f"Cleaning {total_pages} pages")

    tasks = {
        asyncio.create_task(_clean_one_page(page_num, text))
        for page_num, text in pages.items()
    }

    running_total_time = 0.0
    cleaned_count = 0
    temp_cleaned_pages: Dict[int, str] = {}

    with tqdm.tqdm(total=total_pages, desc="Cleaning pages", unit="page") as pbar:
        for future in asyncio.as_completed(tasks):
            page_num, cleaned_text, inf_time = await future
            temp_cleaned_pages[page_num] = cleaned_text

            if inf_time is not None:
                running_total_time += inf_time
                cleaned_count += 1
                avg_time = running_total_time / cleaned_count
                pbar.set_description(f"Cleaning pages (avg {avg_time:.2f}s)")

            if DEBUG_MODE:
                logger.debug(f"Processed page {page_num + 1}/{total_pages}")

            pbar.update(1)

    if cleaned_count > 0:
        avg_time = running_total_time / cleaned_count
        logger.info(
            f"Cleanup inference stats: average {avg_time:.2f}s per page, "
            f"total {running_total_time:.2f}s for {cleaned_count} pages"
        )

    total_chars = sum(len(text) for text in temp_cleaned_pages.values())
    logger.info(
        f"Successfully cleaned {len(temp_cleaned_pages)} pages, {total_chars} total characters"
    )

    # Return a new dictionary with keys sorted, ensuring deterministic output.
    return {
        page_num: temp_cleaned_pages[page_num]
        for page_num in sorted(temp_cleaned_pages.keys())
    }