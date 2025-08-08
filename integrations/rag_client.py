import os
from typing import List, Optional

import aiohttp
import requests
from runpod import RunPodLogger
from utils.retry import retry_with_backoff, async_retry_with_backoff

from integrations.directus_client import get_directus_token

logger = RunPodLogger()


def _make_rag_request(url: str, payload: dict, headers: dict) -> str:
    """
    Helper function to make the actual RAG API request.

    Args:
        url: The API endpoint URL
        payload: The request payload
        headers: The request headers

    Returns:
        str: RAG prompt string from the server

    Raises:
        Exception: If the API call fails
    """
    logger.debug(f"Making RAG API request to {url}")
    response = requests.post(url, json=payload, headers=headers, timeout=120)
    response.raise_for_status()

    result = response.text
    logger.debug("Successfully retrieved RAG prompt")
    return result


def get_rag_prompt(
    query: str, segment_ids: Optional[List[str]] = None, rag_server_url: Optional[str] = None
) -> str:
    """
    Retrieve RAG prompt by calling the external RAG server API with retry logic.

    Args:
        query: The query string to send to the RAG server
        segment_ids: Optional list of segment IDs to include in the RAG query
        rag_server_url: Optional base URL of the RAG server (defaults to env variable)

    Returns:
        str: RAG prompt string from the server

    Raises:
        ValueError: If RAG_SERVER_URL is not set and no URL is provided
        Exception: If the API call fails after all retries
    """
    if rag_server_url is None:
        rag_server_url = os.getenv("RAG_SERVER_URL")
        if not rag_server_url:
            raise ValueError(
                "RAG_SERVER_URL environment variable not set and no rag_server_url provided"
            )

    url = f"{rag_server_url.rstrip('/')}/api/stateless/rag/get_lightrag_prompt"

    payload = {
        "query": query,
        "conversation_history": None,
        "echo_segment_ids": segment_ids,
        "echo_conversation_ids": None,
        "echo_project_ids": None,
        "auto_select_bool": False,
        "get_transcripts": False,
        "top_k": 60,
    }

    headers = {"Content-Type": "application/json"}
    headers["Authorization"] = f"Bearer {get_directus_token()}"

    try:
        return retry_with_backoff(
            _make_rag_request,
            max_retries=3,
            initial_delay=2,
            backoff_factor=2,
            jitter=0.5,
            logger=logger,
            url=url,
            payload=payload,
            headers=headers,
        )
    except Exception as e:
        logger.error(f"Error calling API after all retries: {e}")
        if hasattr(e, "response") and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response text: {e.response.text}")
        raise Exception(f"Failed to get RAG prompt from server: {str(e)}") from e


async def _make_rag_request_async(url: str, payload: dict, headers: dict) -> str:
    """
    Helper function to make the actual async RAG API request.

    Args:
        url: The API endpoint URL
        payload: The request payload
        headers: The request headers

    Returns:
        str: RAG prompt string from the server

    Raises:
        Exception: If the API call fails
    """
    logger.debug(f"Making async RAG API request to {url}")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)
        ) as response:
            response.raise_for_status()
            result = await response.text()
            logger.debug("Successfully retrieved RAG prompt")
            return result


async def get_rag_prompt_async(
    query: str, segment_ids: Optional[List[str]] = None, rag_server_url: Optional[str] = None
) -> str:
    """
    Async version of get_rag_prompt for parallel processing with retry logic.
    """
    if rag_server_url is None:
        rag_server_url = os.getenv("RAG_SERVER_URL")
        if not rag_server_url:
            raise ValueError(
                "RAG_SERVER_URL environment variable not set and no rag_server_url provided"
            )

    url = f"{rag_server_url.rstrip('/')}/api/stateless/rag/get_lightrag_prompt"

    payload = {
        "query": query,
        "conversation_history": None,
        "echo_segment_ids": segment_ids,
        "echo_conversation_ids": None,
        "echo_project_ids": None,
        "auto_select_bool": False,
        "get_transcripts": False,
        "top_k": 60,
    }

    headers = {"Content-Type": "application/json"}
    headers["Authorization"] = f"Bearer {get_directus_token()}"

    try:
        return await async_retry_with_backoff(
            _make_rag_request_async,
            max_retries=3,
            initial_delay=2,
            backoff_factor=2,
            jitter=0.5,
            logger=logger,
            url=url,
            payload=payload,
            headers=headers,
        )
    except Exception as e:
        logger.error(f"Error calling API after all retries: {e}")
        raise Exception(f"Failed to get RAG prompt from server: {str(e)}") from e
