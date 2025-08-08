import os
import asyncio
import tempfile
import urllib.request

import requests
from runpod import RunPodLogger
from utils.retry import retry_with_backoff, async_retry_with_backoff
from integrations.directus_client import DIRECTUS_BASE_URL, get_directus_client

logger = RunPodLogger()


def _generate_dalle_image(prompt: str) -> str:
    """
    Helper function to generate image using DALL-E 3 API.

    Args:
        prompt: The image generation prompt

    Returns:
        str: Generated image URL

    Raises:
        Exception: If the API call fails
    """
    azure_endpoint = os.getenv("AZURE_DALE3_URL").rstrip("/")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('AZURE_API_KEY')}",
    }
    payload = {
        "model": "dall-e-3",
        "prompt": prompt,
        "size": "1024x1024",
        "quality": "standard",
        "n": 1,
        "style": "vivid",
    }

    logger.debug(f"Making DALL-E API request to {azure_endpoint}")
    response = requests.post(
        azure_endpoint,
        headers=headers,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    response_data = response.json()
    image_url = response_data["data"][0]["url"]
    logger.debug(f"Successfully generated image URL: {image_url}")
    return image_url


def _download_and_upload_image(image_url: str, aspect_title: str, aspect_summary: str) -> str:
    """
    Helper function to download image and upload to Directus.

    Args:
        image_url: The generated image URL
        aspect_title: Title for the aspect
        aspect_summary: Summary for the aspect

    Returns:
        str: Directus image URL or empty string if upload fails

    Raises:
        Exception: If download or upload fails
    """
    # Download the image to a temporary location
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_path = tmp_file.name
        urllib.request.urlretrieve(image_url, tmp_path)
        logger.debug(f"Downloaded image to temporary location: {tmp_path}")

    try:
        # Upload the file to Directus
        data = {
            "title": f"Aspect Image - {aspect_title}",
            "description": f"Generated image for aspect: {aspect_summary}",
            "tags": ["aspect", "generated", "dalle"],
        }

        # Upload the file
        uploaded_file = get_directus_client().upload_file(tmp_path, data)

        # Construct the URL of the uploaded file
        if isinstance(uploaded_file, dict) and "id" in uploaded_file:
            directus_image_url = f"{DIRECTUS_BASE_URL}/assets/{uploaded_file['id']}"
            logger.debug(f"Successfully uploaded image to Directus: {directus_image_url}")
            return directus_image_url
        else:
            logger.error(f"Unexpected upload response format: {uploaded_file}")
            raise Exception(f"Unexpected upload response format: {uploaded_file}")

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.debug(f"Cleaned up temporary file: {tmp_path}")


def get_image_url(aspect_title: str, aspect_summary: str) -> str:
    """
    Generate an image URL using the DALL-E 3 model with retry logic.

    Returns:
        str: Generated image URL or empty string if generation fails
    """
    PROMPT = f"""
    In an impressionism style painting, represent the theme of the following context and summary.
    Use shades of neon turquoise, light blue and light pink. Always capture the essence of the text from a larger perspective.
    NEVER INCLUDE text in the image. I REPEAT, don't include any text in the image.

    What the image should be about: "{aspect_title}"
    Summary of ideas: "{aspect_summary}"
    """

    try:
        # Generate image with retry logic
        image_url = retry_with_backoff(
            _generate_dalle_image,
            max_retries=3,
            initial_delay=2,
            backoff_factor=2,
            jitter=0.5,
            logger=logger,
            prompt=PROMPT,
        )

        # Download and upload with retry logic
        directus_url = retry_with_backoff(
            _download_and_upload_image,
            max_retries=3,
            initial_delay=1,
            backoff_factor=2,
            jitter=0.3,
            logger=logger,
            image_url=image_url,
            aspect_title=aspect_title,
            aspect_summary=aspect_summary,
        )

        logger.info(f"Successfully processed image for aspect: {aspect_title}")
        return directus_url

    except Exception as e:
        logger.error(f"Error generating image after all retries: {e}")
        return ""


async def _generate_dalle_image_async(prompt: str) -> str:
    """
    Async helper function to generate image using DALL-E 3 API.

    Args:
        prompt: The image generation prompt

    Returns:
        str: Generated image URL

    Raises:
        Exception: If the API call fails
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _generate_dalle_image, prompt)


async def _download_and_upload_image_async(
    image_url: str, aspect_title: str, aspect_summary: str
) -> str:
    """
    Async helper function to download image and upload to Directus.

    Args:
        image_url: The generated image URL
        aspect_title: Title for the aspect
        aspect_summary: Summary for the aspect

    Returns:
        str: Directus image URL

    Raises:
        Exception: If download or upload fails
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _download_and_upload_image, image_url, aspect_title, aspect_summary
    )


async def get_image_url_async(aspect_title: str, aspect_summary: str) -> str:
    """
    Async version of get_image_url for parallel processing with timeout and retry logic.
    """
    PROMPT = f"""
    In an impressionism style painting, represent the theme of the following context and summary.
    Use shades of neon turquoise, light blue and light pink. Always capture the essence of the text from a larger perspective.
    NEVER INCLUDE text in the image. I REPEAT, don't include any text in the image.

    What the image should be about: "{aspect_title}"
    Summary of ideas: "{aspect_summary}"
    """

    try:
        # Add a timeout to prevent hanging on image generation or upload issues
        async_task = asyncio.create_task(
            _generate_and_upload_async(PROMPT, aspect_title, aspect_summary)
        )
        return await asyncio.wait_for(async_task, timeout=120.0)  # 2 minute timeout

    except asyncio.TimeoutError:
        logger.error(f"Image generation timed out for aspect: {aspect_title}")
        return ""
    except Exception as e:
        logger.error(f"Error in async image generation for aspect '{aspect_title}': {e}")
        return ""


async def _generate_and_upload_async(prompt: str, aspect_title: str, aspect_summary: str) -> str:
    """
    Helper function to generate image and upload with async retry logic.
    """
    # Generate image with async retry logic
    image_url = await async_retry_with_backoff(
        _generate_dalle_image_async,
        max_retries=3,
        initial_delay=2,
        backoff_factor=2,
        jitter=0.5,
        logger=logger,
        prompt=prompt,
    )

    # Download and upload with async retry logic
    directus_url = await async_retry_with_backoff(
        _download_and_upload_image_async,
        max_retries=3,
        initial_delay=1,
        backoff_factor=2,
        jitter=0.3,
        logger=logger,
        image_url=image_url,
        aspect_title=aspect_title,
        aspect_summary=aspect_summary,
    )

    logger.info(f"Successfully processed image async for aspect: {aspect_title}")
    return directus_url
