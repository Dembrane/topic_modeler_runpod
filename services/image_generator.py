import os
import asyncio
import tempfile
import urllib.request
import requests
from runpod import RunPodLogger
from integrations.directus_client import get_directus_client, DIRECTUS_BASE_URL

logger = RunPodLogger()


def get_image_url(aspect_title: str, aspect_summary: str) -> str:
    """
    Generate an image URL using the MODEST model.

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
        # Make direct API call to Azure DALL-E 3
        azure_endpoint = os.getenv("AZURE_DALE3_URL").rstrip("/")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('AZURE_API_KEY')}",
        }
        payload = {
            "model": "dall-e-3",
            "prompt": PROMPT,
            "size": "1024x1024",
            "quality": "standard",
            "n": 1,
            "style": "vivid",  # Adding style parameter for better results
        }

        response = requests.post(
            azure_endpoint,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        response_data = response.json()
        image_url = response_data["data"][0]["url"]
        logger.info(f"Successfully generated image URL: {image_url}")

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
                logger.info(f"Successfully uploaded image to Directus: {directus_image_url}")
                return directus_image_url
            else:
                logger.error(f"Unexpected upload response format: {uploaded_file}")
                logger.info("Could not upload image to Directus")
                return ""

        except Exception as directus_error:
            logger.error(f"Directus upload failed: {directus_error}")
            logger.info("Could not upload image to Directus")
            return ""
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                logger.debug(f"Cleaned up temporary file: {tmp_path}")

    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return ""


async def get_image_url_async(aspect_title: str, aspect_summary: str) -> str:
    """
    Async version of get_image_url for parallel processing with timeout.
    """
    try:
        loop = asyncio.get_event_loop()
        # Add a timeout to prevent hanging on Directus upload issues
        return await asyncio.wait_for(
            loop.run_in_executor(None, get_image_url, aspect_title, aspect_summary),
            timeout=60.0,  # 60 second timeout
        )
    except asyncio.TimeoutError:
        logger.error(f"Image generation timed out for aspect: {aspect_title}")
        return ""
    except Exception as e:
        logger.error(f"Error in async image generation for aspect '{aspect_title}': {e}")
        return ""
