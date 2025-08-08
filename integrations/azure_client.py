import os
import json
import asyncio
from typing import Dict, List

from runpod import RunPodLogger
from litellm import completion
from pydantic import BaseModel

logger = RunPodLogger()


async def run_formated_llm_call_async(
    messages: List[Dict[str, str]], response_format: type[BaseModel], model_type: str = "small"
):
    """
    Async version of run_formated_llm_call for parallel processing.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        response_format: Pydantic model class for response validation
        model_type: "small" or "large" to select the model to use

    Returns:
        dict: Parsed JSON response from the LLM

    Raises:
        ValueError: If no content is received from LLM response or invalid model type
    """
    if model_type == "small":
        model = str(os.getenv("AZURE_MODEL"))
        api_key = os.getenv("AZURE_API_KEY")
        api_base = os.getenv("AZURE_API_BASE")
        api_version = os.getenv("AZURE_API_VERSION")
    elif model_type == "large":
        model = str(os.getenv("AZURE_MODEL_LARGE"))
        api_key = os.getenv("AZURE_API_KEY")
        api_base = os.getenv("AZURE_API_BASE")
        api_version = os.getenv("AZURE_API_VERSION")
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'small' or 'large'")

    # Validate that required environment variables are set
    if not all([model, api_key, api_base, api_version]):
        missing_vars = [
            var
            for var, val in [
                ("AZURE_MODEL" if model_type == "small" else "AZURE_MODEL_LARGE", model),
                ("AZURE_API_KEY", api_key),
                ("AZURE_API_BASE", api_base),
                ("AZURE_API_VERSION", api_version),
            ]
            if not val
        ]
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: completion(
            messages=messages,
            model=model,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            response_format=response_format,
        ),
    )

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError) as e:
        print(f"Error accessing response content: {e}")
        raise e
    if content is None:
        logger.error(f"LLM response content is None. Full response: {response}")
        raise ValueError("LLM response content is None - unable to parse JSON")

    try:
        # First parse the JSON
        parsed_json = json.loads(content)

        # Then validate against the Pydantic model
        try:
            validated_data = response_format.model_validate(parsed_json)
            return validated_data.model_dump()
        except Exception as e:
            logger.error(f"Pydantic validation failed. Error: {e}")
            logger.error(f"Invalid data structure: {parsed_json}")
            raise ValueError(f"Response data does not match expected schema: {e}") from e

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON. Content: {content}")
        raise ValueError(f"Invalid JSON response from LLM: {e}") from e
