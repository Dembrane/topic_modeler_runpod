from typing import Dict, List
from tqdm.asyncio import tqdm
from runpod import RunPodLogger
from data_model import Aspect
from prompts import (
    rag_user_prompt,
    rag_system_prompt,
    initial_rag_prompt,
    fallback_get_aspect_response_list_user_prompt,
    fallback_get_aspect_response_list_system_prompt,
)
from integrations.rag_client import get_rag_prompt_async
from integrations.azure_client import run_formated_llm_call_async
from services.image_generator import get_image_url_async

logger = RunPodLogger()


async def process_single_aspect(
    tentative_aspect_topic: str,
    segment_ids: List[str],
    segment_2_transcript: Dict[int, str],
    response_language: str = "en",
) -> Dict:
    """
    Process a single aspect asynchronously.
    """
    # Format the initial RAG prompt
    formated_initial_rag_prompt = initial_rag_prompt.format(
        tentative_aspect_topic=tentative_aspect_topic
    )

    # Get RAG prompt asynchronously
    rag_prompt = await get_rag_prompt_async(
        formated_initial_rag_prompt, segment_ids=[str(segment_id) for segment_id in segment_ids]
    )

    # Prepare RAG messages
    rag_messages = [
        {"role": "system", "content": rag_system_prompt},
        {
            "role": "user",
            "content": rag_user_prompt.format(
                response_language=response_language,
                input_report=rag_prompt,
                tentative_aspect_topic=tentative_aspect_topic,
            ),
        },
    ]

    # LLM call (litellm handles retries internally)
    formatted_response = await run_formated_llm_call_async(rag_messages, Aspect, model_type="large")

    # Get image URL asynchronously
    try:
        formatted_response["image_url"] = await get_image_url_async(
            formatted_response["title"], formatted_response["description"]
        )
    except Exception as e:
        logger.error(
            f"Error in async image generation for aspect '{formatted_response['title']}': {e}"
        )
        formatted_response["image_url"] = ""

    # Update segments (same logic)
    updated_segments = []
    for segment in formatted_response["segments"]:
        id = segment["segment_id"]
        if id in segment_2_transcript.keys():
            if segment_2_transcript[id] is not None or segment_2_transcript[id] != "":
                segment.pop("segment_id")
                segment["id"] = id
                segment["conversation_id"] = ""
                segment["verbatim_transcript"] = segment_2_transcript[id]
                segment["relevant_segments"] = f"0:{len(segment['verbatim_transcript']) - 1}"
                updated_segments.append(segment)
    formatted_response["segments"] = updated_segments

    return formatted_response


async def get_aspect_response_list(
    aspects: List[str],
    segment_ids: List[str],
    segment_2_transcript: Dict[int, str],
    response_language: str = "en",
):
    """
    Generate detailed responses for each aspect using RAG and LLM processing.
    Now processes aspects in parallel using async/await.

    Args:
        aspects: List of aspect topics to analyze
        segment_ids: List of segment IDs to process
        segment_2_transcript: Dictionary mapping segment IDs to their transcripts
        response_language: Language code for response generation (default: 'en')

    Returns:
        List[Dict]: List of aspect responses, each containing:
            - title: Aspect title
            - description: Detailed description
            - summary: Brief summary
            - segments: List of relevant segments with transcripts
            - image_url: URL for any associated image
    """
    # # Create tasks for all aspects to process them in parallel
    # tasks = [
    #     process_single_aspect(
    #         tentative_aspect_topic, segment_ids, segment_2_transcript, response_language
    #     )
    #     for tentative_aspect_topic in aspects
    # ]

    # # Wait for all tasks to complete and collect results
    # aspect_response_list = await asyncio.gather(*tasks)

    aspect_response_list = []

    for tentative_aspect_topic in tqdm(aspects, desc="Processing aspects"):
        try:
            aspect_response_list.append(
                await process_single_aspect(
                    tentative_aspect_topic, segment_ids, segment_2_transcript, response_language
                )
            )
        except Exception as e:
            logger.error(
                f"Error in process_single_aspect for aspect '{tentative_aspect_topic}': {e}"
            )
            continue

    return aspect_response_list


async def fallback_get_aspect_response_list(
    aspects: List[str],
    document_summaries: str,
    user_prompt: str,
    segment_2_transcript: Dict[int, str],
    response_language: str = "en",
):
    aspect_response_list = []
    for tentative_aspect_topic in aspects:
        messages = [
            {"role": "system", "content": fallback_get_aspect_response_list_system_prompt},
            {
                "role": "user",
                "content": fallback_get_aspect_response_list_user_prompt.format(
                    document_summaries=document_summaries,
                    user_prompt=user_prompt,
                    response_language=response_language,
                    aspect=tentative_aspect_topic,
                ),
            },
        ]
        try:
            formatted_response = await run_formated_llm_call_async(messages, Aspect)
        except Exception as e:
            logger.error(f"Error in LLM call for aspect '{tentative_aspect_topic}': {e}")
            # Create a minimal response to continue processing
            formatted_response = {
                "title": tentative_aspect_topic,
                "description": f"Error processing aspect: {str(e)}",
                "summary": "Unable to generate summary due to processing error",
                "segments": [],
            }

        try:
            formatted_response["image_url"] = await get_image_url_async(
                formatted_response["title"], formatted_response["description"]
            )
        except Exception as e:
            logger.error(f"Error generating image for aspect '{tentative_aspect_topic}': {e}")
            formatted_response["image_url"] = ""

        updated_segments = []
        for segment in formatted_response["segments"]:
            id = segment["segment_id"]
            if id in segment_2_transcript.keys():
                segment.pop("segment_id")
                segment["id"] = id
                segment["conversation_id"] = ""
                segment["verbatim_transcript"] = segment_2_transcript[id]
                segment["relevant_segments"] = f"0:{len(segment['verbatim_transcript']) - 1}"
                updated_segments.append(segment)

        formatted_response["segments"] = updated_segments
        aspect_response_list.append(formatted_response)
    return aspect_response_list
