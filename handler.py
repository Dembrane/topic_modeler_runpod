import os

import runpod
from utils import get_views_aspects, get_views_aspects_fallback
from runpod import RunPodLogger

logger = RunPodLogger()


def handler(event):
    logger.info("Handler started - processing new request")

    input = event["input"]
    logger.info(f"Input received: {list(input.keys())}")

    segment_ids = [str(segment_id) for segment_id in input["segment_ids"]]
    user_prompt = input["user_prompt"]
    response_language = input["response_language"]
    project_analysis_run_id = input["project_analysis_run_id"]

    logger.info(
        f"Processing {len(segment_ids)} segments for project_analysis_run_id: {project_analysis_run_id}"
    )
    logger.info(f"Response language: {response_language}")
    logger.info(f"User prompt: {user_prompt[:100]}...")  # Log first 100 chars of prompt

    if input.get("run_fallback", False) or os.getenv("RUN_FALLBACK", "false").lower() == "true":
        RUN_FALLBACK_BY_DEFAULT = True
        logger.info("Fallback mode enabled - using get_views_aspects_fallback directly")
    else:
        RUN_FALLBACK_BY_DEFAULT = False
        logger.info("Standard mode - attempting get_views_aspects first")

    if RUN_FALLBACK_BY_DEFAULT:
        logger.info("Executing fallback path directly")
        try:
            response = get_views_aspects_fallback(
                segment_ids,
                user_prompt,
                project_analysis_run_id,
                response_language,
                threshold_context_length=100000,
            )
            logger.info("Fallback execution completed successfully")
            return response
        except Exception as e:
            logger.error(f"Error in fallback execution: {e}")
            raise e

    # Standard execution path
    logger.info("Attempting standard get_views_aspects execution")
    try:
        response = get_views_aspects(
            segment_ids, user_prompt, project_analysis_run_id, response_language
        )
        logger.info("Standard execution completed successfully")
        return response
    except Exception as e:
        logger.info(
            f"Error in default get_views_aspects: {e}; falling back to get_views_aspects_fallback"
        )
        try:
            logger.info("Attempting fallback execution after standard method failed")
            response = get_views_aspects_fallback(
                segment_ids,
                user_prompt,
                project_analysis_run_id,
                response_language,
                threshold_context_length=100000,
            )
            logger.info("Fallback execution completed successfully after standard method failure")
            return response
        except Exception as e:
            logger.error(f"Error in get_views_aspects_fallback: {e}")
            logger.error("Both standard and fallback methods failed - raising exception")
            raise e


runpod.serverless.start({"handler": handler})
