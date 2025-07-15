import os

import runpod
from utils import get_views_aspects, get_views_aspects_fallback


def handler(event):
    input = event["input"]
    segment_ids = [str(segment_id) for segment_id in input["segment_ids"]]
    user_prompt = input["user_prompt"]
    response_language = input["response_language"]
    project_analysis_run_id = input["project_analysis_run_id"]
    if input.get("run_fallback", False) or os.getenv("RUN_FALLBACK", "false").lower() == "true":
        RUN_FALLBACK_BY_DEFAULT = True
    else:
        RUN_FALLBACK_BY_DEFAULT = False

    if RUN_FALLBACK_BY_DEFAULT:
        response = get_views_aspects_fallback(
            segment_ids, user_prompt, project_analysis_run_id, response_language
        )
        return response
    try:
        response = get_views_aspects(
            segment_ids, user_prompt, project_analysis_run_id, response_language
        )
    except Exception as e:
        print(
            f"Error in default get_views_aspects: {e}; falling back to get_views_aspects_fallback"
        )
        try:
            response = get_views_aspects_fallback(segment_ids, user_prompt, response_language)
            return response
        except Exception as e:
            print(f"Error in get_views_aspects_fallback: {e}")
            raise e


runpod.serverless.start({"handler": handler})
