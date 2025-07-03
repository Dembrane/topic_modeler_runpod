import runpod
from utils import get_views_aspects

def handler(event):
    input = event["input"]
    segment_ids = [str(segment_id) for segment_id in input["segment_ids"]]
    user_prompt = input["user_prompt"]
    response_language = input["response_language"]
    
    response = get_views_aspects(segment_ids, user_prompt, response_language)

    return response

runpod.serverless.start({"handler": handler})
