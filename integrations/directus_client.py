import os
from datetime import datetime, timezone
from typing import Dict, List
from dotenv import load_dotenv
from directus_sdk_py import DirectusClient
from runpod import RunPodLogger
from utils.helpers import generate_uuid

load_dotenv()
logger = RunPodLogger()

DIRECTUS_BASE_URL = str(os.getenv("DIRECTUS_BASE_URL"))
DIRECTUS_USERNAME = str(os.getenv("DIRECTUS_USERNAME"))
DIRECTUS_PASSWORD = str(os.getenv("DIRECTUS_PASSWORD"))


def get_directus_client() -> DirectusClient:
    directus_client = DirectusClient(
        url=DIRECTUS_BASE_URL, email=DIRECTUS_USERNAME, password=DIRECTUS_PASSWORD
    )
    directus_client.login()
    return directus_client


def get_directus_token():
    DIRECTUS_USERNAME = str(os.getenv("DIRECTUS_USERNAME"))
    DIRECTUS_PASSWORD = str(os.getenv("DIRECTUS_PASSWORD"))
    client = DirectusClient(
        url=DIRECTUS_BASE_URL, email=DIRECTUS_USERNAME, password=DIRECTUS_PASSWORD
    )
    client.login()
    token = client.get_token()
    return token


def update_directus(response, project_analysis_run_id) -> None:
    view = response["view"]
    title = view.get("title", "")
    description = view.get("description", "")
    summary = view.get("summary", "")
    seed = view.get("seed", "")
    language = view.get("language", "en")
    aspects = view.get("aspects", [])
    user_input = view.get("user_input", "")
    user_input_description = view.get("user_input_description", "")
    view_id = generate_uuid()
    get_directus_client().create_item(
        "view",
        item_data={
            "id": str(view_id),
            "name": title,
            "description": description,
            "summary": summary,
            "language": language,
            "processing_status": "Generating Aspects",
            "processing_started_at": str(datetime.now(timezone.utc)),
            "project_analysis_run_id": str(project_analysis_run_id),
            "user_input": user_input,
            "user_input_description": user_input_description,
        },
    )
    for rank, aspect in enumerate(aspects):
        aspect_id = generate_uuid()
        aspect_title = aspect.get("title", "")
        aspect_description = aspect.get("description", "")
        aspect_summary = aspect.get("summary", "")
        segments = aspect.get("segments", [])
        image_url = aspect.get("image_url", "")
        get_directus_client().create_item(
            "aspect",
            {
                "id": str(aspect_id),
                "name": aspect_title,
                "description": aspect_description,
                "short_summary": aspect_description,
                "long_summary": aspect_summary,
                "image_url": image_url,
                "view_id": str(view_id),
                "rank": rank,
            },
        )
        for segment in segments:
            aspect_segment_id = generate_uuid()
            segment_description = segment.get("description", "")
            segment_id = segment.get("id", "")
            conversation_id = segment.get("conversation_id", "")
            verbatim_transcript = segment.get("verbatim_transcript", "")
            relevant_index = segment.get("relevant_segments", "")
            get_directus_client().create_item(
                "aspect_segment",
                {
                    "id": str(aspect_segment_id),
                    "description": segment_description,
                    "aspect": str(aspect_id),
                    "segment": str(segment_id),
                    "conversation_id": str(conversation_id),
                    "verbatim_transcript": verbatim_transcript,
                    "relevant_index": relevant_index,
                },
            )
    get_directus_client().create_item(
        "processing_status",
        {
            "project_analysis_run_id": str(project_analysis_run_id),
            "event": "runpod:topic_modeler.completed",
            "message": "view_id: " + str(view_id),
        },
    )
    return
