# [X] TODO: Add retry logic in rag calls
# [X] TODO: Check why user_input and user_input_description are not being populated in directus
# [X] TODO: Add image URL to the response
# [ ] TODO: Change backend of echo to respond only with data not prompt

import os
import json
import time
import uuid
import random
import asyncio
import tempfile
import urllib.request
from typing import Dict, List, Optional
from datetime import datetime, timezone

import torch
import pandas as pd
import aiohttp
import requests
from umap import UMAP
from dotenv import load_dotenv
from openai import OpenAI
from runpod import RunPodLogger
from hdbscan import HDBSCAN
from litellm import completion
from prompts import (
    rag_user_prompt,
    rag_system_prompt,
    initial_rag_prompt,
    topic_model_user_prompt,
    view_summary_user_prompt,
    topic_model_system_prompt,
    view_summary_system_prompt,
    vanilla_topic_model_user_prompt,
    vanilla_topic_model_system_prompt,
    fallback_get_aspect_response_list_user_prompt,
    fallback_get_aspect_response_list_system_prompt,
)
from bertopic import BERTopic
from pydantic import BaseModel
from data_model import Aspect, TopicModelResponse, ViewSummaryResponse
from litellm.utils import token_counter
from directus_sdk_py import DirectusClient
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

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


client = OpenAI(base_url=os.getenv("OPENAI_API_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY"))


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
            logger.info(f"Downloaded image to temporary location: {tmp_path}")

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
                logger.info(f"Cleaned up temporary file: {tmp_path}")

    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return ""


def generate_uuid() -> str:
    return str(uuid.uuid4())


def get_directus_token():
    DIRECTUS_USERNAME = str(os.getenv("DIRECTUS_USERNAME"))
    DIRECTUS_PASSWORD = str(os.getenv("DIRECTUS_PASSWORD"))
    client = DirectusClient(
        url=DIRECTUS_BASE_URL, email=DIRECTUS_USERNAME, password=DIRECTUS_PASSWORD
    )
    client.login()
    token = client.get_token()
    return token


def initialize_topic_model():
    """
    Initialize BERTopic model with GPU acceleration if available.

    The function configures a BERTopic model with the following components:
    - Sentence transformer for embeddings (GPU-accelerated if available)
    - UMAP for dimensionality reduction
    - HDBSCAN for clustering
    - CountVectorizer for text preprocessing
    - ClassTfidfTransformer for topic representation

    Returns:
        BERTopic: Initialized topic model ready for document processing

    Raises:
        Exception: If model initialization fails
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if os.environ.get("RUN_CPU") == "True":
            device = "cpu"
        logger.info(f"Using device: {device}")

        if device == "cuda":
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        else:
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        umap_model = UMAP(n_neighbors=15, n_components=10, metric="cosine", random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=5, metric="euclidean", prediction_data=True)
        vectorizer_model = CountVectorizer(
            ngram_range=(1, 2), stop_words="english", min_df=2, max_features=5000
        )
        ctfidf_model = ClassTfidfTransformer()

        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            calculate_probabilities=True,
            verbose=True,
            top_n_words=15,
            min_topic_size=5,
        )

        logger.info("BERTopic model initialized successfully")
        return topic_model

    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise e


def run_topic_model_hierarchical(
    topic_model, docs, topics: Optional[List[str]] = None, nr_topics: Optional[int] = None
):
    """
    Run hierarchical topic modeling on the provided documents.

    Args:
        topic_model: Initialized BERTopic model
        docs: List of documents to process
        topics: Optional list of predefined topics
        nr_topics: Optional number of topics to reduce to after fitting

    Returns:
        tuple: Contains:
            - topics: List of identified topics
            - probs: Topic probabilities for each document
            - hierarchical_topics: Hierarchical structure of topics
    """
    # First fit the model normally
    topics, probs = topic_model.fit_transform(docs)

    # If nr_topics is specified, reduce the topics
    if nr_topics is not None:
        topic_model.reduce_topics(docs, nr_topics=nr_topics)
        topics = topic_model.topics_
        probs = topic_model.probabilities_

    hierarchical_topics = topic_model.hierarchical_topics(docs)
    return topics, probs, hierarchical_topics


def run_formated_llm_call(messages: List[Dict[str, str]], response_format: type[BaseModel]):
    """
    Execute an LLM call with formatted response validation.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        response_format: Pydantic model class for response validation

    Returns:
        dict: Parsed JSON response from the LLM

    Raises:
        ValueError: If no content is received from LLM response
    """
    response = completion(
        messages=messages,
        model=str(os.getenv("AZURE_MODEL")),
        api_key=os.getenv("AZURE_API_KEY"),
        api_base=os.getenv("AZURE_API_BASE"),
        api_version=os.getenv("AZURE_API_VERSION"),
        response_format=response_format,
    )

    return json.loads(response.choices[0].message.content)


def retry_with_backoff(
    func, max_retries=3, initial_delay=2, backoff_factor=2, jitter=0.5, logger=None, *args, **kwargs
):
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if logger:
                logger.info(f"Attempt {attempt} failed with error: {e}")
            if attempt == max_retries:
                if logger:
                    logger.error(f"All {max_retries} attempts failed. Raising exception.")
                raise
            sleep_time = delay + random.uniform(0, jitter)
            if logger:
                logger.info(f"Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
            delay *= backoff_factor


def get_rag_prompt(
    query: str, segment_ids: Optional[List[str]] = None, rag_server_url: Optional[str] = None
) -> str:
    """
    Retrieve RAG prompt by calling the external RAG server API.

    Args:
        query: The query string to send to the RAG server
        segment_ids: Optional list of segment IDs to include in the RAG query
        rag_server_url: Optional base URL of the RAG server (defaults to env variable)

    Returns:
        str: RAG prompt string from the server

    Raises:
        ValueError: If RAG_SERVER_URL is not set and no URL is provided
        Exception: If the API call fails
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
        logger.info(f"Making RAG API request to {url}")
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()

        result = response.text
        logger.info("Successfully retrieved RAG prompt")
        return result

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling API: {e}")
        if hasattr(e, "response") and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response text: {e.response.text}")
        raise Exception(f"Failed to get RAG prompt from server: {str(e)}") from e


async def get_rag_prompt_async(
    query: str, segment_ids: Optional[List[str]] = None, rag_server_url: Optional[str] = None
) -> str:
    """
    Async version of get_rag_prompt for parallel processing.
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
        logger.info(f"Making async RAG API request to {url}")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                response.raise_for_status()
                result = await response.text()
                logger.info("Successfully retrieved RAG prompt")
                return result

    except Exception as e:
        logger.error(f"Error calling API: {e}")
        raise Exception(f"Failed to get RAG prompt from server: {str(e)}") from e


async def run_formated_llm_call_async(
    messages: List[Dict[str, str]], response_format: type[BaseModel], model_type: str = "small"
):
    """
    Async version of run_formated_llm_call for parallel processing.
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
    return json.loads(response.choices[0].message.content)


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
    formatted_response["image_url"] = await get_image_url_async(
        formatted_response["title"], formatted_response["description"]
    )

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

    for tentative_aspect_topic in aspects:
        aspect_response_list.append(
            await process_single_aspect(
                tentative_aspect_topic, segment_ids, segment_2_transcript, response_language
            )
        )

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
        formatted_response = run_formated_llm_call(messages, Aspect)
        formatted_response["image_url"] = await get_image_url_async(
            formatted_response["title"], formatted_response["description"]
        )
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
    for aspect in aspects:
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


def summarise_aspects(
    aspect_response_list: List[Dict],
    response_language: str = "en",
    user_prompt: str = "",
):
    """
    Generate a summary of multiple aspects.

    Args:
        aspect_response_list: List of aspect responses to summarize
        response_language: Language code for summary generation (default: 'en')

    Returns:
        Dict: Summary response containing overview of all aspects

    Raises:
        ValueError: If aspect_response_list is empty
    """
    if len(aspect_response_list) == 0:
        raise ValueError("No aspects to summarise")
    aspect_texts = [
        f"{aspect['title']}\n{aspect['description']}\n{aspect['summary']}"
        for aspect in aspect_response_list
    ]
    view_text = "\n\n".join(aspect_texts)
    messages = [
        {"role": "system", "content": view_summary_system_prompt},
        {
            "role": "user",
            "content": view_summary_user_prompt.format(
                view_text=view_text,
                response_language=response_language,
                user_prompt=user_prompt,
            ),
        },
    ]
    view_response = run_formated_llm_call(messages, ViewSummaryResponse)
    return view_response


async def get_views_aspects(
    segment_ids: List[str],
    user_prompt: str,
    project_analysis_run_id: str,
    response_language: str | None = None,
    threshold_context_length: int = int(os.getenv("THRESHOLD_CONTEXT_LENGTH", 100000)),
    user_input: str = "",
    user_input_description: str = "",
) -> Dict:
    """
    Generate comprehensive views and aspects analysis for conversation segments.

    This function performs the following steps:
    1. Retrieves contextual transcripts for given segment IDs
    2. Performs topic modeling using either direct LLM or BERTopic based on context length
    3. Generates detailed aspect responses for each identified topic
    4. Creates a summary of all aspects

    Args:
        segment_ids: List of segment IDs to analyze
        user_prompt: User's query or instruction for analysis
        response_language: Language code for response generation (default: 'en')
        context_length: Maximum token length for direct LLM processing (default: 100000)

    Returns:
        Dict: Contains:
            - views: Summary of all aspects
            - aspects: List of detailed aspect responses
            - seed: Original user prompt
            - language: Response language used
    """
    if response_language is None:
        response_language = "en"

    segments = get_directus_client().get_items(
        "conversation_segment",
        {
            "query": {
                "filter": {"id": {"_in": segment_ids}, "transcript": {"_nnull": True}},
                "fields": ["id", "contextual_transcript", "transcript"],
                "limit": -1,
            },
        },
    )
    logger.debug(f"Retrieved segments: {segments}")
    # Type-safe dictionary comprehension with explicit type checking
    segment_2_transcript: Dict[int, str] = {}
    raw_docs: List[str] = []
    doc_ids: List[str] = []
    for segment in segments:
        if isinstance(segment, dict):
            if "id" in segment and "transcript" in segment:
                segment_2_transcript[int(segment["id"])] = str(segment["transcript"])
                doc_ids.append(str(segment["id"]))
            else:
                raise ValueError(f"Segment {segment} does not have an id or transcript")

            if "contextual_transcript" in segment:
                raw_docs.append(str(segment["contextual_transcript"]))
            else:
                raise ValueError(f"Segment {segment} does not have a contextual transcript")

    # Process docs with explicit type handling
    split_docs: List[List[str]] = [doc.split("\n") for doc in raw_docs if doc != ""]
    docs: List[str] = []

    for sublist in split_docs:
        docs.extend(sublist)

    topic_model = initialize_topic_model()
    token_length = 0
    for doc in docs:
        token_length += token_counter(model=str(os.getenv("AZURE_MODEL")), text=doc)

    if token_length < threshold_context_length:
        docs_with_ids = "---------\n\n".join(
            [f"SEGMENT_ID_{doc_id}: {doc}" for doc_id, doc in zip(doc_ids, raw_docs)]
        )
        messages = [
            {"role": "system", "content": vanilla_topic_model_system_prompt},
            {
                "role": "user",
                "content": vanilla_topic_model_user_prompt.format(
                    docs_with_ids=docs_with_ids,
                    user_prompt=user_prompt,
                    response_language=response_language,
                ),
            },
        ]
        tentative_aspects_response = run_formated_llm_call(messages, TopicModelResponse)
    else:
        topics, probs, hierarchical_topics = run_topic_model_hierarchical(topic_model, docs)
        repr_docs_token_length = threshold_context_length * 1.1
        nr_repr_docs = 100
        while repr_docs_token_length > threshold_context_length * 0.8 and nr_repr_docs > 3:
            doc_topic = pd.DataFrame(
                {
                    "Topic": topic_model.topics_,
                    "ID": range(len(topic_model.topics_)),
                    "Document": docs,
                }
            )
            repr_docs, _, _, _ = topic_model._extract_representative_docs(
                topic_model.c_tf_idf_,
                doc_topic,
                topic_model.topic_representations_,
                nr_samples=1000,
                nr_repr_docs=nr_repr_docs,
            )
            rep_doc_list = {k: v for k, v in repr_docs.items() if k != -1}
            nl = "\n\n"
            rep_docs_formatted = [
                f"Document set {k} : \n {nl.join(v)}" for k, v in rep_doc_list.items()
            ]
            representative_documents = "\n\n\n\n\n\n".join(rep_docs_formatted)
            repr_docs_token_length = token_counter(
                model=str(os.getenv("AZURE_MODEL")),
                text=representative_documents,
            )
            nr_repr_docs = round(nr_repr_docs * 0.75)
        messages = [
            {"role": "system", "content": topic_model_system_prompt},
            {
                "role": "user",
                "content": topic_model_user_prompt.format(
                    representative_documents=representative_documents,
                    user_prompt=user_prompt,
                    response_language=response_language,
                ),
            },
        ]
        tentative_aspects_response = run_formated_llm_call(messages, TopicModelResponse)

    tentative_aspects = tentative_aspects_response["topics"]
    aspect_response_list = await get_aspect_response_list(
        tentative_aspects,
        segment_ids,
        segment_2_transcript,
        response_language=response_language,
    )
    views_dict = summarise_aspects(
        aspect_response_list,
        response_language=response_language,
        user_prompt=user_prompt,
    )
    views_dict["aspects"] = aspect_response_list
    views_dict["seed"] = user_prompt
    views_dict["language"] = response_language
    views_dict["user_input"] = user_input
    views_dict["user_input_description"] = user_input_description
    response = {"view": views_dict}
    update_directus(response, project_analysis_run_id)
    return response


async def get_views_aspects_fallback(
    segment_ids: List[str],
    user_prompt: str,
    project_analysis_run_id: str,
    response_language: str | None = None,
    threshold_context_length: int = int(os.getenv("THRESHOLD_CONTEXT_LENGTH", 100000)),
    user_input: str = "",
    user_input_description: str = "",
) -> Dict:
    import random

    summaries = get_directus_client().get_items(
        "conversation_segment",
        {
            "query": {
                "filter": {"id": {"_in": segment_ids}},
                "fields": ["id", "transcript", "conversation_id.summary"],
            },
        },
    )
    logger.debug(f"Retrieved summaries: {summaries}")
    segment_2_transcript: Dict[int, str] = {}
    for summary in summaries:
        segment_2_transcript[int(summary["id"])] = str(summary["transcript"])
    summaries_list = list(
        set([(summary["id"], summary["conversation_id"]["summary"]) for summary in summaries])
    )
    random.shuffle(summaries_list)
    samples_to_summarise = []
    token_count = 0
    for summary in summaries_list:
        if (
            token_count + token_counter(model=str(os.getenv("AZURE_MODEL")), text=summary[1])
            > threshold_context_length * 0.8
        ):
            break
        samples_to_summarise.append(summary)
        token_count += token_counter(model=str(os.getenv("AZURE_MODEL")), text=summary[1])

    # Do the vanilla path
    docs_with_ids = "---------\n\n".join(
        [f"SEGMENT_ID_{summary[0]}: {summary[1]}" for summary in samples_to_summarise]
    )
    messages = [
        {"role": "system", "content": vanilla_topic_model_system_prompt},
        {
            "role": "user",
            "content": vanilla_topic_model_user_prompt.format(
                docs_with_ids=docs_with_ids,
                user_prompt=user_prompt,
                response_language=response_language,
            ),
        },
    ]
    tentative_aspects_response = run_formated_llm_call(messages, TopicModelResponse)
    tentative_aspects = tentative_aspects_response["topics"]
    aspect_response_list = await fallback_get_aspect_response_list(
        tentative_aspects,
        docs_with_ids,
        user_prompt,
        segment_2_transcript,
        response_language=response_language,
    )
    views_dict = summarise_aspects(
        aspect_response_list,
        response_language=response_language,
        user_prompt=user_prompt,
    )
    views_dict["aspects"] = aspect_response_list
    views_dict["seed"] = user_prompt
    views_dict["language"] = response_language
    views_dict["user_input"] = user_input
    views_dict["user_input_description"] = user_input_description
    response = {"view": views_dict}
    update_directus(response, project_analysis_run_id)
    return response
