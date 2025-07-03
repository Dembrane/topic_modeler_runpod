import os
import json
import logging
from typing import Dict, List, Optional

import torch
import requests
from umap import UMAP
from dotenv import load_dotenv
from hdbscan import HDBSCAN
from litellm import completion
from prompts import (
    rag_user_prompt,
    rag_system_prompt,
    initial_rag_prompt,
    topic_model_prompt,
    view_summary_prompt,
    topic_model_system_prompt,
    view_summary_system_prompt,
    vanilla_topic_model_user_prompt,
    vanilla_topic_model_system_prompt,
)
from bertopic import BERTopic
from pydantic import BaseModel
from data_model import Aspect, TopicModelResponse, ViewSummaryResponse
from litellm.utils import token_counter
from directus_sdk_py import DirectusClient
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

DIRECTUS_BASE_URL = str(os.getenv("DIRECTUS_BASE_URL"))
DIRECTUS_TOKEN = str(os.getenv("DIRECTUS_TOKEN"))


directus = DirectusClient(url=DIRECTUS_BASE_URL, token=DIRECTUS_TOKEN)


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
            embedding_model = SentenceTransformer(
                "intfloat/multilingual-e5-large-instruct", device=device
            )
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


def run_topic_model_hierarchical(topic_model, docs, topics: Optional[List[str]] = None):
    """
    Run hierarchical topic modeling on the provided documents.

    Args:
        topic_model: Initialized BERTopic model
        docs: List of documents to process
        topics: Optional list of predefined topics

    Returns:
        tuple: Contains:
            - topics: List of identified topics
            - probs: Topic probabilities for each document
            - hierarchical_topics: Hierarchical structure of topics
    """
    topics, probs = topic_model.fit_transform(docs)
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


def get_rag_prompt(
    query: str,
    segment_ids: Optional[List[str]] = None,
    rag_server_url: Optional[str] = None,
    auth_token: Optional[str] = None, # TODO: @sameer, please look into how to make this call with token
) -> str:
    """
    Retrieve RAG prompt by calling the external RAG server API.

    Args:
        query: The query string to send to the RAG server
        segment_ids: Optional list of segment IDs to include in the RAG query
        rag_server_url: Optional base URL of the RAG server (defaults to env variable)
        auth_token: Optional bearer token for authentication (defaults to env variable)

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
    cookies = {
        "directus_session_token": os.getenv("RAG_SERVER_AUTH_TOKEN")# TODO: @sameer, please look into how to make this call with token
    }

    try:
        logger.info(f"Making RAG API request to {url}")
        response = requests.post(url, json=payload, headers=headers, cookies=cookies, timeout=120)
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


def get_aspect_response_list(
    aspects: List[str],
    segment_ids: List[str],
    segment_2_transcript: Dict[str, str],
    auth_token: Optional[str] = None,
    response_language: str = "en",
):
    """
    Generate detailed responses for each aspect using RAG and LLM processing.

    Args:
        aspects: List of aspect topics to analyze
        segment_ids: List of segment IDs to process
        segment_2_transcript: Dictionary mapping segment IDs to their transcripts
        auth_token: Optional authentication token for RAG server
        response_language: Language code for response generation (default: 'en')

    Returns:
        List[Dict]: List of aspect responses, each containing:
            - title: Aspect title
            - description: Detailed description
            - summary: Brief summary
            - segments: List of relevant segments with transcripts
            - image_url: URL for any associated image
    """
    aspect_response_list = []
    for tentative_aspect_topic in aspects:
        formated_initial_rag_prompt = initial_rag_prompt.format(
            tentative_aspect_topic=tentative_aspect_topic
        )
        rag_prompt = get_rag_prompt(
            formated_initial_rag_prompt,
            segment_ids=[str(segment_id) for segment_id in segment_ids],
            auth_token=auth_token,
        )
        rag_messages = [
            {"role": "system", "content": rag_system_prompt},
            {"role": "user", "content": rag_user_prompt.format(response_language=response_language, input_report=rag_prompt)},
        ]
        rag_response = run_formated_llm_call(rag_messages, Aspect)
        rag_response["image_url"] = ""
        updated_segments = []
        for segment in rag_response["segments"]:
            id = segment["segment_id"]
            if id in segment_2_transcript.keys():
                segment.pop("segment_id")
                segment["id"] = id
                segment["conversation_id"] = ""
                segment["verbatim_transcript"] = segment_2_transcript[id]
                updated_segments.append(segment)
        rag_response["segments"] = updated_segments
        aspect_response_list.append(rag_response)
    return aspect_response_list


def summarise_aspects(aspect_response_list: List[Dict], response_language: str = "en"):
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
            "content": view_summary_prompt.format(
                view_text=view_text, response_language=response_language
            ),
        },
    ]
    view_response = run_formated_llm_call(messages, ViewSummaryResponse)
    return view_response


def get_views_aspects(
    segment_ids: List[str],
    user_prompt: str,
    response_language: str|None = None,
    threshold_context_length: int = 100000,
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

    segments = directus.get_items(
        "conversation_segment",
        {
            "query": {
                "filter": {"id": {"_in": segment_ids}},
                "fields": ["id", "contextual_transcript", "transcript"],
            },
        },
    )

    # Type-safe dictionary comprehension with explicit type checking
    segment_2_transcript: Dict[int, str] = {}
    raw_docs: List[str] = []
    
    for segment in segments:
        if isinstance(segment, dict):
            if 'id' in segment and 'transcript' in segment:
                segment_2_transcript[int(segment['id'])] = str(segment['transcript'])
            if 'contextual_transcript' in segment:
                raw_docs.append(str(segment['contextual_transcript']))

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
        docs_with_ids = "\n\n".join([f"{doc_id}: {doc}" for doc_id, doc in enumerate(docs)])
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
        print(hierarchical_topics)
        messages = [
            {"role": "system", "content": topic_model_system_prompt},
            {
                "role": "user",
                "content": topic_model_prompt.format(
                    global_topic_hierarchy=topic_model.get_topic_tree(hierarchical_topics),
                    user_prompt="Please summarise all the topics.",
                    response_language=response_language,
                ),
            },
        ]
        tentative_aspects_response = run_formated_llm_call(messages, TopicModelResponse)

    tentative_aspects = tentative_aspects_response["topics"]
    aspect_response_list = get_aspect_response_list(
        tentative_aspects,
        segment_ids,
        segment_2_transcript,
        auth_token=os.getenv("RAG_SERVER_AUTH_TOKEN"),
        response_language=response_language,
    )
    views_dict = summarise_aspects(aspect_response_list, response_language=response_language)
    views_dict["aspects"] = aspect_response_list
    views_dict["seed"] = user_prompt
    views_dict["language"] = response_language
    response = {"view":views_dict}
    return response
