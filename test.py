import runpod
import torch
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
import logging
import gc
from dotenv import load_dotenv
import os
from typing import Optional, List, Dict
from sklearn.datasets import fetch_20newsgroups
from litellm import completion
from pydantic import BaseModel
import json
import re
import requests
from requests.exceptions import RequestException
import random
from prompts import (
    topic_model_prompt,
    topic_model_system_prompt,
    initial_rag_prompt,
    rag_system_prompt,
    view_summary_system_prompt,
    view_summary_prompt,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


class References(BaseModel):
    segment_id: int
    description: str
    verbatim_transcript: str


class Aspect(BaseModel):
    title: str
    description: str
    summary: str
    segments: List[References]


class ViewSummaryResponse(BaseModel):
    title: str
    description: str
    summary: str


class TopicModelResponse(BaseModel):
    topics: List[str]


class GetLightragQueryRequest(BaseModel):
    query: str
    conversation_history: Optional[List[Dict[str, str]]] = None
    echo_segment_ids: Optional[List[str]] = None
    echo_conversation_ids: Optional[List[str]] = None
    echo_project_ids: Optional[List[str]] = None
    auto_select_bool: bool = False
    get_transcripts: bool = False
    top_k: int = 60


def initialize_topic_model():
    """Initialize BERTopic model with GPU acceleration"""
    try:
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if os.environ.get("RUN_CPU") == "True":
            device = "cpu"
        logger.info(f"Using device: {device}")

        # Initialize sentence transformer with GPU support
        # if device is cuda:
        if device == "cuda":
            embedding_model = SentenceTransformer(
                "intfloat/multilingual-e5-large-instruct", device=device
            )
        else:
            # Use a smaller model for CPU
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Configure UMAP for dimensionality reduction
        # UMAP doesn't have direct GPU support, but we keep it lightweight
        umap_model = UMAP(
            n_neighbors=15, n_components=10, metric="cosine", random_state=42
        )

        # Configure HDBSCAN for clustering
        hdbscan_model = HDBSCAN(
            min_cluster_size=5, metric="euclidean", prediction_data=True
        )

        # Configure vectorizer
        vectorizer_model = CountVectorizer(
            ngram_range=(1, 2), stop_words="english", min_df=2, max_features=5000
        )

        # Configure c-TF-IDF
        ctfidf_model = ClassTfidfTransformer()

        # Initialize BERTopic model
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
    """Run the topic model on the documents"""
    topics, probs = topic_model.fit_transform(docs)
    hierarchical_topics = topic_model.hierarchical_topics(docs)
    return topics, probs, hierarchical_topics


def run_formated_llm_call(
    messages: List[Dict[str, str]], response_format: BaseModel
):
    """Run the LLM call on the prompt"""
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
    auth_token: Optional[str] = None,
) -> str:
    """
    Get RAG prompt by calling the external RAG server API

    Args:
        query: The query string to send to the RAG server
        segment_ids: List of segment IDs to include in the RAG query
        rag_server_url: Base URL of the RAG server (defaults to env variable RAG_SERVER_URL)
        auth_token: Bearer token for authentication (defaults to env variable RAG_SERVER_AUTH_TOKEN)

    Returns:
        RAG prompt string from the server
    """
    # Get server URL from environment or parameter
    if rag_server_url is None:
        rag_server_url = os.getenv("RAG_SERVER_URL")
        if not rag_server_url:
            raise ValueError(
                "RAG_SERVER_URL environment variable not set and no rag_server_url provided"
            )

    # API endpoint URL
    url = f"{rag_server_url.rstrip('/')}/stateless/rag/get_lightrag_prompt"

    # Request payload based on GetLightragQueryRequest model
    payload = {
        "query": query,
        "conversation_history": None,
        "echo_segment_ids": segment_ids,
        "echo_conversation_ids": None,
        "echo_project_ids": None,
        "auto_select_bool": False,
        "get_transcripts": False,
        "top_k": 60
    }

    # Headers
    headers = {
        "Content-Type": "application/json"
    }

    # Add Bearer authentication header
    if auth_token is None:
        auth_token = os.getenv("RAG_SERVER_AUTH_TOKEN")

    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    try:
        logger.info(f"Making RAG API request to {url}")
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        # The endpoint returns a string directly
        result = response.text
        logger.info("Successfully retrieved RAG prompt")
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response text: {e.response.text}")
        raise Exception(f"Failed to get RAG prompt from server: {str(e)}")


def get_aspect_response_list(
    aspects: List[str], segment_ids: List[str], auth_token: Optional[str] = None
):
    """Get the response list for the aspects"""
    aspect_response_list = []
    for tentative_aspect_topic in aspects:
        formated_initial_rag_prompt = initial_rag_prompt.format(
            tentative_aspect_topic=tentative_aspect_topic
        )
        rag_prompt = get_rag_prompt(
            formated_initial_rag_prompt, segment_ids=segment_ids, auth_token=auth_token
        )
        rag_messages = [
            {"role": "system", "content": rag_system_prompt},
            {"role": "user", "content": rag_prompt},
        ]
        rag_response = run_formated_llm_call(rag_messages, Aspect)
        aspect_response_list.append(rag_response)
    return aspect_response_list


def summarise_aspects(aspect_response_list: List[Aspect]):
    """Summarise the aspects"""
    aspect_texts = [
        f"{aspect.title}\n{aspect.description}\n{aspect.summary}"
        for aspect in aspect_response_list
    ]
    view_text = "\n\n".join(aspect_texts)
    messages = [
        {"role": "system", "content": view_summary_system_prompt},
        {
            "role": "user",
            "content": view_summary_prompt.format(
                view_text=view_text,
            ),
        },
    ]
    response = run_formated_llm_call(messages, ViewSummaryResponse)
    return response


if __name__ == "__main__":
    topic_model = initialize_topic_model()
    segment_ids = ["123", "456", "789"]
    docs = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))[
        "data"
    ]
    # sample 2000 docs
    random.seed(24)
    docs = random.sample(docs, 500)
    topics, probs, hierarchical_topics = run_topic_model_hierarchical(topic_model, docs)
    print(hierarchical_topics)
    messages = [
        {"role": "system", "content": topic_model_system_prompt},
        {
            "role": "user",
            "content": topic_model_prompt.format(
                global_topic_hierarchy=topic_model.get_topic_tree(hierarchical_topics),
                user_prompt="Please summarise all the topics.",
            ),
        },
    ]
    response = run_formated_llm_call(messages, TopicModelResponse)
    aspects = response["topics"]
    aspect_response_list = get_aspect_response_list(
        aspects, segment_ids, auth_token=os.getenv("RAG_SERVER_AUTH_TOKEN")
    )
    summarised_aspects_dict = summarise_aspects(aspect_response_list)
    summarised_aspects_dict["aspects"] = aspect_response_list


    # aspect_response_list = []
    # for tentative_aspect_topic in aspects:
    #     initial_rag_prompt = initial_rag_prompt.format(tentative_aspect_topic=tentative_aspect_topic)
    #     rag_prompt = get_rag_prompt(initial_rag_prompt, segment_ids=segment_ids)
    #     rag_messages = [
    #         {
    #             "role": "system",
    #             "content": "You are a helpful summarising assistant that can help with summarising the following text.",
    #         },
    #         {"role": "user", "content": rag_prompt},
    #     ]
    #     rag_response = run_formated_llm_call(rag_messages, Aspect)
    #     aspect_response_list.append(rag_response)
