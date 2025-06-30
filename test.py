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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


class Segment(BaseModel):
    id: str
    segment_id: str
    description: str
    verbatim_transcript: str


class Aspect(BaseModel):
    title: str
    description: str
    summary: str
    segments: List[Segment]


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


def run_topic_model(topic_model, docs, topics: Optional[List[str]] = None):
    """Run the topic model on the documents"""
    topics, probs = topic_model.fit_transform(docs)
    return topics, probs


def run_formated_llm_call(messages: List[Dict[str, str]], response_format: BaseModel):
    """Run the LLM call on the prompt"""
    # TODO: Implement LiteLLM call
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
) -> str:
    """
    Get RAG prompt by calling the external RAG server API

    Args:
        aspects: List of topic aspects to use as the query
        segment_ids: List of segment IDs to include in the RAG query
        rag_server_url: Base URL of the RAG server (defaults to env variable RAG_SERVER_URL)

    Returns:
        RAG prompt string from the server
    """
    try:
        # Get server URL from environment or parameter
        if rag_server_url is None:
            rag_server_url = os.getenv("RAG_SERVER_URL")
            if not rag_server_url:
                raise ValueError(
                    "RAG_SERVER_URL environment variable not set and no rag_server_url provided"
                )

        # Construct the full API endpoint URL
        endpoint = f"{rag_server_url.rstrip('/')}/rag/get_lightrag_prompt"

        # Create query from aspects
        query = (
            "Based on the following topic aspects, provide relevant information: "
            + ", ".join(aspects)
        )

        # Prepare the request payload
        payload = GetLightragQueryRequest(
            query=query,
            echo_segment_ids=segment_ids,
            auto_select_bool=segment_ids
            is None,  # Auto-select if no specific segments provided
            top_k=60,
        )

        # Make the API request
        headers = {"Content-Type": "application/json"}

        # Add authentication headers if needed
        auth_token = os.getenv("RAG_SERVER_AUTH_TOKEN")
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        logger.info(f"Making RAG API request to {endpoint}")
        response = requests.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=30,  # 30 second timeout
        )

        # Check if request was successful
        response.raise_for_status()

        # Return the RAG prompt
        rag_prompt = response.text
        logger.info("Successfully retrieved RAG prompt")
        return rag_prompt

    except RequestException as e:
        logger.error(f"HTTP request failed: {str(e)}")
        raise Exception(f"Failed to get RAG prompt from server: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting RAG prompt: {str(e)}")
        raise e


if __name__ == "__main__":
    import random
    from prompts import topic_model_prompt, topic_model_system_prompt

    topic_model = initialize_topic_model()
    segment_ids = ["123", "456", "789"]
    docs = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))[
        "data"
    ]
    # sample 2000 docs
    random.seed(21)
    # docs = random.sample(docs, 2000)
    topics, probs = run_topic_model(topic_model, docs)
    hierarchical_topics = topic_model.hierarchical_topics(docs)
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
    aspect_response_list = []
    for tentative_aspect_topic in aspects:
        prompt = f'''
        Please create a detailed description of the following topic: {tentative_aspect_topic}
        '''

        # Please return a professional report on the topic. Here is the explanation of the fields to fill :
        # title: string - A detailed title of the topic;
        # description: string - A short description of the topic;
        # summary: string - Multi section markdown report of the topic;
        # segments: ARRAY{{
        #     id: int - The id of the segment, always a number;
        #     description: string - A short description of the segment and its relevance to the topic;
        #     verbatim_transcript: string - The verbatim transcript of the segment;
        # }}
        # '''
        rag_prompt = get_rag_prompt(prompt, segment_ids=segment_ids)
        rag_messages = [
            {
                "role": "system",
                "content": "You are a helpful summarising assistant that can help with summarising the following text.",
            },
            {"role": "user", "content": rag_prompt},
        ]
        rag_response = run_formated_llm_call(rag_messages, Aspect)
        aspect_response_list.append(rag_response)
        print("RAG Prompt:")
        print(rag_prompt)

    # Get RAG prompt using the extracted aspects
    try:
        print("RAG Prompt:")
        print(rag_prompt)
    except Exception as e:
        logger.error(f"Failed to get RAG prompt: {e}")
        print(f"Error getting RAG prompt: {e}")
