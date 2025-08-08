import os
from typing import List, Optional

import torch
from umap import UMAP
from runpod import RunPodLogger
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

logger = RunPodLogger()


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
