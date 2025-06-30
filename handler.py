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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

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
            embedding_model = SentenceTransformer('intfloat/multilingual-e5-large-instruct', device=device)
        else:
            # Use a smaller model for CPU
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Configure UMAP for dimensionality reduction
        # UMAP doesn't have direct GPU support, but we keep it lightweight
        umap_model = UMAP(
            n_neighbors=15, 
            n_components=5, 
            min_dist=0.0, 
            metric='cosine',
            random_state=42
        )
        
        # Configure HDBSCAN for clustering
        hdbscan_model = HDBSCAN(
            min_cluster_size=10,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        # Configure vectorizer
        vectorizer_model = CountVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            max_features=5000
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
            verbose=True
        )
        
        logger.info("BERTopic model initialized successfully")
        return topic_model
    
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise e

topic_model = initialize_topic_model()

def process_hierarchical_topics(documents):
    """Process documents and generate hierarchical topic model"""
    try:
        logger.info(f"Processing {len(documents)} documents")
        
        # Fit the model
        logger.info("Fitting BERTopic model...")
        topics, probabilities = topic_model.fit_transform(documents)
        
        # Generate hierarchical topics
        logger.info("Generating hierarchical topics...")
        hierarchical_topics = topic_model.hierarchical_topics(documents)
        
        # Get topic information
        topic_info = topic_model.get_topic_info()
        
        # Prepare response data
        response_data = {
            "hierarchical_topics": hierarchical_topics.to_dict('records'),
            "topic_info": topic_info.to_dict('records'),
            "num_topics": len(topic_model.get_topics()),
            "num_documents": len(documents),
            "topics_per_document": topics.tolist() if isinstance(topics, list) else topics,
        }
        
        # Add topic representations
        all_topics = topic_model.get_topics()
        topic_representations = {}
        for topic_id, words in all_topics.items():
            if topic_id != -1:  # Exclude outlier topic
                topic_representations[topic_id] = [{"word": word, "score": score} for word, score in words[:10]]
        
        response_data["topic_representations"] = topic_representations
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("Hierarchical topic modeling completed successfully")
        return response_data
    
    except Exception as e:
        logger.error(f"Error in hierarchical topic processing: {str(e)}")
        # Clean up GPU memory in case of error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise e

def handler(event):
    """RunPod handler function for hierarchical topic modeling"""
    try:
        # Extract input data
        input_data = event.get("input", {})
        
        # Validate input
        if "documents" not in input_data:
            return {
                "error": "Missing 'documents' field in input. Please provide a list of documents."
            }
        
        documents = input_data["documents"]
        
        # Validate documents
        if not isinstance(documents, list):
            return {
                "error": "Documents must be provided as a list of strings."
            }
        
        if len(documents) < 5:
            return {
                "error": "At least 5 documents are required for topic modeling."
            }
        
        # Filter out empty documents
        documents = [doc for doc in documents if doc and isinstance(doc, str) and doc.strip()]
        
        if len(documents) < 5:
            return {
                "error": "At least 5 non-empty documents are required for topic modeling."
            }
        
        logger.info(f"Processing {len(documents)} documents for hierarchical topic modeling")
        
        # Process hierarchical topics
        result = process_hierarchical_topics(documents)
        
        return {
            "status": "success",
            "data": result
        }
    
    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        logger.error(error_msg)
        return {
            "error": error_msg
        }

if __name__ == "__main__":
    # Start the RunPod serverless handler
    runpod.serverless.start({"handler": handler})
