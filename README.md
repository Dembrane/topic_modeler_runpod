# Hierarchical Topic Modeling RunPod Handler

This RunPod handler performs hierarchical topic modeling using BERTopic with GPU acceleration. It takes a list of documents as input and returns a comprehensive hierarchical topic analysis.

## Features

- **GPU Acceleration**: Uses CUDA-enabled sentence transformers for fast embeddings
- **Hierarchical Topic Modeling**: Generates topic hierarchies to understand relationships between topics
- **Comprehensive Output**: Returns hierarchical topics dataframe, topic information, and topic representations
- **Memory Management**: Automatic GPU memory cleanup after processing
- **Input Validation**: Robust error handling and input validation

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Input Format

The handler expects an input JSON with the following structure:

```json
{
  "input": {
    "documents": [
      "Your first document text here...",
      "Your second document text here...",
      "Your third document text here...",
      "..."
    ]
  }
}
```

### Input Requirements

- **documents**: A list of strings (documents)
- **Minimum**: At least 5 non-empty documents are required
- **Content**: Documents should be meaningful text for topic modeling

## Output Format

The handler returns a JSON response with the following structure:

```json
{
  "status": "success",
  "data": {
    "hierarchical_topics": [
      {
        "Parent_ID": 0,
        "Parent_Name": "parent_topic_name",
        "Topics": [1, 2],
        "Child_Left_ID": 1,
        "Child_Left_Name": "child_left_name",
        "Child_Right_ID": 2,
        "Child_Right_Name": "child_right_name",
        "Distance": 0.5
      }
    ],
    "topic_info": [
      {
        "Topic": 0,
        "Count": 10,
        "Name": "topic_representation",
        "Representation": ["word1", "word2", "word3"]
      }
    ],
    "topic_representations": {
      "0": [
        {"word": "example", "score": 0.123},
        {"word": "topic", "score": 0.098}
      ]
    },
    "num_topics": 5,
    "num_documents": 100,
    "topics_per_document": [0, 1, 2, 0, 1, ...]
  }
}
```

### Output Fields

- **hierarchical_topics**: DataFrame showing the hierarchical structure of topics
- **topic_info**: Information about each discovered topic
- **topic_representations**: Top words and their scores for each topic
- **num_topics**: Total number of topics discovered
- **num_documents**: Number of documents processed
- **topics_per_document**: Topic assignment for each document

## Usage

### Local Testing

Run the test script to verify the handler works:

```bash
python test_handler.py
```

### RunPod Deployment

1. Upload the handler files to your RunPod serverless endpoint
2. Ensure GPU support is enabled
3. Make API calls with your document data

### Example API Call

```python
import requests
import json

# Your RunPod endpoint URL
url = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"

headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}

data = {
    "input": {
        "documents": [
            "Machine learning is transforming healthcare with predictive analytics.",
            "Electric vehicles are becoming mainstream as battery technology improves.",
            "Climate change is affecting global weather patterns significantly.",
            "Artificial intelligence is revolutionizing various industries.",
            "Renewable energy sources are becoming more cost-effective.",
            # Add more documents...
        ]
    }
}

response = requests.post(url, headers=headers, json=data)
result = response.json()

print(json.dumps(result, indent=2))
```

## GPU Optimization

The handler is optimized for GPU usage:

- **Sentence Transformers**: Automatically uses CUDA when available
- **Memory Management**: Clears GPU cache after processing
- **Efficient Embeddings**: Uses lightweight but effective models

## Error Handling

The handler includes comprehensive error handling:

- Input validation for document format and count
- GPU memory management
- Detailed error messages for debugging

## Hierarchical Topic Analysis

Based on the [BERTopic hierarchical topic modeling](https://maartengr.github.io/BERTopic/getting_started/hierarchicaltopics/hierarchicaltopics.html) approach, the handler:

1. Creates embeddings for all documents using sentence transformers
2. Performs dimensionality reduction with UMAP
3. Clusters documents using HDBSCAN
4. Generates c-TF-IDF representations for topics
5. Creates hierarchical structure using distance matrices
6. Returns comprehensive hierarchical topic information

## Performance Tips

- **Batch Size**: Process documents in reasonable batches (100-10,000 documents)
- **GPU Memory**: Monitor GPU memory usage for large document sets
- **Document Length**: Longer documents may require more processing time
- **Topic Quality**: More documents generally lead to better topic quality

## Support

For issues or questions about the hierarchical topic modeling handler, please check:

1. Input format matches the expected structure
2. Documents contain meaningful text
3. Sufficient GPU memory is available
4. All dependencies are properly installed 