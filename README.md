# Topic Modeler with RAG Integration

A topic modeling and content analysis system that combines BERTopic with Retrieval-Augmented Generation (RAG) to generate in depth reports from data segments of Dembrane ECHO.

## 🚀 Features

- **Advanced Topic Modeling**: Uses BERTopic with hierarchical clustering for intelligent topic extraction
- **RAG Integration**: Retrieves relevant context from external knowledge bases to enhance analysis
- **GPU Acceleration**: Leverages CUDA for faster processing when available
- **Multi-language Support**: Generate reports in multiple languages
- **Professional Reporting**: Creates structured, journalistic-style reports with proper citations
- **Serverless Deployment**: Built for RunPod serverless infrastructure
- **Flexible Input**: Accepts segment IDs and custom user prompts for tailored analysis

## 🛠️ Technologies

- **Machine Learning**: BERTopic, Sentence Transformers, UMAP, HDBSCAN
- **AI/LLM**: Azure OpenAI, LiteLLM
- **Backend**: PyTorch, Directus SDK
- **Infrastructure**: RunPod, Docker
- **Data Processing**: Pydantic for data validation and structured outputs

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, falls back to CPU)
- Access to Azure OpenAI API
- Directus instance for data management
- External RAG server for context retrieval

## 🔧 Installation

### Using Docker (Recommended)

1. Clone the repository
2. Build the Docker image:
```bash
docker build -t topic-modeler .
```

3. Run the container:
```bash
docker run --gpus all -p 8000:8000 topic-modeler
```

### Local Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (see Configuration section)

3. Run the handler:
```bash
python handler.py
```

## ⚙️ Configuration

Set the following environment variables:

```bash
# Azure OpenAI Configuration
AZURE_API_KEY=your_azure_api_key
AZURE_API_BASE=your_azure_endpoint
AZURE_API_VERSION=2024-02-01
AZURE_MODEL=gpt-4

# Directus Configuration
DIRECTUS_BASE_URL=https://your-directus-instance.com
DIRECTUS_TOKEN=your_directus_token

# RAG Server Configuration
RAG_SERVER_URL=https://your-rag-server.com
RAG_SERVER_AUTH_TOKEN=your_rag_auth_token

# Optional: Force CPU usage
RUN_CPU=False
```

## 📊 Usage

### Input Format

The system accepts JSON input with the following structure:

```json
{
  "input": {
    "segment_ids": [360, 361, 362],
    "user_prompt": "Please summarize all the topics related to AI development",
    "response_language": "en"
  }
}
```

### Parameters

- **segment_ids**: List of segment IDs to analyze
- **user_prompt**: Custom prompt describing what analysis you want
- **response_language**: Target language for the output (e.g., "en", "es", "fr")

### Output Format

The system returns structured analysis with:

```json
{
  "title": "AI Development Trends Analysis",
  "description": "Comprehensive analysis of AI development topics...",
  "summary": "## Key Findings\n\n### Topic 1: Machine Learning\n...",
  "seed": "Intial prompt",

  "aspects": [
    {
      "title": "Machine Learning Advances",
      "description": "Recent developments in ML algorithms",
      "summary": "Detailed analysis...",
      "image_url":"",
      "segments": [
        {
          "segment_id": 360,
          "description": "Discusses neural network improvements"
        }
      ]
    }
  ]
}
```

## 🧠 How It Works

1. **Input Processing**: Receives segment IDs and user prompts
2. **Data Retrieval**: Fetches relevant data segments from Directus
3. **RAG Enhancement**: Retrieves additional context using external RAG server
4. **Topic Modeling**: Applies BERTopic for intelligent topic extraction
5. **Content Generation**: Uses Azure OpenAI to generate professional reports
6. **Output Structuring**: Returns well-formatted, citable analysis

## 🎯 Use Cases

- **Content Analysis**: Analyze large volumes of text for key themes
- **Research Synthesis**: Combine multiple data sources into coherent reports
- **Journalistic Research**: Generate professional reports from raw data
- **Business Intelligence**: Extract insights from corporate communications
- **Academic Research**: Analyze literature and research data

## 🔍 Key Components

- **`handler.py`**: Main entry point for RunPod serverless function
- **`utils.py`**: Core functionality including topic modeling and RAG integration
- **`prompts.py`**: Professional prompt templates for different analysis types
- **`data_model.py`**: Pydantic models for structured input/output validation

## 🚀 Deployment

The system is designed for serverless deployment on RunPod:

1. Build the Docker image
2. Deploy to RunPod with GPU support
3. Configure environment variables in RunPod dashboard
4. Test with sample inputs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔧 Troubleshooting

### Common Issues

- **CUDA Out of Memory**: Set `RUN_CPU=True` to use CPU-only mode
- **RAG Server Connection**: Verify RAG_SERVER_URL and authentication tokens
- **Topic Quality**: Adjust `min_topic_size` parameter in topic model initialization
- **Language Issues**: Ensure proper language codes in `response_language` field

### Performance Tips

- Use GPU acceleration for faster processing
- Adjust `top_k` parameter in RAG queries based on your needs
- Monitor memory usage with large document sets
- Consider batch processing for multiple analyses

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the configuration requirements
- Ensure all environment variables are properly set