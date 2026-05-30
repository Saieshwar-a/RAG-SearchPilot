# Simple RAG (Retrieval-Augmented Generation)

A Python-based implementation of a Retrieval-Augmented Generation (RAG) system that combines document retrieval with large language models (LLMs) to provide accurate, context-aware responses based on custom data sources.

## Overview

This project builds a RAG pipeline that:
- **Ingests** PDF and text documents from a specified directory
- **Processes** documents into manageable chunks
- **Embeds** documents using sentence transformers
- **Stores** embeddings in a vector database (Chroma)
- **Retrieves** relevant documents based on semantic similarity
- **Generates** responses using Ollama LLM with retrieved context

## Features

- 📄 **Multi-format Document Support**: Process PDF and text files
- 🔍 **Semantic Search**: Find relevant documents using embeddings
- 🤖 **LLM Integration**: Generate responses using Ollama
- 💾 **Vector Store**: Persistent storage with Chroma database
- 🔄 **Pipeline Notebooks**: Interactive Jupyter notebooks for exploration

## Project Structure

```
SimpleRag/
├── app.py                      # Main entry point
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── DataIngestion.ipynb         # Data ingestion pipeline
├── VectorDbPipeline.ipynb      # Vector database setup pipeline
├── Data/
│   ├── PdfData/               # PDF documents for ingestion
│   ├── TextData/              # Text files (KeyStages.txt, RagDescription.txt)
│   └── vector_store/          # Chroma vector database
├── src/
│   ├── __init__.py
│   ├── RAGAssistant.py        # Main RAG orchestrator
│   └── modules/
│       ├── DataProcessor.py       # Document processing & chunking
│       ├── EmbeddingManager.py    # Embedding generation
│       ├── RAGEngine.py           # LLM response generation
│       ├── RAGRetriver.py         # Document retrieval
│       └── VectorStore.py         # Vector database management
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Ollama installed and running (for LLM inference)

### Setup

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - Windows:
     ```bash
     .\venv\Scripts\Activate.ps1
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Start Ollama service** (if not already running):
   ```bash
   ollama serve
   ```

## Usage

### Basic Usage

```python
from src import RAGAssistant

# Initialize the RAG assistant with your data directory
# On first run: builds embeddings and stores them in vector DB
# On subsequent runs: reuses stored embeddings (much faster!)
rag = RAGAssistant("./Data/PdfData")

# Ask a question
response = rag.ask("What is the Attention Mechanism in Neural Networks?")

# Print the response
print(response["response"])
```

### Rebuild Vector Store

If you've added new documents or want to regenerate embeddings:

```python
from src import RAGAssistant

# Set rebuild=True to force rebuilding the knowledge base
rag = RAGAssistant("./Data/PdfData", rebuild=True)

response = rag.ask("Your question here")
```

### Vector Store Persistence

The vector database is automatically persisted to disk in `Data/vector_store/`. This means:
- **First run**: Documents are processed and embeddings are generated (takes time)
- **Subsequent runs**: Existing embeddings are loaded from disk (much faster!)
- **rebuild=True**: Forces re-processing and re-embedding of all documents

### Run the Application

```bash
python app.py
```

### Retrieve Documents

You can also retrieve relevant documents without generating a response:

```python
from src import RAGAssistant

rag = RAGAssistant("./Data/PdfData")

# Retrieve top 5 relevant documents
docs = rag.retrieve("Your query here", top_k=5)
```

## Architecture

### Components

**1. DataProcessor** (`src/modules/DataProcessor.py`)
- Loads PDF and text documents
- Splits documents into chunks
- Prepares documents for embedding

**2. EmbeddingManager** (`src/modules/EmbeddingManager.py`)
- Generates embeddings using sentence transformers
- Handles batch embedding generation

**3. VectorStore** (`src/modules/VectorStore.py`)
- Manages Chroma vector database
- Stores and indexes embeddings
- Handles document metadata

**4. RAGRetriver** (`src/modules/RAGRetriver.py`)
- Performs semantic similarity search
- Retrieves top-k relevant documents
- Filters results by score threshold

**5. RAGEngine** (`src/modules/RAGEngine.py`)
- Uses Ollama LLM for response generation
- Constructs prompts with retrieved context
- Generates contextual responses

**6. RAGAssistant** (`src/RAGAssistant.py`)
- Orchestrates the entire pipeline
- Builds the knowledge base on initialization
- Provides `ask()` and `retrieve()` methods

## Dependencies

- **langchain** - LLM framework and utilities
- **langchain-core** - Core LangChain components
- **langchain-community** - Community integrations
- **pypdf** - PDF processing
- **pymupdf** - Alternative PDF processing
- **sentence-transformers** - Embedding generation
- **chromadb** - Vector database

## Configuration

### Default Settings

- **Embedding Model**: `sentence-transformers` (auto-selected)
- **LLM Model**: `mistral:latest` (via Ollama)
- **Ollama Host**: `http://localhost:11434`
- **Vector Database**: Chroma (persistent storage in `Data/vector_store/`)

### Customization

You can modify the model and host in `src/modules/RAGEngine.py`:

```python
engine = RAGEngine(
    retriever,
    model_name="llama2:latest",  # Change model
    ollama_host="http://your-host:11434"
)
```

## Troubleshooting

### "Connection refused" error
- Ensure Ollama is running: `ollama serve`
- Check Ollama host is accessible at the configured address

### No documents retrieved
- Verify documents exist in the data directory
- Check document encoding and format compatibility
- Review `DataProcessor.py` for supported file types

### Out of memory
- Reduce batch sizes in `EmbeddingManager.py`
- Process smaller document chunks

## Future Enhancements

- [ ] Support for more document formats (DOCX, XLSX, etc.)
- [ ] Query rewriting and decomposition
- [ ] Caching for frequently asked questions
- [ ] Web interface
- [ ] Streaming responses
- [ ] Multi-turn conversation support

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
