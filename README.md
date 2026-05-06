# QA RAG Project

A **Question Answering Retrieval-Augmented Generation (RAG)** system that leverages FastAPI, LLaMA Index, Qdrant vector database, and OpenAI to enable intelligent document-based question answering with event-driven workflows.

## 📋 Project Overview

This project implements a production-ready RAG pipeline featuring:

- **Document Ingestion & Indexing**: Automatically process and embed documents using LLaMA Index and OpenAI embeddings
- **Vector Search**: Store and retrieve document vectors using Qdrant vector database
- **Event-Driven Architecture**: Handle asynchronous workflows and events using Inngest
- **REST API**: FastAPI-based API for querying and document operations
- **Batch Processing**: Efficient PDF ingestion and embedding generation

## ✨ Key Features

- 🚀 **Fast & Scalable**: Built with FastAPI and Inngest for high-performance async operations
- 🔍 **Semantic Search**: Powered by OpenAI embeddings and Qdrant vector database
- 📄 **Document Support**: Handle multiple file formats via LLaMA Index readers
- 🎯 **RAG Pipeline**: End-to-end retrieval-augmented generation for accurate Q&A
- 🔄 **Event-Driven Workflows**: Scalable background jobs with Inngest
- 🗄️ **Vector Storage**: Persistent vector embeddings with Qdrant

## 🏗️ Architecture

```
┌─────────────┐
│   FastAPI   │  REST API Endpoints
└──────┬──────┘
       │
       ├─→ Inngest  (Event workflows)
       │
       ├─→ LLaMA Index  (Document processing)
       │
       ├─→ OpenAI API  (Embeddings & LLM)
       │
       └─→ Qdrant Vector DB  (Storage & retrieval)
```

## 🛠️ Prerequisites

- **Python**: 3.13 or higher
- **Qdrant**: Running instance (local or Docker)
- **OpenAI API Key**: For embeddings and LLM calls
- **Package Manager**: `uv` (recommended) or `pip`

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/bhanuanishakkineni/qa-rag-project.git
cd qa-rag-project
```

### 2. Set Up Python Environment

Using `uv` (fastest):
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

Or using `pip`:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install .
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Start Qdrant Vector Database

**Using Docker** (recommended):
```bash
docker run -d --name qdrant-vectordb -p 6333:6333 -v ./qdrant:/qdrant/storage qdrant/qdrant
```

Or download and run [Qdrant locally](https://qdrant.tech/documentation/quick-start/).

## 🚀 Running the Project

### Start the FastAPI Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`
- Docs: `http://localhost:8000/docs` (Swagger UI)
- ReDoc: `http://localhost:8000/redoc`

### (Optional) Start Inngest Event Monitoring

In a new terminal:
```bash
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
```

## 📡 API Usage

### Trigger PDF Ingestion
```bash
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "path/to/document.pdf",
    "source": "document_source"
  }'
```

### Query the RAG System
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main topic of the document?"
  }'
```

## 📁 Project Structure

```
qa-rag-project/
├── main.py                 # FastAPI application & Inngest setup
├── vector_db.py           # Qdrant vector database client
├── data_loader.py         # Document loading & processing
├── custom_types.py        # Type definitions & models
├── pyproject.toml         # Project metadata & dependencies
├── .env                   # Environment variables (not in git)
├── .gitignore            # Ignored files/folders
└── README.md             # This file
```

## 📚 Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `uvicorn` | ASGI server |
| `inngest` | Event-driven workflows |
| `llama-index-core` | Document indexing & RAG |
| `llama-index-readers-file` | File format support |
| `openai` | Embeddings & LLM |
| `qdrant-client` | Vector database |
| `streamlit` | UI framework |
| `python-dotenv` | Environment management |


### Qdrant dashboard
```
http://localhost:6333/dashboard
```