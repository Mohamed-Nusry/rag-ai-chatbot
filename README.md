# RAG Chatbot Agent

A Retrieval-Augmented Generation (RAG) chatbot using Qdrant for vector storage and Groq (Llama 3) for the language model.

## Features
- **Document Indexing**: Support for PDF partitioning and text embedding.
- **Multimodal Support**: CLIP-based embedding for figures in PDFs.
- **API Interface**: FastAPI/LangServe implementation for easy integration.

## Setup Instructions

### 1. Prerequisites
- Python 3.10+
- Docker (for Qdrant)
- Groq API Key

### 2. Environment Configuration
Copy the example environment file and fill in your API keys:
```bash
cp .env.example .env
```
Edit `.env` and add your `GROQ_API_KEY`.

### 3. Run Qdrant Database
Use Docker Compose to start the Qdrant service:
```bash
docker-compose up -d
```
Qdrant will be available at `http://localhost:6333`.

### 4. Installation
Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
*(Note: If requirements.txt is missing, installment of `qdrant-client`, `sentence-transformers`, `langchain-groq`, `fastapi`, `uvicorn`, `langserve`, `unstructured[pdf]`, `transformers`, `torch` is required.)*

## Usage

### Indexing Data
To index sample text or a PDF file, run:
```bash
python insert_data.py
```

### Starting the Bot
To start the FastAPI server:
```bash
python server.py
```
The chatbot API will be available at `http://localhost:8000/chatbot`.

## Project Structure
- `main.py`: Core logic for the RAG chain.
- `server.py`: FastAPI server setup.
- `chatbot.py`: Chatbot class for searching and answering.
- `database.py`: Qdrant client connection utility.
- `documents_service.py`: Logic for data insertion and PDF processing.
- `insert_data.py`: CLI script for indexing data.
