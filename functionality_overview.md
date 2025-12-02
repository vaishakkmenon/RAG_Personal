# Descriptive Functionality Document

This document provides a descriptive overview of the Personal RAG System, explaining its functionality, how it works, and its purpose.

## 🎯 Purpose

The Personal RAG (Retrieval-Augmented Generation) System is designed to act as an intelligent knowledge assistant for personal documents. Its primary purpose is to allow users to query their own data—such as resumes, transcripts, certifications, and project documentation—using natural language.

Unlike standard keyword search, this system understands the *semantic meaning* of queries, allowing it to answer complex questions like "What certifications do I have?" or "Summarize my experience with Kubernetes" by synthesizing information from multiple documents.

## 🧠 Core Functionality

### 1. Retrieval-Augmented Generation (RAG)
The core of the system is the RAG pipeline. When a user asks a question, the system doesn't just ask the LLM to hallucinate an answer. Instead, it:
1.  **Retrieves** relevant excerpts from your personal documents.
2.  **Augments** the LLM's prompt with these excerpts as context.
3.  **Generates** a factual answer based *only* on that context.

### 2. Semantic Search & Hybrid Reranking
-   **Semantic Search**: Uses vector embeddings (via `sentence-transformers` and `ChromaDB`) to find text that matches the *meaning* of the query, not just exact keywords.
-   **Hybrid Reranking**: Combines semantic similarity scores with lexical (keyword) matching to ensure the most relevant results are prioritized.

### 3. Intelligent Retrieval Features
The system uses data-driven techniques to improve retrieval quality:
-   **Negative Inference Detection**: Automatically detects when a user asks about entities that don't exist (e.g., "Do I have a PhD?") and reformulates the query to search for comprehensive lists instead.
-   **Adaptive Thresholding**: Uses statistical analysis of distance distributions to dynamically determine if entities exist in the knowledge base, rather than relying on fixed thresholds.
-   **Ambiguity Detection**: If a query is too vague (e.g., single-word questions), the system asks for clarification instead of guessing.

### 4. Document Ingestion Pipeline
The system includes a robust pipeline to process raw files into searchable knowledge:
-   **Markdown Support**: Natively understands Markdown structure.
-   **Metadata Extraction**: Parses YAML front-matter (e.g., `doc_type: resume`) to enable powerful filtering.
-   **Smart Chunking**: Splits long documents into smaller, meaningful "chunks" while preserving context (e.g., keeping headers with their content).

### 5. High-Speed Inference & Local Testing
-   **Primary (Groq)**: The system primarily uses **Groq** for lightning-fast inference in production, ensuring near-instant responses to user queries.
-   **Testing (Ollama)**: For development and testing purposes, the system supports **Ollama** to run models locally, allowing for offline development and privacy-focused experiments.

## ⚙️ How It Works (Workflow)

### Step 1: Ingestion (The "Learning" Phase)
1.  **Input**: You place Markdown files in the `data/mds/` directory.
2.  **Processing**: The system reads these files, extracts metadata, and splits the text into chunks.
3.  **Embedding**: Each chunk is converted into a numerical vector (embedding) that represents its meaning.
4.  **Storage**: These vectors are stored in **ChromaDB**.

### Step 2: Querying (The "Asking" Phase)
1.  **User Question**: You send a question to the `/chat` endpoint (e.g., "Do I have a CKA cert?").
2.  **Entity Analysis**: The negative inference detector extracts potential entities ("CKA") and checks if they exist in the knowledge base using adaptive thresholding.
3.  **Query Reformulation**: If the entity doesn't exist, the system automatically searches for the category ("certifications") to provide a complete answer.
4.  **Retrieval**: The system searches ChromaDB for semantically relevant chunks using BGE v1.5 embeddings.
5.  **Reranking**: The top results are refined using hybrid semantic + lexical scoring to ensure the best matches are at the top.
6.  **Generation**: The system constructs a prompt for the LLM:
    > "Context: [Content of retrieved chunks]
    > Question: Do I have a CKA cert?
    > Answer the question using only the context above."
7.  **Response**: The LLM generates the answer, which is sent back to you.

## 🛡️ Security & Observability

-   **API Key Authentication**: Protects the API from unauthorized access.
-   **Input Validation**: Prevents malicious inputs and path traversal attacks.
-   **Prometheus Metrics**: Tracks performance (latency, request counts) to ensure the system is running smoothly.
-   **Docker Isolation**: Runs in containers with limited privileges for enhanced security.

## 📡 API Endpoints & Usage

The system exposes a RESTful API via FastAPI. Below are the available endpoints and how to access them.

### 1. Chat (Main Interface)

**Endpoint**: `POST /chat`

**Description**: The primary endpoint for asking questions. It handles negative inference detection, retrieval with reranking, and LLM generation.

**Usage**:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "question": "What certifications do I hold?",
    "top_k": 5,
    "doc_type": "certificate"
  }'
```

**Parameters**:
- `question` (required): The user's query.
- `top_k` (optional): Number of document chunks to retrieve.
- `doc_type` (optional): Filter by document type (e.g., `resume`, `certificate`).
- `temperature` (optional): LLM creativity (0.0 - 1.0).

### 2. Simple Chat (Testing)

**Endpoint**: `POST /chat/simple`

**Description**: A bare-bones RAG endpoint without negative inference detection or metadata filtering. Useful for testing raw retrieval performance.

**Usage**:
```bash
curl -X POST http://localhost:8000/chat/simple \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "question": "Kubernetes experience"
  }'
```

### 3. Ingestion

**Endpoint**: `POST /ingest`

**Description**: Triggers the document ingestion pipeline to process Markdown files and update the vector database.

**Usage**:
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "paths": ["./data/mds"]
  }'
```

### 4. Health Check

**Endpoint**: `GET /health`

**Description**: Checks if the system is running and returns basic status info.

**Usage**:
```bash
curl http://localhost:8000/health
```

### 5. Debugging

**Search**: `GET /search?q=query&k=5`
- Returns raw search results from ChromaDB without LLM generation.

**Samples**: `GET /samples?n=5`
- Returns random chunks from the database to verify data integrity.
