# Personal RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for querying personal documents including resumes, transcripts, certifications, and project documentation. Built with FastAPI, ChromaDB, Groq, and SentenceTransformers.

## 🏗️ Architecture

### System Overview
```
┌─────────────────────────────────────────────────┐
│                FastAPI Server                   │
│              (app/main.py - 67 lines)           │
└─────────────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
    ┌────────┐   ┌────────┐   ┌──────────┐
    │ Health │   │ Ingest │   │   Chat   │
    │        │   │        │   │          │
    └────────┘   └────────┘   └──────────┘
                                    │
                                    ▼
                            ┌─────────────────┐
                            │   ChatService   │
                            │   (Core Logic)  │
                            └─────────────────┘
                                     │
           ┌─────────────┬───────────┴────────────┬─────────────┐
           ▼             ▼                        ▼             ▼
      ┌────────┐   ┌──────────┐              ┌────────┐   ┌─────────┐
      │Negative│   │ Retrieval│              │  LLM   │   │Prompt   │
      │Inferenc│   │ (Chroma) │              │ (Groq) │   │Builder  │
      └────────┘   └──────────┘              └────────┘   └─────────┘
```

### Core Components

- **`app/api/`** - HTTP API layer
  - `routes/` - Individual endpoint modules (health, ingest, chat, debug)
  - `dependencies.py` - Shared dependencies (auth, service factories)

- **`app/core/`** - Business logic layer
  - `chat_service.py` - Main RAG orchestration (~500 lines)

- **`app/services/`** - External service integrations
  - `llm.py` - Groq LLM integration
  - `reranker.py` - Hybrid lexical + semantic reranking

- **`app/retrieval/`** - Vector database operations
  - `store.py` - ChromaDB integration, embeddings, search
  - `negative_inference_helper.py` - Missing entity detection
  - `adaptive_threshold.py` - Data-driven threshold calculation

- **`app/prompting/`** - Prompt engineering
  - `builder.py` - Prompt construction and validation
  - `config.py` - Prompt templates and settings
  - `clarification.py` - Ambiguous query handling

- **`app/ingest/`** - Document processing pipeline
  - `processor.py` - Main ingestion orchestrator
  - `discovery.py` - File finding and validation
  - `metadata.py` - YAML front-matter extraction
  - `chunking.py` - Text splitting and section handling

- **`app/certifications/`** - Certification management
  - `registry.py` - Certification metadata registry
  - `models.py` - Certification data models
  - `formatter.py` - Display formatting

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** (tested with 3.13.1)
- **Groq API Key** (free tier available at https://console.groq.com)
- **Docker & Docker Compose** (optional, for containerized deployment)

### Local Development Setup

1. **Clone and navigate to repository**
   ```bash
   cd RAG_Personal
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env .env  # Edit with your settings
   ```

   Key settings in `.env`:
   ```bash
   API_KEY=your-secure-api-key-here
   LLM_PROVIDER=groq
   LLM_GROQ_API_KEY=your-groq-api-key-here
   LLM_GROQ_MODEL=llama-3.1-8b-instant
   EMBED_MODEL=BAAI/bge-small-en-v1.5
   CHROMA_DIR=./data/chroma
   DOCS_DIR=./data/mds
   ```

5. **Prepare your documents**
   - Place markdown files in `./data/mds/`
   - Use YAML front-matter for metadata:
     ```yaml
     ---
     doc_type: resume
     section: experience
     ---
     # Your content here
     ```

6. **Start the server**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

7. **Ingest documents** (first time only)
   ```bash
   curl -X POST http://localhost:8000/ingest \
     -H "X-API-Key: your-secure-api-key-here"
   ```

8. **Test the system**
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your-secure-api-key-here" \
     -d '{"question": "What certifications do I hold?"}'
   ```

### Docker Deployment

1. **Build and start services**
   ```bash
   docker-compose up -d
   ```

2. **Ingest documents**
   ```bash
   docker-compose exec api python -m uvicorn app.main:app
   # Then call /ingest endpoint
   ```

3. **View logs**
   ```bash
   docker-compose logs -f api
   ```

4. **Stop services**
   ```bash
   docker-compose down
   ```

## 📁 Directory Structure

```
RAG_Personal/
├── app/                          # Application code
│   ├── api/                      # API layer
│   │   ├── routes/              # Endpoint handlers
│   │   └── dependencies.py      # Shared dependencies
│   ├── core/                     # Business logic
│   │   ├── chat_service.py      # Main RAG orchestration
│   │   ├── auth.py              # JWT authentication
│   │   └── session_manager.py   # Session management
│   ├── services/                 # External integrations
│   │   ├── llm.py               # Groq client
│   │   └── reranker.py          # Result reranking
│   ├── retrieval/               # Vector database & search
│   │   ├── vector_store.py      # ChromaDB integration
│   │   ├── search_engine.py     # Search orchestration
│   │   ├── ranking.py           # Result ranking
│   │   ├── bm25_search.py       # Hybrid BM25 search
│   │   ├── pattern_matcher.py   # Query pattern matching
│   │   └── query_rewriter.py    # Query rewriting
│   ├── prompting/               # Prompt engineering
│   ├── ingest/                  # Document processing
│   ├── middleware/              # HTTP middleware
│   ├── monitoring/              # Performance tracking
│   ├── main.py                  # FastAPI app setup
│   ├── models.py                # Pydantic models
│   └── settings.py              # Configuration
├── data/
│   ├── mds/                     # Source documents (tracked)
│   ├── chroma/                  # Vector database (gitignored)
│   └── pdfs/                    # Original PDFs (gitignored)
├── tests/                       # Test suite
├── docker-compose.yml           # Container orchestration
├── Dockerfile                   # Container image
├── requirements.txt             # Python dependencies
├── .env                         # Environment config (gitignored)
├── README.md                    # This file
└── PRODUCTION_READINESS_CHECKLIST.md  # Production readiness tracking
```

## 🔧 Configuration

### Environment Variables

All settings can be configured via `.env` file or environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | `change-me` | API authentication key |
| `LLM_PROVIDER` | `groq` | LLM provider (must be groq) |
| `LLM_GROQ_API_KEY` | - | Groq API key (required) |
| `LLM_GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model name |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `CHROMA_DIR` | `./data/chroma` | Vector database path |
| `DOCS_DIR` | `./data/mds` | Document directory |
| `COLLECTION_NAME` | `personal_rag` | ChromaDB collection |
| `TOP_K` | `5` | Default retrieval count |
| `MAX_DISTANCE` | `0.60` | Max cosine distance |
| `NULL_THRESHOLD` | `0.60` | Grounding threshold |
| `CHUNK_SIZE` | `450` | Characters per chunk |
| `CHUNK_OVERLAP` | `90` | Chunk overlap size |
| `RERANK` | `true` | Enable hybrid reranking |
| `NEGATIVE_INFERENCE_THRESHOLD` | `0.37` | Entity existence threshold |
| `NEGATIVE_INFERENCE_METHOD` | `gap_based` | Threshold method |

### Document Metadata

Add YAML front-matter to your markdown files:

```yaml
---
doc_type: resume | certificate | transcript | project
section: experience | education | skills
term_id: spring-2023  # For transcripts
level: undergraduate | graduate  # For transcripts
certification_id: cka  # For certificates
---
```

## 🎯 API Endpoints

### Health Check
```bash
GET /health
```

### Ingest Documents
```bash
POST /ingest
Headers: X-API-Key: your-api-key
```

### Chat (RAG Query)
```bash
POST /chat
Headers:
  Content-Type: application/json
  X-API-Key: your-api-key
Body:
  {
    "question": "What certifications do I hold?",
    "top_k": 5,              # Optional
    "temperature": 0.0,      # Optional
    "doc_type": "certificate" # Optional filter
  }
```

### Debug - Search
```bash
POST /search
Headers:
  Content-Type: application/json
  X-API-Key: your-api-key
Body:
  {
    "question": "kubernetes",
    "k": 10
  }
```

### Debug - Sample Chunks
```bash
GET /sample?n=5
Headers: X-API-Key: your-api-key
```

### Authentication (Admin)
```bash
# Get JWT token
POST /auth/token
Content-Type: application/x-www-form-urlencoded
Body: username=admin&password=your-password

# Access protected endpoints
GET /auth/users/me
Headers: Authorization: Bearer <token>
```

## 🔍 Features

### RAG Pipeline
- ✅ **Semantic Search** - BGE v1.5 embeddings with ChromaDB
- ✅ **Metadata Filtering** - Filter by doc_type, term_id, level, etc.
- ✅ **Hybrid Reranking** - Lexical + semantic similarity
- ✅ **Grounding Checks** - Distance thresholds prevent hallucination
- ✅ **Ambiguity Detection** - Asks for clarification on vague queries
- ✅ **Source Citations** - Returns source documents with answers

### Intelligent Retrieval
- ✅ **Negative Inference Detection** - Automatically detects queries about non-existent entities
- ✅ **Adaptive Thresholding** - Data-driven entity existence detection using gap analysis
- ✅ **Query Reformulation** - Reformulates entity queries to category searches when needed
- ✅ **Context-Aware Processing** - Different thresholds for acronyms, proper nouns, etc.

### Document Ingestion
- ✅ **Markdown Processing** - Reads .md and .txt files
- ✅ **YAML Metadata Extraction** - Parses front-matter
- ✅ **Smart Chunking** - Section-aware text splitting
- ✅ **Batch Processing** - Efficient large-scale ingestion
- ✅ **Security Checks** - Path traversal prevention

### LLM Integration
- ✅ **Groq LLM Integration** - Fast cloud-based inference with Groq API
- ✅ **Model Flexibility** - Swap models via config
- ✅ **Streaming Support** - For real-time responses (if needed)
- ✅ **Timeout Handling** - Graceful degradation

### Security
- ✅ **API Key Authentication** - Bearer token required for chat/ingest
- ✅ **JWT Authentication** - OAuth2 password flow for admin endpoints
- ✅ **Circuit Breaker** - Groq API resilience with automatic recovery
- ✅ **CORS Configuration** - Restricts cross-origin requests
- ✅ **Request Size Limits** - Prevents DoS attacks
- ✅ **Path Traversal Protection** - Secure file access
- ✅ **Docker Security** - Read-only filesystem, dropped capabilities
- ✅ **Graceful Shutdown** - Clean connection cleanup on restart

### Observability
- ✅ **Prometheus Metrics** - Request counts, latencies, chunk retrieval
- ✅ **Structured Logging** - JSON logs with context
- ✅ **Health Checks** - Liveness and readiness probes
- ✅ **Performance Monitoring** - Execution time tracking

## 🧪 Testing

```bash
# Run test suite
python run_tests.py --api-key your-api-key

# Run specific test
python run_tests.py --api-key your-api-key --test health

# Docker testing
docker-compose run test python run_tests.py --api-url http://api:8000
```

## 📊 Current Status

### ✅ Phase 1 & 2 Complete: Robust RAG with Enhanced Understanding

**Successfully matured** from a prototype to a **production-ready service**.

**Key Achievements**:
- ✅ **Pure RAG Architecture**: Zero hardcoded logic; fully semantic retrieval.
- ✅ **Intelligent Retrieval**: Negative Inference + Adaptive Thresholding prevent hallucinations.
- ✅ **Production Engineering**: Structured JSON logging, Prometheus metrics, and Docker containerization.
- ✅ **Security First**: API Key auth, CORS limits, request validation, and Prompt Injection Guardrails.
- ✅ **Test Coverage**: Comprehensive `pytest` suite covering unit, integration, and security scenarios.

**Architecture Improvements**:
- ✅ Modular design (Clear separation of `api`, `core`, `retrieval`, `services`).
- ✅ Consolidated configuration via Pydantic (`app/settings.py`).
- ✅ Hybrid Search (Vector + BM25-style Lexical Reranking).
- ✅ Groq LLM Integration for fast cloud-based inference.

**Known Issues**:
- ⚠️ **Ambiguity Detection**: Currently uses a lightweight rule-based approach to minimize latency and cost. Extremely vague queries may receive generic answers instead of clarification requests.

### 🎯 System Behavior

- **Queries**: Processed via semantic search with hybrid reranking.
- **Missing Entities**: Automatically detected via Negative Inference; queries reformulated to categories.
- **Safety**: Prompt injection attempts are blocked by Llama Prompt Guard.
- **Observability**: Every request is logged with latency, status, and tokens; metrics available at `/metrics`.

## 🗺️ Roadmap

### Phase 3: Advanced Features (Next Focus)
- [x] **Multi-turn Conversation**: Context tracking and history management.
- [ ] **Multi-hop Reasoning**: Handling complex queries requiring synthesis from multiple sources.
- [ ] **Fact Verification**: Post-generation grounding scores and citation checking.
- [ ] **Comparative Analysis**: Supporting "compare X vs Y" style queries.
- [ ] **Admin Dashboard**: Web UI for system monitoring and document management.

### Phase 4: Production Readiness (✅ Completed)
- [x] Comprehensive test coverage (Unit + Integration + Security).
- [x] Benchmark suite and performance testing.
- [x] Security hardening (Auth, CORS, Limits, Guardrails).
- [x] Docker containerization and deployment configuration.
- [x] Observability (Structured Logs + Prometheus Metrics).

## 🤝 Contributing

This is a personal project, but suggestions are welcome!

## 📝 License

Private project - not licensed for public use.

## 🔗 Resources

- **FastAPI**: https://fastapi.tiangolo.com/
- **ChromaDB**: https://docs.trychroma.com/
- **Groq**: https://console.groq.com/
- **SentenceTransformers**: https://www.sbert.net/
- **BGE Embeddings**: https://huggingface.co/BAAI/bge-small-en-v1.5

## 📧 Contact

For questions or issues, refer to the documentation in `docs/`.

---

**Status**: 🟢 **Production Ready (v1.0)**
**Version**: 1.0.0
**Last Updated**: 2025-12-24
**Recent Changes**:
- **JWT Authentication**: Added OAuth2 password flow for admin endpoints
- **Circuit Breaker**: Groq API resilience with automatic state transitions
- **Graceful Shutdown**: Clean connection cleanup on restart
- **CI Improvements**: Test coverage (60% threshold) and Trivy vulnerability scanning
