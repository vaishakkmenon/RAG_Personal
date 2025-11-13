# Personal RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for querying personal documents including resumes, transcripts, certifications, and project documentation. Built with FastAPI, ChromaDB, Ollama, and SentenceTransformers.

## 🏗️ Architecture

### System Overview
```
┌─────────────────────────────────────────────────┐
│                FastAPI Server                    │
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
                      ┌─────────────┴─────────────┐
                      ▼                           ▼
            ┌─────────────────┐      ┌──────────────────┐
            │   ChatService   │      │ CertHandler      │
            │   (Core Logic)  │      │ (Cert Queries)   │
            └─────────────────┘      └──────────────────┘
                      │
        ┌─────────────┼─────────────┬─────────────┐
        ▼             ▼             ▼             ▼
   ┌────────┐   ┌──────────┐   ┌────────┐   ┌─────────┐
   │ Query  │   │ Retrieval│   │  LLM   │   │Prompt   │
   │ Router │   │ (Chroma) │   │(Ollama)│   │Builder  │
   └────────┘   └──────────┘   └────────┘   └─────────┘
```

### Core Components

- **`app/api/`** - HTTP API layer
  - `routes/` - Individual endpoint modules (health, ingest, chat, debug)
  - `dependencies.py` - Shared dependencies (auth, service factories)

- **`app/core/`** - Business logic layer
  - `chat_service.py` - Main RAG orchestration (~500 lines)
  - `certification_handler.py` - Certification-specific logic (~450 lines)

- **`app/services/`** - External service integrations
  - `llm.py` - Ollama LLM integration
  - `reranker.py` - Hybrid lexical + semantic reranking

- **`app/query_router/`** - Query analysis and routing
  - `router.py` - Main query router
  - `patterns.py` - Pattern matching utilities
  - `route_helpers/` - Query analyzer and response builder

- **`app/retrieval/`** - Vector database operations
  - `store.py` - ChromaDB integration, embeddings, search

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
- **Ollama** for local LLM hosting
- **Docker & Docker Compose** (optional, for containerized deployment)
- **CUDA-capable GPU** (optional, for faster inference)

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
   OLLAMA_HOST=http://127.0.0.1:11434
   OLLAMA_MODEL=llama3.2:3b-instruct-q4_K_M
   EMBED_MODEL=BAAI/bge-small-en-v1.5
   CHROMA_DIR=./data/chroma
   DOCS_DIR=./data/mds
   ```

5. **Start Ollama and pull model**
   ```bash
   ollama serve
   ollama pull llama3.2:3b-instruct-q4_K_M
   ```

6. **Prepare your documents**
   - Place markdown files in `./data/mds/`
   - Use YAML front-matter for metadata:
     ```yaml
     ---
     doc_type: resume
     section: experience
     ---
     # Your content here
     ```

7. **Start the server**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

8. **Ingest documents** (first time only)
   ```bash
   curl -X POST http://localhost:8000/ingest \
     -H "X-API-Key: your-secure-api-key-here"
   ```

9. **Test the system**
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
│   │   └── certification_handler.py  # Cert logic
│   ├── services/                 # External integrations
│   │   ├── llm.py               # Ollama client
│   │   └── reranker.py          # Result reranking
│   ├── query_router/            # Query analysis
│   ├── retrieval/               # Vector database
│   ├── prompting/               # Prompt engineering
│   ├── ingest/                  # Document processing
│   ├── certifications/          # Cert management
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
├── latest_analysis.md           # Latest codebase analysis
└── next_steps.md                # Refactoring action plan
```

## 🔧 Configuration

### Environment Variables

All settings can be configured via `.env` file or environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | `change-me` | API authentication key |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3.2:3b-instruct-q4_K_M` | LLM model name |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `CHROMA_DIR` | `./data/chroma` | Vector database path |
| `DOCS_DIR` | `./data/mds` | Document directory |
| `COLLECTION_NAME` | `personal_rag` | ChromaDB collection |
| `TOP_K` | `5` | Default retrieval count |
| `MAX_DISTANCE` | `0.50` | Max cosine distance |
| `NULL_THRESHOLD` | `0.50` | Grounding threshold |
| `CHUNK_SIZE` | `450` | Characters per chunk |
| `CHUNK_OVERLAP` | `90` | Chunk overlap size |

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

## 🔍 Features

### RAG Pipeline
- ✅ **Semantic Search** - BGE v1.5 embeddings with ChromaDB
- ✅ **Metadata Filtering** - Filter by doc_type, term_id, level, etc.
- ✅ **Hybrid Reranking** - Lexical + semantic similarity
- ✅ **Grounding Checks** - Distance thresholds prevent hallucination
- ✅ **Ambiguity Detection** - Asks for clarification on vague queries
- ✅ **Source Citations** - Returns source documents with answers

### Query Routing
- ✅ **Automatic Query Analysis** - Detects technologies, categories, intents
- ✅ **Certificate Detection** - Recognizes cert names and aliases
- ✅ **Parameter Adjustment** - Tunes retrieval based on question type
- ✅ **Confidence Scoring** - Measures routing confidence

### Document Ingestion
- ✅ **Markdown Processing** - Reads .md and .txt files
- ✅ **YAML Metadata Extraction** - Parses front-matter
- ✅ **Smart Chunking** - Section-aware text splitting
- ✅ **Batch Processing** - Efficient large-scale ingestion
- ✅ **Security Checks** - Path traversal prevention

### LLM Integration
- ✅ **Local Hosting** - Ollama for privacy and cost control
- ✅ **Model Flexibility** - Swap models via config
- ✅ **Streaming Support** - For real-time responses (if needed)
- ✅ **Timeout Handling** - Graceful degradation

### Security
- ✅ **API Key Authentication** - Bearer token required
- ✅ **CORS Configuration** - Restricts cross-origin requests
- ✅ **Request Size Limits** - Prevents DoS attacks
- ✅ **Path Traversal Protection** - Secure file access
- ✅ **Docker Security** - Read-only filesystem, dropped capabilities

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

### ✅ Phase 1 Complete: True RAG Implementation

**Successfully refactored** from hybrid rule-based/RAG system to **pure RAG implementation**!

**What Was Achieved**:
- ✅ Removed **1,116 lines** of hardcoded keyword-based logic
- ✅ Deleted certification registry and handler (no more forced templates)
- ✅ Removed all keyword-based parameter overrides
- ✅ Enabled LLM generation for ALL query types
- ✅ System now uses true retrieval-augmented generation

**Architecture Improvements**:
- ✅ Modular architecture refactoring (970 lines → organized packages)
- ✅ Clean separation of concerns (API, core, services, utilities)
- ✅ ChromaDB vector store integration
- ✅ Ollama LLM integration with streaming
- ✅ Hybrid reranking (lexical + semantic)
- ✅ Query routing with semantic pattern detection
- ✅ Document ingestion pipeline
- ✅ Comprehensive configuration management
- ✅ Docker deployment setup
- ✅ Security hardening (API key, CORS, size limits)
- ✅ Prometheus metrics integration

**Test Results** (2025-11-13):
```
✅ "Do I have CKA?" → Natural LLM response with correct info
✅ "When did I earn CKA and when does it expire?" → Multi-part answer
✅ All queries use semantic search + LLM generation
✅ No hardcoded templates or keyword forcing
```

### 🎯 System Behavior

**Before Refactoring**:
- ❌ Keyword detection ("do i have", "transcript", etc.)
- ❌ Forced response templates
- ❌ Hardcoded parameter overrides
- ❌ Certification registry with duplicated data
- ❌ ~1,116 lines of anti-RAG code

**After Refactoring** (Current):
- ✅ Pure semantic search for all queries
- ✅ LLM generates all responses from context
- ✅ Natural language flexibility
- ✅ Single source of truth (markdown documents)
- ✅ Clean, maintainable codebase

## 🗺️ Roadmap

### Phase 2: Enhanced Semantic Understanding (Next)
- [ ] Improve system prompt for better focused answers
- [ ] Add query-specific context window (include question in context)
- [ ] Embedding-based query classification
- [ ] LLM-powered intent detection for ambiguous queries
- [ ] Dynamic clarification generation based on available data
- [ ] Context-aware parameter tuning

### Phase 3: Advanced Features
- [ ] Multi-hop reasoning for complex queries
- [ ] Conversational context tracking (chat history)
- [ ] Query reformulation for better retrieval
- [ ] Fact verification and grounding scores
- [ ] Comparative analysis (e.g., "compare my AWS and GCP experience")
- [ ] Support for "what if" and hypothetical queries

### Phase 4: Production Readiness
- [ ] Comprehensive test coverage (unit + integration)
- [ ] Benchmark suite and performance testing
- [ ] Rate limiting and quota management
- [ ] Caching layer for common queries
- [ ] Admin dashboard for monitoring
- [ ] A/B testing framework for prompt improvements

## 🤝 Contributing

This is a personal project, but suggestions are welcome! If you notice issues:

1. Check `latest_analysis.md` for current known issues
2. Review `next_steps.md` for planned fixes
3. Open an issue describing the problem and potential solution

## 📝 License

Private project - not licensed for public use.

## 🔗 Resources

- **FastAPI**: https://fastapi.tiangolo.com/
- **ChromaDB**: https://docs.trychroma.com/
- **Ollama**: https://ollama.ai/
- **SentenceTransformers**: https://www.sbert.net/
- **BGE Embeddings**: https://huggingface.co/BAAI/bge-small-en-v1.5

## 📧 Contact

For questions or issues, refer to the documentation or check the analysis files in the repository.

---

**Status**: ✅ **Stable - True RAG Implementation Complete**
**Version**: 0.4.0
**Last Updated**: 2025-11-13
**Phase 1 Refactoring**: COMPLETE (removed 1,116 lines of anti-RAG code)
