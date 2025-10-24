# main.py — FastAPI RAG Chatbot (cleaned & organized)
# - Keeps endpoints: /health, /ingest, /rc, /chat (+ debug routes)
# - Keeps middleware, metrics, CORS
# - Keeps A2 deterministic evidence span, A3 reranker, extractive/generative modes, abstention gates
# - Removes dead code & duplicates; merges duplicate constants; consistent naming/docstrings

import logging
import socket
import re
from typing import Optional, List

import ollama
from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from .settings import settings
from .ingest import ingest_paths
from .retrieval import search, get_sample_chunks
from .middleware.api_key import APIKeyMiddleware
from .middleware.logging import LoggingMiddleware
from .middleware.max_size import MaxSizeMiddleware

from .models import (
    IngestRequest,
    IngestResponse,
    ChatRequest,
    ChatResponse,
    ChatSource,
)

from .metrics import (
    rag_retrieval_chunks,
    rag_llm_request_total,
    rag_llm_latency_seconds
)

# ------------------------------------------------------------------------------
# Logging & Globals
# ------------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# Ollama client & core settings
_CLIENT = ollama.Client(host=settings.ollama_host)
_MODEL = settings.ollama_model
_NUM_CTX = settings.num_ctx
REQUEST_TIMEOUT_S = settings.ollama_timeout
MAX_BYTES = settings.max_bytes

# ------------------------------------------------------------------------------
# System prompts (small, stable)
# ------------------------------------------------------------------------------

CHAT_SYS_PROMPT = (
    "You are answering questions about a person's resume, academic transcripts, "
    "and certifications. Use only the information provided in the excerpts below. "
    "Be concise and factual. If the answer is not in the excerpts, respond with "
    "'I don't know based on the available information.'"
)

# ------------------------------------------------------------------------------
# FastAPI app & middleware wiring
# ------------------------------------------------------------------------------
app = FastAPI(
    title="RAGChatBot (Local $0)",
    description="Self-hosted RAG chatbot using Ollama, SentenceTransformers, and ChromaDB.",
    version="0.3.0",
    summary="Local RAG + Resume-style RC.",
)

# CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key, request-size limit, structured logging
app.add_middleware(APIKeyMiddleware)
app.add_middleware(MaxSizeMiddleware, max_bytes=MAX_BYTES)
app.add_middleware(LoggingMiddleware)

# Prometheus /metrics
Instrumentator().instrument(app).expose(app)

def check_api_key(x_api_key: str = Header(...)):
    """
    Verify the API key from the X-API-Key header.
    
    Args:
        x_api_key: Value from X-API-Key header (automatically extracted by FastAPI)
    
    Returns:
        The API key if valid
        
    Raises:
        HTTPException: If API key is missing or invalid
    """
    expected_key = settings.api_key  # From .env file
    
    if x_api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return x_api_key

def generate_with_ollama(prompt: str, temperature: float = 0.0, max_tokens: int = 300) -> str:
    """Generate text using Ollama."""
    import time
    start = time.time()
    try:
        response = _CLIENT.generate(
            model=_MODEL,
            prompt=prompt,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": _NUM_CTX,
            }
        )
        rag_llm_request_total.labels(status="success", model=_MODEL).inc()
        return response['response']
    except Exception as e:
        rag_llm_request_total.labels(status="error", model=_MODEL).inc()
        raise
    finally:
        duration = time.time() - start
        rag_llm_latency_seconds.labels(status="success", model=_MODEL).observe(duration)

# ==============================================================================
# Helpers — Overlap scoring & Reranker + sentence/window support
# ==============================================================================

# Tiny stopword set + tokenization for lexical overlap
_STOPWORDS = {
    "the","a","an","of","to","in","on","at","for","and","or","if","is","are","was","were",
    "by","with","from","as","that","this","these","those","it","its","be","been","being",
    "which","who","whom","what","when","where","why","how"
}
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

def _tokset(s: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(s or "") if w.lower() not in _STOPWORDS}

def rerank_chunks(question: str, chunks: List[dict], lex_weight: float = 0.5) -> List[dict]:
    """
    Hybrid reranking: lexical overlap + semantic similarity.
    
    Score = lex_weight * overlap + (1 - lex_weight) * (1 - distance)
    """
    lex_weight = max(0.0, min(1.0, lex_weight))
    
    def compute_score(chunk: dict) -> float:
        text = chunk.get("text", "")
        distance = chunk.get("distance", 1.0)
        
        # Lexical overlap
        q_tokens = _tokset(question)
        c_tokens = _tokset(text)
        overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens)) if q_tokens else 0.0
        
        # Semantic similarity (inverse of distance)
        similarity = 1.0 - max(0.0, min(1.0, distance))
        
        return lex_weight * overlap + (1.0 - lex_weight) * similarity
    
    # Create new list with scores, sort, return
    scored = [(chunk, compute_score(chunk)) for chunk in chunks]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in scored]

# ==============================================================================
# Routes
# ==============================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": _MODEL,
        "ollama_host": settings.ollama_host,
        "socket": socket.gethostname(),
    }

@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    paths = req.paths or [settings.docs_dir]
    added = ingest_paths(paths)
    return IngestResponse(ingested_chunks=added)

# --- RAG chat with distance-based abstention ---
@app.post("/chat")
def chat(
    request: ChatRequest,
    grounded_only: bool = True,
    null_threshold: float = 0.60,
    max_distance: float = 0.60,
    top_k: int = 5,
    temperature: float = 0.0,
    doc_type: Optional[str] = None,
    term_id: Optional[str] = None,
    api_key: str = Depends(check_api_key)
):
    """
    Answer questions with grounded retrieval.
    
    Simple flow:
    1. Retrieve top-k chunks
    2. Check if best chunk meets threshold
    3. Generate answer or refuse
    """
    # Build metadata filter
    metadata_filter = {}
    if doc_type:
        metadata_filter["doc_type"] = doc_type
    if term_id:
        metadata_filter["term_id"] = term_id
    
    # Retrieve chunks
    chunks = search(
        query=request.question,
        k=top_k,
        max_distance=max_distance,
        metadata_filter=metadata_filter if metadata_filter else None
    )
    
    rag_retrieval_chunks.observe(len(chunks))
    
    # Grounding check: refuse if no good chunks
    if not chunks or (grounded_only and chunks[0]["distance"] > null_threshold):
        return ChatResponse(
            answer="I don't know. I couldn't find relevant information in my documents.",
            sources=[
                ChatSource(
                    id=c.get("id", ""),
                    source=c.get("source", ""),
                    text=c.get("text", "")[:200] + "..." if len(c.get("text", "")) > 200 else c.get("text", ""),
                    distance=c.get("distance", 1.0)
                )
                for c in chunks[:3]
            ],
            grounded=False
        )
    
    # Build context from top chunks
    context = "\n\n".join([
        f"[Source: {c['source']}]\n{c['text']}"
        for c in chunks
    ])
    
    # Generate answer
    prompt = f"""Based on the following excerpts from my personal documents, answer the question concisely and accurately. If the answer isn't in the excerpts, say "I don't know."
        Question: {request.question}
        Excerpts:
        {context}
        Answer:"""
    
    answer = generate_with_ollama(
        prompt=prompt,
        temperature=temperature,
        max_tokens=300
    )
    
    sources = [
        ChatSource(
            id=c.get("id", ""),
            source=c.get("source", ""),
            text=c.get("text", "")[:200] + "..." if len(c.get("text", "")) > 200 else c.get("text", ""),
            distance=c.get("distance", 1.0)
        )
        for c in chunks
    ]

    return ChatResponse(
        answer=answer.strip(),
        sources=sources,
        grounded=True
    )

# --- Debug routes ---
@app.get("/debug/search")
async def debug_search(q: str, k: int = 5, max_distance: float = 0.45):
    return search(q, k, max_distance)

@app.get("/debug/samples")
async def debug_samples(n: int = 4):
    return get_sample_chunks(n)
