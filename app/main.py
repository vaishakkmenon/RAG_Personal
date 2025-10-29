# main.py — FastAPI RAG Chatbot (cleaned & organized)
# - Keeps endpoints: /health, /ingest, /chat (+ debug routes)
# - Keeps middleware, metrics, CORS
# - Keeps A3 reranker
# - Removes dead code & duplicates; merges duplicate constants; consistent naming/docstrings

import logging
import socket
import re
from typing import Optional, List

import ollama
from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from .middleware.api_key import APIKeyMiddleware
from .middleware.logging import LoggingMiddleware
from .middleware.max_size import MaxSizeMiddleware
from .ingest import ingest_paths
from .query_router import route_query
from .retrieval import search, get_sample_chunks
from .settings import settings

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
_CLIENT = ollama.Client(host=settings.ollama_host, timeout=settings.ollama_timeout)
_MODEL = settings.ollama_model
_NUM_CTX = settings.num_ctx
REQUEST_TIMEOUT_S = settings.ollama_timeout
MAX_BYTES = settings.max_bytes

# ------------------------------------------------------------------------------
# FastAPI app & middleware wiring
# ------------------------------------------------------------------------------
app = FastAPI(
    title=settings.api.title,
    description=settings.api.description,
    version=settings.api.version,
    summary=settings.api.summary,
)

# CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
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

def generate_with_ollama(
    prompt: str, 
    temperature: Optional[float] = None, 
    max_tokens: Optional[int] = None
):
    """Generate text using Ollama with configurable parameters.
    
    Args:
        prompt: The input prompt to generate text from
        temperature: Sampling temperature (None uses default from settings)
        max_tokens: Maximum number of tokens to generate (None uses default from settings)
    """
    temperature = temperature if temperature is not None else settings.retrieval.temperature
    max_tokens = max_tokens if max_tokens is not None else settings.retrieval.max_tokens
    
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

def merge_params(manual, routed, defaults):
    """Merge manual overrides with routed params and defaults"""
    result = defaults.copy()
    result.update(routed)
    result.update({k: v for k, v in manual.items() if v is not None})
    return result

def is_refusal(answer: str, chunks: list) -> bool:
    """Check if answer is a refusal/ungrounded response"""
    if not chunks:
        return True
    
    answer_lower = answer.lower()
    
    # Check explicit refusals
    refusal_patterns = [
        "i don't know", "i do not know", "i couldn't find",
        "there is no mention", "there is no information",
        "not mentioned in", "not mentioned", "not listed",
        "not specified", "not provided", "not included"
    ]
    
    return any(pattern in answer_lower for pattern in refusal_patterns)

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
    # Make all parameters optional with None default
    grounded_only: Optional[bool] = None,
    null_threshold: Optional[float] = None,
    max_distance: Optional[float] = None,
    top_k: Optional[int] = None,
    temperature: Optional[float] = None,
    rerank: Optional[bool] = None,
    rerank_lex_weight: Optional[float] = None,
    doc_type: Optional[str] = None,
    term_id: Optional[str] = None,
    level: Optional[str] = None,
    use_router: bool = True,
    api_key: str = Depends(check_api_key)
):
    # Apply defaults from settings if not provided
    temperature = temperature if temperature is not None else settings.retrieval.temperature
    
    # Default parameters (can be overridden by router or explicitly)
    params = {
        'top_k': top_k if top_k is not None else settings.retrieval.top_k,
        'max_distance': max_distance if max_distance is not None else settings.retrieval.max_distance,
        'null_threshold': null_threshold if null_threshold is not None else settings.retrieval.null_threshold,
        'rerank': rerank if rerank is not None else settings.retrieval.rerank,
        'rerank_lex_weight': rerank_lex_weight if rerank_lex_weight is not None else settings.retrieval.rerank_lex_weight,
        'temperature': temperature,
        'doc_type': doc_type,
        'term_id': term_id,
        'level': level,
    }
    
    logger.info(f"Question: {request.question}")
    
    if use_router:
        routed_params = route_query(request.question)
        
        params = merge_params(
            manual={
                'doc_type': doc_type,
                'term_id': term_id,
                'top_k': top_k,
                'null_threshold': null_threshold,
                'max_distance': max_distance,
                'rerank': rerank,
                'rerank_lex_weight': rerank_lex_weight,
                'temperature': temperature
            },
            routed=routed_params,
            defaults=params
        )
        
        logger.info(f"Routed parameters: doc_type={params.get('doc_type')}, top_k={params.get('top_k')}")
    else:
        logger.info(f"Using default parameters: {params}")
    
    # Build metadata filter
    metadata_filter = {
        k: v for k, v in {
            "doc_type": params.get("doc_type"),
            "term_id": params.get("term_id"),
            "level": level
        }.items() if v is not None
    }
    
    # Retrieve chunks
    try:
        chunks = search(
            query=request.question,
            k=params['top_k'],
            max_distance=params['max_distance'],
            metadata_filter=metadata_filter if metadata_filter else None
        )
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents"
        )
    
    # Post-filter by section prefix if requested
    if use_router and routed_params.get("post_filter_section_prefix"):
        section_prefix = routed_params["post_filter_section_prefix"]
        original_count = len(chunks)
        chunks = [
            c for c in chunks 
            if c.get("metadata", {}).get("section", "").startswith(section_prefix)
        ]
        logger.info(f"Section prefix filter '{section_prefix}': {original_count} → {len(chunks)} chunks")
    
    # Optional reranking
    if params['rerank'] and chunks:
        chunks = rerank_chunks(request.question, chunks, lex_weight=params['rerank_lex_weight'])
        logger.info(f"Reranked {len(chunks)} chunks")
    
    rag_retrieval_chunks.observe(len(chunks))
    
    # Grounding check
    if not chunks:
        logger.warning("No chunks retrieved")
        return ChatResponse(
            answer="I don't know. I couldn't find any relevant information in my documents.",
            sources=[],
            grounded=False
        )
    
    best_distance = chunks[0]["distance"]
    logger.info(f"Best chunk distance: {best_distance:.3f}, threshold: {params['null_threshold']}")
    
    if best_distance > params['null_threshold']:
        logger.info(f"Refusing: best distance {best_distance:.3f} > threshold {params['null_threshold']}")
        return ChatResponse(
            answer="I don't know. I couldn't find sufficiently relevant information in my documents to answer this question confidently.",
            sources=[],
            grounded=False
        )
    
    # Build context
    context = "\n\n".join([
        f"[Source: {c['source']}]\n{c['text']}"
        for c in chunks
    ])
    
    # Generate answer
    # Build base rules
    base_rules = """CRITICAL RULES:
    1. READ values directly - DO NOT calculate or compute.
    2. For course questions, look for tables with format: | Course | Title | Credits | Grade |
    3. If you see a section header (like "## Coursework") without details, CHECK THE NEXT EXCERPTS.
    4. Each excerpt has a [Source: ...] label showing which document it's from - all excerpts from the same source are related.
    5. For "undergraduate/graduate GPA", look for "Overall GPA: X.XX" or "Cumulative GPA: X.XX"
    6. "IMPORTANT: 'Graduate credits' or 'Master's degree' does NOT mean PhD. Only say 'PhD' if explicitly mentioned."
    7. If answer not in excerpts, say "I don't know."
    """

    # Add counting rule only for "how many" questions
    if "how many" in request.question.lower():
        base_rules += "8. Count ALL separate entries with different date ranges.\n"

    prompt = f"""Based on the following excerpts from my personal documents, answer the question concisely and accurately.

{base_rules}

Question: {request.question}

Excerpts:
{context}

Answer (read directly from the excerpts):"""
    
    try:
        answer = generate_with_ollama(
            prompt=prompt,
            temperature=params['temperature'],
            max_tokens=300
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate answer"
        )
    
    logger.info(f"Generated answer: {answer[:100]}...")
    
    answer_stripped = answer.strip()

    if is_refusal(answer_stripped, chunks):
        return ChatResponse(answer=answer_stripped, sources=[], grounded=False)

    # Format sources (this is a grounded answer)
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
        answer=answer_stripped,
        sources=sources,
        grounded=True
    )

# --- Debug routes ---
@app.get("/debug/search")
async def debug_search(
    q: str, 
    k: Optional[int] = None, 
    max_distance: Optional[float] = None
):
    """Debug endpoint for testing search functionality.
    
    Args:
        q: Search query
        k: Number of results to return (defaults to settings.retrieval.top_k)
        max_distance: Maximum cosine distance (defaults to settings.retrieval.max_distance)
    """
    return search(
        q, 
        k=k if k is not None else settings.retrieval.top_k, 
        max_distance=max_distance if max_distance is not None else settings.retrieval.max_distance
    )

@app.get("/debug/samples")
async def debug_samples(n: int = 4):
    """Debug endpoint to retrieve sample chunks from the vector store.
    
    Args:
        n: Number of samples to return (capped at 20)
    """
    return get_sample_chunks(min(n, 20))  # Cap at 20 samples for safety
