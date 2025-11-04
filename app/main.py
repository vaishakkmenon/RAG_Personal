# main.py — FastAPI RAG Chatbot (cleaned & organized)
# - Keeps endpoints: /health, /ingest, /chat (+ debug routes)
# - Keeps middleware, metrics, CORS
# - Keeps A3 reranker
# - Removes dead code & duplicates; merges duplicate constants; consistent naming/docstrings

import logging
import socket
import re
from typing import Optional, List, Dict

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
from .certifications import get_registry

from .models import (
    IngestRequest,
    IngestResponse,
    ChatRequest,
    ChatResponse,
    ChatSource,
    AmbiguityMetadata,
)

from .metrics import (
    rag_retrieval_chunks,
    rag_llm_request_total,
    rag_llm_latency_seconds,
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
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )

    return x_api_key


def generate_with_ollama(
    prompt: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
):
    """Generate text using Ollama with configurable parameters.

    Args:
        prompt: The input prompt to generate text from
        temperature: Sampling temperature (None uses default from settings)
        max_tokens: Maximum number of tokens to generate (None uses default from settings)
        model: Optional override for the Ollama model name
    """
    temperature = (
        temperature if temperature is not None else settings.retrieval.temperature
    )
    max_tokens = max_tokens if max_tokens is not None else settings.retrieval.max_tokens
    model_name = model or _MODEL

    import time

    start = time.time()
    try:
        response = _CLIENT.generate(
            model=model_name,
            prompt=prompt,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": _NUM_CTX,
            },
        )
        rag_llm_request_total.labels(status="success", model=model_name).inc()
        return response["response"]
    except Exception as e:
        rag_llm_request_total.labels(status="error", model=model_name).inc()
        raise
    finally:
        duration = time.time() - start
        rag_llm_latency_seconds.labels(status="success", model=model_name).observe(duration)


# ==============================================================================
# Helpers — Overlap scoring & Reranker + sentence/window support
# ==============================================================================

# Tiny stopword set + tokenization for lexical overlap
_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "and",
    "or",
    "if",
    "is",
    "are",
    "was",
    "were",
    "by",
    "with",
    "from",
    "as",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "be",
    "been",
    "being",
    "which",
    "who",
    "whom",
    "what",
    "when",
    "where",
    "why",
    "how",
}
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

def _tokset(s: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(s or "") if w.lower() not in _STOPWORDS}


def rerank_chunks(
    question: str, chunks: List[dict], lex_weight: float = 0.5
) -> List[dict]:
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


from .prompting import create_default_prompt_builder, build_clarification_message

# Initialize prompt builder and certification registry
prompt_builder = create_default_prompt_builder()
cert_registry = get_registry()


def _format_sources(chunks: List[dict]) -> List[Dict[str, str]]:
    """Format chunks for the prompt builder."""
    return [
        {"source": c.get("source", "unknown"), "text": c.get("text", "")}
        for c in chunks
        if c.get("text", "").strip()
    ]


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
    try:
        cert_registry.reload()
    except Exception as exc:  # pragma: no cover - diagnostic logging only
        logger.warning("Failed to reload certification registry: %s", exc)
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
    max_tokens: Optional[int] = None,
    rerank: Optional[bool] = None,
    rerank_lex_weight: Optional[float] = None,
    doc_type: Optional[str] = None,
    term_id: Optional[str] = None,
    level: Optional[str] = None,
    model: Optional[str] = None,
    use_router: bool = True,
    api_key: str = Depends(check_api_key),
):
    # Apply defaults from settings if not provided
    temperature = (
        temperature if temperature is not None else settings.retrieval.temperature
    )
    max_tokens = max_tokens if max_tokens is not None else settings.retrieval.max_tokens
    model_name = model or settings.ollama_model

    # Default parameters (can be overridden by router or explicitly)
    params = {
        "model": model_name,
        "top_k": top_k if top_k is not None else settings.retrieval.top_k,
        "max_distance": (
            max_distance
            if max_distance is not None
            else settings.retrieval.max_distance
        ),
        "null_threshold": (
            null_threshold
            if null_threshold is not None
            else settings.retrieval.null_threshold
        ),
        "rerank": rerank if rerank is not None else settings.retrieval.rerank,
        "rerank_lex_weight": (
            rerank_lex_weight
            if rerank_lex_weight is not None
            else settings.retrieval.rerank_lex_weight
        ),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "doc_type": doc_type,
        "term_id": term_id,
        "level": level,
        "is_ambiguous": False,
        "ambiguity_score": 0.0,
    }

    logger.info(f"Question: {request.question}")

    if use_router:
        routed_params = route_query(request.question)

        params = merge_params(
            manual={
                "doc_type": doc_type,
                "term_id": term_id,
                "top_k": top_k,
                "null_threshold": null_threshold,
                "max_distance": max_distance,
                "rerank": rerank,
                "rerank_lex_weight": rerank_lex_weight,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            routed=routed_params,
            defaults=params,
        )

        logger.info(
            f"Routed parameters: doc_type={params.get('doc_type')}, top_k={params.get('top_k')}"
        )
    else:
        routed_params = {}
        logger.info(f"Using default parameters: {params}")

    if use_router and params.get("is_ambiguous"):
        logger.info(
            "Ambiguous query detected; prompting user for clarification",
            extra={
                "question": request.question,
                "ambiguity_score": params.get("ambiguity_score"),
            },
        )
        clarification = build_clarification_message(request.question, prompt_builder.config)
        return ChatResponse(
            answer=clarification,
            sources=[],
            grounded=False,
            ambiguity=AmbiguityMetadata(
                is_ambiguous=True,
                score=max(params.get("ambiguity_score", 0.0), 0.8),
                clarification_requested=True,
            ),
        )

    heuristic_ambiguous, _ = prompt_builder.is_ambiguous(request.question)
    if heuristic_ambiguous:
        logger.info("Heuristic ambiguity detected; prompting user for clarification")
        params["is_ambiguous"] = True
        clarification = build_clarification_message(request.question, prompt_builder.config)
        return ChatResponse(
            answer=clarification,
            sources=[],
            grounded=False,
            ambiguity=AmbiguityMetadata(
                is_ambiguous=True,
                score=0.9,
                clarification_requested=True,
            ),
        )

    # Build metadata filter
    metadata_filter = {
        k: v
        for k, v in {
            "doc_type": params.get("doc_type"),
            "term_id": params.get("term_id"),
            "level": level,
        }.items()
        if v is not None
    }

    # Retrieve chunks
    try:
        chunks = search(
            query=request.question,
            k=params["top_k"],
            max_distance=params["max_distance"],
            metadata_filter=metadata_filter if metadata_filter else None,
        )
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents",
        )

    # Post-filter by section prefix if requested
    if use_router and routed_params.get("post_filter_section_prefix"):
        section_prefix = routed_params["post_filter_section_prefix"]
        original_count = len(chunks)
        chunks = [
            c
            for c in chunks
            if c.get("metadata", {}).get("section", "").startswith(section_prefix)
        ]
        logger.info(
            f"Section prefix filter '{section_prefix}': {original_count} → {len(chunks)} chunks"
        )

    # Optional reranking
    if params["rerank"] and chunks:
        chunks = rerank_chunks(
            request.question, chunks, lex_weight=params["rerank_lex_weight"]
        )
        logger.info(f"Reranked {len(chunks)} chunks")

    rag_retrieval_chunks.observe(len(chunks))

    # Grounding check
    if not chunks:
        logger.warning("No chunks retrieved")
        return ChatResponse(
            answer="I don't know. I couldn't find any relevant information in my documents.",
            sources=[],
            grounded=False,
            ambiguity=AmbiguityMetadata(
                is_ambiguous=params.get("is_ambiguous", False),
                score=params.get("ambiguity_score", 0.0),
                clarification_requested=False,
            ),
        )

    best_distance = chunks[0]["distance"]
    logger.info(
        f"Best chunk distance: {best_distance:.3f}, threshold: {params['null_threshold']}"
    )

    if best_distance > params["null_threshold"]:
        logger.info(
            f"Refusing: best distance {best_distance:.3f} > threshold {params['null_threshold']}"
        )
        return ChatResponse(
            answer="I don't know. I couldn't find sufficiently relevant information in my documents to answer this question confidently.",
            sources=[],
            grounded=False,
            ambiguity=AmbiguityMetadata(
                is_ambiguous=params.get("is_ambiguous", False),
                score=params.get("ambiguity_score", 0.0),
                clarification_requested=False,
            ),
        )

    # Prepare context for prompt builder
    formatted_chunks = _format_sources(chunks)

    # Build and validate prompt
    prompt_result = prompt_builder.build_prompt(
        question=request.question, context_chunks=formatted_chunks
    )

    # Handle ambiguous questions or missing context
    if prompt_result.status == "ambiguous":
        params["is_ambiguous"] = True
        clarification = prompt_result.message or build_clarification_message(
            request.question, prompt_builder.config
        )
        return ChatResponse(
            answer=clarification,
            sources=[],
            grounded=False,
            ambiguity=AmbiguityMetadata(
                is_ambiguous=True,
                score=max(params.get("ambiguity_score", 0.0), 0.85),
                clarification_requested=True,
            ),
        )

    if prompt_result.status == "no_context":
        logger.info("Prompt builder reported no context; issuing refusal")
        return ChatResponse(
            answer="I don't know. I couldn't find sufficiently relevant information in my documents to answer this question confidently.",
            sources=[],
            grounded=False,
            ambiguity=AmbiguityMetadata(
                is_ambiguous=params.get("is_ambiguous", False),
                score=params.get("ambiguity_score", 0.0),
                clarification_requested=False,
            ),
        )

    # Generate response using the validated prompt
    response_text = generate_with_ollama(
        prompt=prompt_result.prompt,
        temperature=params.get("temperature", 0.1),
        max_tokens=params.get("max_tokens", 1000),
        model=params.get("model"),
    )

    answer = (response_text or "").strip()
    logger.info(f"Generated answer: {answer[:100]}...")
    if prompt_builder.is_refusal(answer):
        return ChatResponse(
            answer=answer,
            sources=[],
            grounded=False,
            ambiguity=AmbiguityMetadata(
                is_ambiguous=params.get("is_ambiguous", False),
                score=params.get("ambiguity_score", 0.0),
                clarification_requested=False,
            ),
        )

    if params.get("is_ambiguous"):
        if prompt_builder.needs_clarification(answer):
            logger.info(
                "Ambiguity flagged but answer lacked clarification; returning clarification prompt"
            )
            clarification = build_clarification_message(
                request.question, prompt_builder.config
            )
            return ChatResponse(
                answer=clarification,
                sources=[],
                grounded=False,
                ambiguity=AmbiguityMetadata(
                    is_ambiguous=True,
                    score=max(params.get("ambiguity_score", 0.0), 0.8),
                    clarification_requested=True,
                ),
            )

    # Format sources (this is a grounded answer)
    sources = [
        ChatSource(
            id=c.get("id", ""),
            source=c.get("source", ""),
            text=(
                c_text[:200] + "..." if len(c_text := c.get("text", "")) > 200 else c_text
            ),
            distance=c.get("distance", 1.0),
        )
        for c in chunks
    ]

    return ChatResponse(
        answer=answer,
        sources=sources,
        grounded=True,
        ambiguity=AmbiguityMetadata(
            is_ambiguous=params.get("is_ambiguous", False),
            score=params.get("ambiguity_score", 0.0),
            clarification_requested=False,
        ),
    )


# --- Debug routes ---
@app.get("/debug/search")
async def debug_search(
    q: str, k: Optional[int] = None, max_distance: Optional[float] = None
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
        max_distance=(
            max_distance
            if max_distance is not None
            else settings.retrieval.max_distance
        ),
    )


@app.get("/debug/samples")
async def debug_samples(n: int = 4):
    """Debug endpoint to retrieve sample chunks from the vector store.

    Args:
        n: Number of samples to return (capped at 20)
    """
    return get_sample_chunks(min(n, 20))  # Cap at 20 samples for safety
