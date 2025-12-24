# app/metrics.py
"""
Comprehensive Prometheus metrics for RAG system monitoring.

Metrics are organized by component:
- Ingestion: Document processing and chunking
- Retrieval: Vector search and ranking
- LLM: Language model requests
- Prompt Guard: Security and injection detection
- Sessions: User session management
- Query Rewriting: Pattern matching and transformations
- Ambiguity: Question clarity detection
- Errors: System-wide error tracking
"""

from prometheus_client import Counter, Histogram, Gauge

# ============================================================================
# INGESTION METRICS
# ============================================================================

rag_ingested_chunks_total = Counter(
    "rag_ingested_chunks_total", "Total number of document chunks successfully ingested"
)

rag_ingest_skipped_files_total = Counter(
    "rag_ingest_skipped_files_total",
    "Number of files skipped during ingestion",
    ["reason"],  # label: e.g. "too_large", "invalid_ext", "outside_docs_dir"
)

# ============================================================================
# RETRIEVAL METRICS
# ============================================================================

rag_retrieval_chunks = Histogram(
    "rag_retrieval_chunks",
    "Number of chunks retrieved per query",
    buckets=[0, 1, 2, 4, 8, 16, 32, 50],
)

rag_retrieval_distance = Histogram(
    "rag_retrieval_distance",
    "Cosine distance of retrieved chunks (lower is more similar)",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

rag_grounding_total = Counter(
    "rag_grounding_total",
    "Total queries by grounding result",
    ["grounded"],  # "true" or "false"
)

rag_retrieval_latency_seconds = Histogram(
    "rag_retrieval_latency_seconds",
    "Time spent on retrieval operations",
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5),
)

# ============================================================================
# RERANKING METRICS
# ============================================================================

rag_rerank_total = Counter(
    "rag_rerank_total",
    "Total reranking operations",
    ["method"],  # "bm25", "cross_encoder"
)

rag_rerank_latency_seconds = Histogram(
    "rag_rerank_latency_seconds",
    "Time spent on reranking",
    ["method"],  # "bm25", "cross_encoder"
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1, 2),
)

rag_rerank_score_distribution = Histogram(
    "rag_rerank_score_distribution",
    "Distribution of reranking scores",
    ["method"],  # "bm25", "cross_encoder"
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ============================================================================
# LLM METRICS
# ============================================================================

rag_llm_request_total = Counter(
    "rag_llm_request_total",
    "Total LLM requests",
    ["status", "model"],  # status: "success", "error"; model: "groq:llama-3.1-8b", etc.
)

rag_llm_latency_seconds = Histogram(
    "rag_llm_latency_seconds",
    "LLM request latency",
    ["status", "model"],
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60),
)

rag_llm_token_usage_total = Counter(
    "rag_llm_token_usage_total",
    "Total tokens used by LLM",
    ["type", "model"],  # type: "input", "output"
)

rag_llm_cost_total = Counter(
    "rag_llm_cost_total",
    "Estimated cost of LLM usage in USD",
    ["model"],
)

# ============================================================================
# CIRCUIT BREAKER METRICS
# ============================================================================

rag_circuit_breaker_state = Gauge(
    "rag_circuit_breaker_state",
    "Circuit breaker state (0=closed, 0.5=half_open, 1=open)",
    ["name"],  # e.g., "groq_api"
)

rag_circuit_breaker_transitions_total = Counter(
    "rag_circuit_breaker_transitions_total",
    "Total circuit breaker state transitions",
    ["name", "from_state", "to_state"],  # e.g., "groq_api", "closed", "open"
)

# ============================================================================
# PROMPT GUARD METRICS (Security)
# ============================================================================

prompt_guard_checks_total = Counter(
    "prompt_guard_checks_total",
    "Total prompt guard checks",
    ["result"],  # "safe", "blocked"
)

prompt_guard_blocked_total = Counter(
    "prompt_guard_blocked_total",
    "Total blocked requests by classification",
    ["label"],  # "malicious", "LABEL_1", etc.
)

prompt_guard_api_latency_seconds = Histogram(
    "prompt_guard_api_latency_seconds",
    "Groq API call latency for prompt guard",
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5),
)

prompt_guard_cache_operations_total = Counter(
    "prompt_guard_cache_operations_total",
    "Prompt guard cache operations",
    ["result"],  # "hit", "miss"
)

prompt_guard_errors_total = Counter(
    "prompt_guard_errors_total",
    "Prompt guard errors",
    ["error_type"],  # "timeout", "api_error", "parse_error"
)

prompt_guard_retries_total = Counter(
    "prompt_guard_retries_total",
    "Number of retry attempts",
    ["attempt"],  # "1", "2", "3"
)

prompt_guard_context_size_chars = Histogram(
    "prompt_guard_context_size_chars",
    "Size of input checked (with conversation context)",
    buckets=[0, 50, 100, 200, 500, 1000, 2000, 5000],
)

# ============================================================================
# SESSION MANAGEMENT METRICS
# ============================================================================

rag_sessions_active = Gauge(
    "rag_sessions_active", "Number of currently active sessions"
)

rag_session_operations_total = Counter(
    "rag_session_operations_total",
    "Session operations",
    ["operation"],  # "created", "retrieved", "updated", "expired", "deleted"
)

rag_rate_limit_violations_total = Counter(
    "rag_rate_limit_violations_total",
    "Rate limit violations",
    ["limit_type"],  # "queries_per_hour", "max_sessions_per_ip"
)

rag_session_query_count = Histogram(
    "rag_session_query_count",
    "Number of queries per session",
    buckets=[1, 2, 5, 10, 20, 50, 100],
)

rag_session_duration_seconds = Histogram(
    "rag_session_duration_seconds",
    "Session duration from creation to last activity",
    buckets=[60, 300, 600, 1800, 3600, 7200, 21600],  # 1m, 5m, 10m, 30m, 1h, 2h, 6h
)

# ============================================================================
# QUERY REWRITING METRICS
# ============================================================================

rag_query_rewrite_total = Counter(
    "rag_query_rewrite_total",
    "Query rewriting operations",
    ["matched"],  # "true" (pattern matched), "false" (no match)
)

rag_query_rewrite_pattern_matches_total = Counter(
    "rag_query_rewrite_pattern_matches_total",
    "Pattern matches by pattern name",
    ["pattern_name"],  # e.g., "gpa_query", "course_lookup"
)

rag_query_rewrite_latency_seconds = Histogram(
    "rag_query_rewrite_latency_seconds",
    "Time spent on query rewriting",
    buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1),
)

# ============================================================================
# AMBIGUITY DETECTION METRICS
# ============================================================================

rag_ambiguity_checks_total = Counter(
    "rag_ambiguity_checks_total",
    "Ambiguity detection checks",
    ["result"],  # "ambiguous", "clear"
)

rag_clarification_requests_total = Counter(
    "rag_clarification_requests_total", "Number of clarification requests sent to users"
)

rag_ambiguity_score = Histogram(
    "rag_ambiguity_score",
    "Ambiguity score distribution",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ============================================================================
# ERROR TRACKING METRICS
# ============================================================================

rag_errors_total = Counter(
    "rag_errors_total",
    "System-wide errors by component and type",
    ["component", "error_type"],  # component: "retrieval", "llm", "cache", etc.
)

rag_fallback_operations_total = Counter(
    "rag_fallback_operations_total",
    "Fallback operations when primary fails",
    ["from_service", "to_service"],  # e.g., from_service="groq", to_service="ollama"
)

# ============================================================================
# FEEDBACK METRICS
# ============================================================================

rag_feedback_total = Counter(
    "rag_feedback_total",
    "Total user feedback received",
    ["thumbs_up"],  # "true", "false"
)

# ============================================================================
# END-TO-END REQUEST METRICS
# ============================================================================

rag_request_total = Counter(
    "rag_request_total",
    "Total RAG requests",
    [
        "endpoint",
        "status",
    ],  # endpoint: "/chat", "/chat/simple"; status: "success", "error"
)

rag_request_latency_seconds = Histogram(
    "rag_request_latency_seconds",
    "End-to-end request latency",
    ["endpoint"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60),
)

# ============================================================================
# EXPORT ALL METRICS
# ============================================================================

__all__ = [
    # Ingestion
    "rag_ingested_chunks_total",
    "rag_ingest_skipped_files_total",
    # Retrieval
    "rag_retrieval_chunks",
    "rag_retrieval_distance",
    "rag_grounding_total",
    "rag_retrieval_latency_seconds",
    # Reranking
    "rag_rerank_total",
    "rag_rerank_latency_seconds",
    "rag_rerank_score_distribution",
    # LLM
    "rag_llm_request_total",
    "rag_llm_latency_seconds",
    "rag_llm_token_usage_total",
    "rag_llm_cost_total",
    # Circuit Breaker
    "rag_circuit_breaker_state",
    "rag_circuit_breaker_transitions_total",
    # Prompt Guard
    "prompt_guard_checks_total",
    "prompt_guard_blocked_total",
    "prompt_guard_api_latency_seconds",
    "prompt_guard_cache_operations_total",
    "prompt_guard_errors_total",
    "prompt_guard_retries_total",
    "prompt_guard_context_size_chars",
    # Sessions
    "rag_sessions_active",
    "rag_session_operations_total",
    "rag_rate_limit_violations_total",
    "rag_session_query_count",
    "rag_session_duration_seconds",
    # Query Rewriting
    "rag_query_rewrite_total",
    "rag_query_rewrite_pattern_matches_total",
    "rag_query_rewrite_latency_seconds",
    # Ambiguity
    "rag_ambiguity_checks_total",
    "rag_clarification_requests_total",
    "rag_ambiguity_score",
    # Errors
    "rag_errors_total",
    "rag_fallback_operations_total",
    # End-to-end
    "rag_request_total",
    "rag_request_latency_seconds",
    "rag_feedback_total",
]
