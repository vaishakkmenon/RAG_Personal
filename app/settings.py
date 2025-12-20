"""
Application settings for Personal RAG system.

Loads configuration from environment variables (.env file) with sensible defaults.
All settings can be overridden via environment variables.
"""

import os
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Dict, List, Set
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


class APISettings(BaseModel):
    """API and server settings."""

    title: str = Field(default="RAGChatBot (Local $0)", description="API title")
    description: str = Field(
        default="Self-hosted RAG chatbot using Groq, SentenceTransformers, and ChromaDB.",
        description="API description",
    )
    version: str = Field(default="0.3.0", description="API version")
    summary: str = Field(
        default="Local RAG + Resume-style RC.", description="API summary"
    )
    cors_origins: List[str] = Field(
        default_factory=lambda: [
            url.strip()
            for url in os.getenv(
                "ALLOWED_ORIGINS",
                "http://localhost:3000,http://localhost:5173,https://vaishakmenon.com",
            ).split(",")
            if url.strip()
        ],
        description="Allowed CORS origins",
    )


class IngestSettings(BaseModel):
    """Configuration for document ingestion."""

    allowed_extensions: Set[str] = Field(
        default_factory=lambda: {
            ext.strip().lower()
            for ext in os.getenv("INGEST_ALLOWED_EXTENSIONS", "txt,md").split(",")
            if ext.strip()
        },
        description="File extensions allowed for ingestion",
    )

    max_file_size: int = Field(
        default=int(os.getenv("INGEST_MAX_FILE_SIZE", str(10 * 1024 * 1024))),
        description="Maximum file size in bytes",  # 10MB
    )

    batch_size: int = Field(
        default=int(os.getenv("INGEST_BATCH_SIZE", "100")),
        description="Number of documents to process in a single batch",
    )

    chunk_size: int = Field(
        default=int(os.getenv("CHUNK_SIZE", "600")),
        description="Number of characters per chunk",
    )

    chunk_overlap: int = Field(
        default=int(os.getenv("CHUNK_OVERLAP", "120")),
        description="Number of characters to overlap between chunks",
    )

    # Validation
    # Validation
    @field_validator("max_file_size")
    def validate_max_file_size(cls, v):
        if v <= 0:
            raise ValueError("Max file size must be positive")
        return v

    @field_validator("batch_size", "chunk_size", "chunk_overlap")
    def validate_positive_integer(cls, v):
        if v <= 0:
            raise ValueError("Value must be positive")
        return v


class RetrievalSettings(BaseModel):
    """Settings for retrieval and search."""

    top_k: int = Field(
        default=int(os.getenv("TOP_K", "5")),
        description="Number of chunks to send to LLM after reranking",
    )

    rerank_retrieval_k: int = Field(
        default=int(os.getenv("RERANK_RETRIEVAL_K", "50")),
        description="Number of chunks to retrieve when reranking is enabled (before reranking)",
    )

    null_threshold: float = Field(
        default=0.60,  # Increased from 0.50 to allow for reranking distance changes
        description="Confidence threshold for considering a response grounded",
    )

    max_distance: float = Field(
        default=0.60,
        description="Maximum cosine distance for retrieval (lower is more similar)",
    )

    rerank: bool = Field(
        default=os.getenv("RERANK", "true").lower() == "true",
        description="Whether to enable reranking of results",
    )

    rerank_lex_weight: float = Field(
        default=float(os.getenv("RERANK_LEX_WEIGHT", "0.6")),
        description="Weight for lexical vs semantic similarity in reranking (0.0-1.0)",
    )


class SessionSettings(BaseModel):
    """Session management configuration."""

    # Storage backend
    storage_backend: str = Field(
        default=os.getenv("SESSION_STORAGE_BACKEND", "redis"),
        description="Session storage backend: 'redis' or 'memory'",
    )
    redis_url: str = Field(
        default=os.getenv("SESSION_REDIS_URL", "redis://localhost:6379/0"),
        description="Redis connection URL",
    )

    # Session limits
    max_total_sessions: int = Field(
        default=int(os.getenv("SESSION_MAX_TOTAL", "1000")),
        description="Maximum total active sessions",
    )
    max_sessions_per_ip: int = Field(
        default=int(os.getenv("SESSION_MAX_PER_IP", "5")),
        description="Maximum sessions per IP address",
    )

    # Rate limiting
    queries_per_hour: int = Field(
        default=int(os.getenv("SESSION_QUERIES_PER_HOUR", "10")),
        description="Maximum queries per session per hour (0 = disabled)",
    )

    # TTL and cleanup
    ttl_seconds: int = Field(
        default=int(os.getenv("SESSION_TTL_SECONDS", "21600")),  # 6 hours
        description="Session TTL in seconds",
    )
    cleanup_interval_seconds: int = Field(
        default=int(os.getenv("SESSION_CLEANUP_INTERVAL", "1800")),  # 30 minutes
        description="Cleanup interval in seconds",
    )

    # History limits (token budget management)
    max_history_tokens: int = Field(
        default=int(os.getenv("SESSION_MAX_HISTORY_TOKENS", "200")),
        description="Maximum tokens to allocate for conversation history",
    )
    max_history_turns: int = Field(
        default=int(os.getenv("SESSION_MAX_HISTORY_TURNS", "5")),
        description="Maximum number of conversation turns to keep",
    )

    # Validation
    # Validation
    @field_validator("storage_backend")
    def validate_storage_backend(cls, v):
        if v not in ["redis", "memory"]:
            raise ValueError("storage_backend must be 'redis' or 'memory'")
        return v

    @field_validator("ttl_seconds", "cleanup_interval_seconds")
    def validate_positive_seconds(cls, v):
        if v <= 0:
            raise ValueError("Value must be positive")
        return v


class QueryRewriterSettings(BaseModel):
    """Configuration for query rewriting system."""

    enabled: bool = Field(
        default=os.getenv("QUERY_REWRITER_ENABLED", "true").lower() == "true",
        description="Enable/disable query rewriting",
    )

    pattern_config_path: str = Field(
        default=os.getenv("QUERY_REWRITER_CONFIG", "config/query_patterns.yaml"),
        description="Path to pattern configuration YAML",
    )

    hot_reload: bool = Field(
        default=os.getenv("QUERY_REWRITER_HOT_RELOAD", "true").lower() == "true",
        description="Enable hot-reloading of pattern config",
    )

    hot_reload_interval_seconds: int = Field(
        default=int(os.getenv("QUERY_REWRITER_RELOAD_INTERVAL", "60")),
        description="Hot-reload check interval (seconds)",
    )

    analytics_enabled: bool = Field(
        default=os.getenv("QUERY_REWRITER_ANALYTICS", "true").lower() == "true",
        description="Enable pattern analytics tracking",
    )

    analytics_storage_path: str = Field(
        default=os.getenv(
            "QUERY_REWRITER_ANALYTICS_PATH", "data/analytics/pattern_effectiveness.json"
        ),
        description="Path to analytics JSON file",
    )

    failed_queries_storage_path: str = Field(
        default=os.getenv(
            "QUERY_REWRITER_FAILED_PATH", "data/analytics/failed_queries.json"
        ),
        description="Path to failed queries JSON file",
    )

    max_latency_ms: float = Field(
        default=float(os.getenv("QUERY_REWRITER_MAX_LATENCY", "10.0")),
        description="Maximum allowed latency for query rewriting (ms)",
    )

    capture_failed_threshold: float = Field(
        default=float(os.getenv("QUERY_REWRITER_FAILED_THRESHOLD", "0.5")),
        description="Distance threshold for capturing failed queries",
    )


class CrossEncoderSettings(BaseModel):
    """Cross-encoder reranking configuration (optimized for production)."""

    enabled: bool = Field(
        default=os.getenv("CROSS_ENCODER_ENABLED", "false").lower() == "true",
        description="Enable cross-encoder neural reranking",
    )

    model: str = Field(
        default=os.getenv(
            "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        description="HuggingFace cross-encoder model",
    )

    cache_dir: str = Field(
        default=os.getenv("CROSS_ENCODER_CACHE_DIR", "/tmp/cross-encoder"),
        description="Directory to cache model files",
    )

    retrieval_k: int = Field(
        default=int(os.getenv("CROSS_ENCODER_RETRIEVAL_K", "15")),
        description="Number of chunks to retrieve before cross-encoder (optimized from 25)",
    )

    top_k: int = Field(
        default=int(os.getenv("CROSS_ENCODER_TOP_K", "5")),
        description="Final number of chunks after cross-encoder reranking",
    )

    max_latency_ms: float = Field(
        default=float(os.getenv("CROSS_ENCODER_MAX_LATENCY_MS", "400.0")),
        description="Maximum acceptable latency (warning threshold)",
    )


class ResponseCacheSettings(BaseModel):
    """Response cache configuration for common queries."""

    enabled: bool = Field(
        default=os.getenv("RESPONSE_CACHE_ENABLED", "true").lower() == "true",
        description="Enable/disable response caching",
    )

    ttl_seconds: int = Field(
        default=int(os.getenv("RESPONSE_CACHE_TTL_SECONDS", "3600")),
        description="Time-to-live for cached responses (seconds)",
    )

    max_cache_size_mb: int = Field(
        default=int(os.getenv("RESPONSE_CACHE_MAX_SIZE_MB", "100")),
        description="Maximum cache size in MB (soft limit)",
    )

    prompt_version: str = Field(
        default=os.getenv("RESPONSE_CACHE_PROMPT_VERSION", "1"),
        description="Prompt version for cache invalidation. Increment when system prompt changes to invalidate old cached responses.",
    )


class BM25Settings(BaseModel):
    """BM25 parameter configuration for keyword search optimization."""

    k1: float = Field(
        default=float(os.getenv("BM25_K1", "1.5")),
        description="Term frequency saturation parameter (typical: 1.2-2.0)",
    )

    b: float = Field(
        default=float(os.getenv("BM25_B", "0.5")),
        description="Document length normalization parameter (typical: 0-1)",
    )

    rrf_k: int = Field(
        default=int(os.getenv("BM25_RRF_K", "60")),
        description="Reciprocal Rank Fusion parameter for combining rankings",
    )


class PromptGuardSettings(BaseModel):
    """Prompt injection guardrail configuration using Llama Prompt Guard 2."""

    enabled: bool = Field(
        default=os.getenv("PROMPT_GUARD_ENABLED", "true").lower() == "true",
        description="Enable/disable prompt injection guard",
    )

    model: str = Field(
        default=os.getenv("PROMPT_GUARD_MODEL", "meta-llama/llama-prompt-guard-2-86m"),
        description="Groq model for prompt guard (86m for accuracy, 22m for speed)",
    )

    fail_open: bool = Field(
        default=os.getenv("PROMPT_GUARD_FAIL_OPEN", "true").lower() == "true",
        description="If True, allow requests when guard errors; if False, block on error",
    )

    timeout_seconds: float = Field(
        default=float(os.getenv("PROMPT_GUARD_TIMEOUT_SECONDS", "3.0")),
        description="Timeout for Groq API calls in seconds",
    )

    max_retries: int = Field(
        default=int(os.getenv("PROMPT_GUARD_MAX_RETRIES", "2")),
        description="Maximum number of retry attempts on transient failures",
    )

    cache_ttl_seconds: int = Field(
        default=int(os.getenv("PROMPT_GUARD_CACHE_TTL_SECONDS", "3600")),
        description="Cache TTL for prompt guard results in seconds",
    )

    cache_max_size: int = Field(
        default=int(os.getenv("PROMPT_GUARD_CACHE_MAX_SIZE", "1000")),
        description="Maximum number of entries in LRU cache",
    )

    blocked_patterns: List[str] = Field(
        default_factory=lambda: [
            # System Leakage
            r"(?i)\b(system\s+prompt|system\s+instruction|prompt\s+instructions)\b",
            r"(?i)\b(ignore\s+previous\s+instructions|reveal\s+your\s+instructions)\b",
            # PII & Financial
            r"(?i)\b(ssn|social\s+security\s+num(ber)?)\b",
            r"(?i)\b(credit\s+card|cc\s+num(ber)?|cvv|cvc)\b",
            r"(?i)\b(api\s+key|private\s+key|secret\s+key)\b",
            r"(?i)\b(passport\s+num(ber)?|driver'?s\s+license)\b",
            # Jailbreaks
            r"(?i)\b(dan\s+mode|do\s+anything\s+now|jailbreak)\b",
            r"(?i)\b(act\s+as\s+an\s+unrestricted)\b",
        ],
        description="Regex patterns to block (case-insensitive, includes PII and leakage)",
    )


class MetadataInjectionSettings(BaseModel):
    """Configuration for metadata injection into LLM context."""

    injection_config: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "certificate": [
                "earned",  # When certification was earned
                "expires",  # Expiration date
                "issuer",  # Issuing organization
                "credential_id",  # Credential ID for verification
            ],
            "resume": ["owner", "last_updated"],  # Resume owner  # When last updated
            "term": [
                "term_id",  # e.g., "spring_2023"
                "level",  # undergraduate/graduate
                "gpa",  # Term GPA
                "credits",  # Credits earned
            ],
        },
        description="Metadata fields to inject per document type",
    )

    date_fields: List[str] = Field(
        default_factory=lambda: [
            "earned",
            "expires",
            "last_updated",
            "start_date",
            "end_date",
        ],
        description="Metadata fields that contain dates",
    )

    enabled: bool = Field(default=True, description="Enable/disable metadata injection")


class LLMSettings(BaseModel):
    """LLM provider configuration (Groq only)."""

    # Provider selection (Groq only)
    provider: str = Field(
        default=os.getenv("LLM_PROVIDER", "groq"),
        description="LLM provider: must be 'groq'",
    )

    # Groq settings
    groq_api_key: str = Field(
        default=os.getenv("LLM_GROQ_API_KEY", ""),
        description="Groq API key (required)",
    )
    groq_model: str = Field(
        default=os.getenv("LLM_GROQ_MODEL", "llama-3.1-8b-instant"),
        description="Groq model name",
    )
    groq_tier: str = Field(
        default=os.getenv("LLM_GROQ_TIER", "free"),
        description="Groq API tier: 'free', 'developer', or 'enterprise'",
    )
    groq_requests_per_minute: int = Field(
        default=int(os.getenv("LLM_GROQ_REQUESTS_PER_MINUTE", "28")),
        description="Rate limit: requests per minute (based on Groq tier)",
    )
    groq_requests_per_day: int = Field(
        default=int(os.getenv("LLM_GROQ_REQUESTS_PER_DAY", "13680")),
        description="Rate limit: requests per day (based on Groq tier)",
    )

    # Shared generation settings
    temperature: float = Field(
        default=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        description="Sampling temperature for generation (0.0-2.0)",
    )
    max_tokens: int = Field(
        default=int(os.getenv("LLM_MAX_TOKENS", "1000")),
        description="Maximum number of tokens to generate",
    )
    num_ctx: int = Field(
        default=int(os.getenv("LLM_NUM_CTX", "2048")),
        description="Context window size (tokens)",
        ge=512,
        le=16384,
    )

    # Validation
    @field_validator("provider")
    def validate_provider(cls, v):
        if v != "groq":
            raise ValueError(
                "Provider must be 'groq' - Ollama is no longer supported in production"
            )
        return v

    @field_validator("temperature")
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    @field_validator("max_tokens")
    def validate_max_tokens(cls, v):
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v


class Settings(BaseModel):
    """Global application configuration.

    Configuration priority:
    1. Environment variables (.env file or system)
    2. Defaults specified below

    Categories:
    - LLM: LLM provider settings (Groq only)
    - Embeddings: Sentence transformer model
    - Storage: ChromaDB and document paths
    - Retrieval: Search and chunking parameters
    - Security: API authentication
    - Query Routing: Query analysis and routing
    - Ingest: Document ingestion settings
    - API: FastAPI and server settings
    """

    # LLM Settings (new structure)
    llm: LLMSettings = Field(
        default_factory=LLMSettings,
        description="LLM provider configuration",
    )

    # Embeddings
    embed_model: str = Field(
        default=os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        description="SentenceTransformer model for embeddings",
    )

    # Storage
    chroma_dir: str = Field(
        default=os.getenv("CHROMA_DIR", "./data/chroma"),
        description="ChromaDB persistent storage directory",
    )
    docs_dir: str = Field(
        default=os.getenv("DOCS_DIR", "./data/mds"),
        description="Directory containing documents to ingest (resume, transcripts, certs)",
    )
    collection_name: str = Field(
        default=os.getenv("COLLECTION_NAME", "personal_rag"),
        description="ChromaDB collection name",
    )

    # Retrieval
    top_k: int = Field(
        default=int(os.getenv("TOP_K", "5")),
        description="Default number of chunks to retrieve",
    )
    max_distance: float = Field(
        default=float(os.getenv("MAX_DISTANCE", "0.60")),
        description="Maximum cosine distance for retrieval (0-2, lower = more similar)",
    )
    null_threshold: float = Field(
        default=float(os.getenv("NULL_THRESHOLD", "0.50")),
        description="Distance threshold for grounding check (refuse if distance > threshold)",
    )
    chunk_size: int = Field(
        default=int(os.getenv("CHUNK_SIZE", "450")),
        description="Target chunk size in characters",
    )
    chunk_overlap: int = Field(
        default=int(os.getenv("CHUNK_OVERLAP", "90")),
        description="Overlap between consecutive chunks in characters",
    )

    # Security
    api_key: str = Field(
        default=os.getenv("API_KEY", "change-me"),
        description="API key for authentication (change in production!)",
    )
    max_bytes: int = Field(
        default=int(os.getenv("MAX_BYTES", "32768")),
        description="Maximum HTTP request body size in bytes",
    )

    # Component-specific settings
    api: APISettings = Field(
        default_factory=APISettings, description="API and server configuration"
    )
    retrieval: RetrievalSettings = Field(
        default_factory=RetrievalSettings,
        description="Retrieval and search configuration",
    )
    ingest: IngestSettings = Field(
        default_factory=IngestSettings, description="Document ingestion configuration"
    )

    metadata_injection: MetadataInjectionSettings = Field(
        default_factory=MetadataInjectionSettings,
        description="Metadata injection configuration",
    )

    session: SessionSettings = Field(
        default_factory=SessionSettings, description="Session management configuration"
    )

    query_rewriter: QueryRewriterSettings = Field(
        default_factory=QueryRewriterSettings,
        description="Query rewriting configuration",
    )

    cross_encoder: CrossEncoderSettings = Field(
        default_factory=CrossEncoderSettings,
        description="Cross-encoder reranking configuration",
    )

    response_cache: ResponseCacheSettings = Field(
        default_factory=ResponseCacheSettings,
        description="Response cache configuration",
    )

    bm25: BM25Settings = Field(
        default_factory=BM25Settings, description="BM25 parameter configuration"
    )

    prompt_guard: PromptGuardSettings = Field(
        default_factory=PromptGuardSettings,
        description="Prompt injection guardrail configuration",
    )

    model_config = ConfigDict(
        extra="forbid",  # Prevent typos in environment variables
        validate_assignment=True,  # Validate on attribute assignment
    )


# Global settings instance
settings = Settings()

# Component-specific settings
ingest_settings = settings.ingest
