"""
Application settings for Personal RAG system.

Loads configuration from environment variables (.env file) with sensible defaults.
All settings can be overridden via environment variables.
"""

import os
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Set, Any, Optional
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


class APISettings(BaseModel):
    """API and server settings."""

    title: str = Field(default="RAGChatBot (Local $0)", description="API title")
    description: str = Field(
        default="Self-hosted RAG chatbot using Ollama, SentenceTransformers, and ChromaDB.",
        description="API description",
    )
    version: str = Field(default="0.3.0", description="API version")
    summary: str = Field(
        default="Local RAG + Resume-style RC.", description="API summary"
    )
    cors_origins: List[str] = Field(
        default=["http://localhost:3000"], description="Allowed CORS origins"
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
    @validator("max_file_size")
    def validate_max_file_size(cls, v):
        if v <= 0:
            raise ValueError("Max file size must be positive")
        return v

    @validator("batch_size", "chunk_size", "chunk_overlap")
    def validate_positive_integer(cls, v):
        if v <= 0:
            raise ValueError("Value must be positive")
        return v


class RetrievalSettings(BaseModel):
    """Settings for retrieval and search."""

    top_k: int = Field(
        default=int(os.getenv("TOP_K", "5")),
        description="Number of chunks to send to LLM after reranking"
    )

    rerank_retrieval_k: int = Field(
        default=int(os.getenv("RERANK_RETRIEVAL_K", "50")),
        description="Number of chunks to retrieve when reranking is enabled (before reranking)"
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
        description="Whether to enable reranking of results"
    )

    rerank_lex_weight: float = Field(
        default=float(os.getenv("RERANK_LEX_WEIGHT", "0.6")),
        description="Weight for lexical vs semantic similarity in reranking (0.0-1.0)",
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
    """LLM provider configuration (Groq or Ollama)."""

    # Provider selection
    provider: str = Field(
        default=os.getenv("LLM_PROVIDER", "ollama"),
        description="LLM provider: 'ollama' or 'groq'",
    )

    # Groq settings
    groq_api_key: str = Field(
        default=os.getenv("LLM_GROQ_API_KEY", ""),
        description="Groq API key (required if provider='groq')",
    )
    groq_model: str = Field(
        default=os.getenv("LLM_GROQ_MODEL", "llama-3.1-8b-instant"),
        description="Groq model name",
    )

    # Ollama settings
    ollama_host: str = Field(
        default=os.getenv("LLM_OLLAMA_HOST", os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")),
        description="URL for the Ollama API server",
    )
    ollama_model: str = Field(
        default=os.getenv("LLM_OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "llama3.2:3b-instruct-q4_K_M")),
        description="Ollama model name (with quantization tag)",
    )
    ollama_timeout: int = Field(
        default=int(os.getenv("LLM_OLLAMA_TIMEOUT", os.getenv("OLLAMA_TIMEOUT", "60"))),
        description="Maximum seconds to wait for Ollama responses",
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
        default=int(os.getenv("LLM_NUM_CTX", os.getenv("NUM_CTX", "2048"))),
        description="Context window size for Ollama model (tokens)",
        ge=512,
        le=16384,
    )

    # Validation
    @validator("provider")
    def validate_provider(cls, v):
        if v not in ["ollama", "groq"]:
            raise ValueError("Provider must be 'ollama' or 'groq'")
        return v

    @validator("temperature")
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    @validator("max_tokens")
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
    - LLM: LLM provider settings (Groq or Ollama)
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

    class Config:
        """Pydantic configuration."""

        extra = "forbid"  # Prevent typos in environment variables
        validate_assignment = True  # Validate on attribute assignment


# Global settings instance
settings = Settings()

# Component-specific settings
ingest_settings = settings.ingest
