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


class QueryRouterSettings(BaseModel):
    """Configuration for the query routing system."""

    # General settings
    cumulative_query_boost: int = Field(
        default=int(os.getenv("QUERY_ROUTER_CUMULATIVE_BOOST", "2")),
        description="Boost factor for cumulative queries",
    )

    # Technology terms and patterns
    technology_terms: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "kubernetes": ["kubernetes", "k8s", "kube"],
            "docker": ["docker", "containers"],
            "aws": ["aws", "amazon web services"],
            "gcp": ["gcp", "google cloud", "google cloud platform"],
            "python": ["python", "python3", "python 3"],
            "terraform": ["terraform", "iac", "infrastructure as code"],
            "kubernetes_administration": ["cka", "certified kubernetes administrator"],
            "aws_cloud": ["aws cloud", "amazon cloud"],
        },
        description="Technology terms and their common aliases",
    )

    # Categories and their related terms
    categories: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "technical_skills": ["technical", "skills", "programming", "coding"],
            "cloud_platforms": ["cloud", "aws", "gcp", "azure"],
            "programming_languages": [
                "python",
                "javascript",
                "java",
                "go",
                "typescript",
            ],
            "certifications": ["certification", "certificate", "certified"],
            "devops": ["devops", "ci/cd", "continuous", "deployment"],
            "data_science": [
                "data science",
                "machine learning",
                "ml",
                "ai",
                "pandas",
                "numpy",
            ],
            "cloud_certifications": [
                "aws certified",
                "azure certified",
                "gcp certified",
            ],
        },
        description="Categories and their related terms",
    )

    # Question type patterns
    question_patterns: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "do_i_have": [r"do\s+i\s+have\s+(a\s+)?(.*?)(\?|$)"],
            "what_do_i_know": [r"what\s+(.*?)\s+do\s+i\s+(know|have)\??"],
            "list_my": [r"list\s+(my\s+)?(.*?)(\?|$)"],
            "which": [r"which\s+(.*?)\s+(do|have)\s+i\s+(.*?)\??"],
            "compare": [r"compare\s+my\s+(.*?)\s+to\s+(.*?)\??"],
            "what_is_my": [r"what('s| is) my (.*?)\??"],
            "how_much": [r"how (much|many) (.*?) (do|have) i"],
        },
        description="Patterns for different question types",
    )

    ambiguity_threshold: float = Field(
        default=float(os.getenv("QUERY_ROUTER_AMBIGUITY_THRESHOLD", "0.7")),
        description="Confidence score threshold above which a query is considered ambiguous (0-1)",
    )

    short_query_word_limit: int = Field(
        default=int(os.getenv("QUERY_ROUTER_SHORT_QUERY_WORD_LIMIT", "3")),
        description="Word count at or below which a query is treated as short",
    )

    single_word_char_limit: int = Field(
        default=int(os.getenv("QUERY_ROUTER_SINGLE_WORD_CHAR_LIMIT", "15")),
        description="Maximum character length for a single-word query to be treated as highly ambiguous",
    )

    ambiguous_keywords: List[str] = Field(
        default_factory=lambda: [
            "experience",
            "background",
            "qualifications",
            "overview",
            "summary",
            "about",
        ],
        description="Keywords that increase ambiguity confidence when present in the query",
    )

    # Document type patterns
    doc_type_patterns: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "certificate": [
                r"\bcertificat(e|ion)s?\b",
                r"\bcertified\b",
                r"\bcka\b",
                r"\baws\b.*\b(ccp|practitioner)\b",
                r"\bprofessional\s+credential",
            ],
            "resume": [
                r"\b(work|job|employment|position)\s+(experience|history)\b",
                r"\bcompan(y|ies)\b.*\bworked\b",
                r"\bintern(ship|ed)?\b",
                r"\b(current|previous|recent)\s+(role|job|position)\b",
                r"\bskills?\b",
                r"\bpersonal\s+project",
                r"\bproject",
                r"\bcompan(y|ies)\b",
                r"\bmaven\s+wave\b",
            ],
            "transcript_analysis": [
                r"\b(overall|cumulative|total|complete)\b.*\b(gpa|credit|grade|academic)\b",
                r"\b(undergraduate|graduate)\b.*\b(gpa|cumulative)\b",
                r"\bhow\s+many\s+(total\s+)?credits?\b",
                r"\btotal\s+credits?\s+(earned|completed)\b",
                r"\bacademic\s+(summary|overview)\b",
                r"\bwhat\s+degrees?\s+(did|have)\b",
                r"\bgraduation\s+date\b",
                r"\bwhen\s+did.*graduate\b",
                r"\bsumma\s+cum\s+laude\b",
            ],
            "term": [
                r"\bterm\s+gpa\b",
                r"\bsemester\s+gpa\b",
                r"\bacademic\s+(record|standing)\b",
                r"\bhonors?\b",
                r"\bgpa\b",
                r"\bgrade\s+point\s+average\b",
                r"\bcredit(s)?\b.*\b(earned|completed)\b",
                r"\bcourse(s)?\b",
                r"\bclass(es)?\b",
                r"\b(cs|ee|ma|ph|eh)\s*\d{3}",
                r"\b(spring|fall|summer|winter)\s+\d{4}\b",
                r"\bsemester",
                r"\bstudy|studied|took",
            ],
        },
        description="Regex patterns for document type detection",
    )

    # Question patterns
    broad_question_patterns: List[str] = Field(
        default_factory=lambda: [
            r"\bsummar(y|ize|ization)\b",
            r"\boverall\b",
            r"\beverything\b",
            r"\ball\b.*\b(about|my)\b",
            r"\bbackground\b",
            r"\bprofile\b",
            r"\bqualifications?\b",
        ],
        description="Patterns that indicate a broad question",
    )

    specific_question_patterns: List[str] = Field(
        default_factory=lambda: [
            r"\bwhat\s+was\b",
            r"\bwhen\s+did\b",
            r"\bwhere\b",
            r"\bhow\s+many\b",
            r"\bwhich\b",
            r"\blist\b",
            r"\bwhat\s+is\b",
        ],
        description="Patterns that indicate a specific question",
    )

    cumulative_patterns: List[str] = Field(
        default_factory=lambda: [
            r"\b(overall|cumulative|total|complete|entire)\b",
            r"\b(undergraduate|graduate)\b.*\bgpa\b",
            r"\bhow\s+many\s+total\b",
            r"\bacademic\s+(summary|overview|performance)\b",
        ],
        description="Patterns that indicate a cumulative query",
    )

    # Retrieval parameters
    default_top_k: int = Field(
        default=int(os.getenv("DEFAULT_TOP_K", "5")),
        description="Default number of chunks to retrieve",
    )

    default_null_threshold: float = Field(
        default=float(os.getenv("DEFAULT_NULL_THRESHOLD", "0.5")),
        description="Default threshold for null responses",
    )

    default_max_distance: float = Field(
        default=float(os.getenv("DEFAULT_MAX_DISTANCE", "0.5")),
        description="Default maximum distance for retrieval",
    )

    # Validation
    @validator("ambiguity_threshold")
    def validate_ambiguity_threshold(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("Ambiguity threshold must be between 0 and 1")
        return v

    @validator("short_query_word_limit", "single_word_char_limit")
    def validate_positive_ints(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @validator("default_null_threshold", "default_max_distance")
    def validate_thresholds(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        return v


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
        default=int(os.getenv("CHUNK_SIZE", "450")),
        description="Number of characters per chunk",
    )

    chunk_overlap: int = Field(
        default=int(os.getenv("CHUNK_OVERLAP", "90")),
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

    top_k: int = Field(default=5, description="Default number of chunks to retrieve")

    null_threshold: float = Field(
        default=0.50,
        description="Confidence threshold for considering a response grounded",
    )

    max_distance: float = Field(
        default=0.50,
        description="Maximum cosine distance for retrieval (lower is more similar)",
    )

    rerank: bool = Field(
        default=False, description="Whether to enable reranking of results"
    )

    rerank_lex_weight: float = Field(
        default=0.5,
        description="Weight for lexical vs semantic similarity in reranking",
    )

    # DEPRECATED: Use LLM_TEMPERATURE and LLM_MAX_TOKENS instead
    temperature: float = Field(
        default=float(os.getenv("RETRIEVAL_TEMPERATURE", "0.1")),
        description="Default temperature for LLM generation (DEPRECATED: use LLM_TEMPERATURE)"
    )

    max_tokens: int = Field(
        default=int(os.getenv("RETRIEVAL_MAX_TOKENS", "300")),
        description="Default maximum number of tokens to generate (DEPRECATED: use LLM_MAX_TOKENS)"
    )


# Add this new class to app/settings.py
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

    # Legacy Ollama settings (for backward compatibility)
    ollama_host: str = Field(
        default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        description="URL for the Ollama API server (DEPRECATED: use LLM_OLLAMA_HOST)",
    )
    ollama_model: str = Field(
        default=os.getenv("OLLAMA_MODEL", "llama3.2:3b-instruct-q4_K_M"),
        description="Ollama model name (DEPRECATED: use LLM_OLLAMA_MODEL)",
    )
    num_ctx: int = Field(
        default=int(os.getenv("NUM_CTX", "2048")),
        description="Context window size (DEPRECATED: use LLM_NUM_CTX)",
        ge=512,
        le=16384,
    )
    ollama_timeout: int = Field(
        default=int(os.getenv("OLLAMA_TIMEOUT", "60")),
        description="Ollama timeout (DEPRECATED: use LLM_OLLAMA_TIMEOUT)",
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
        default=float(os.getenv("MAX_DISTANCE", "0.50")),
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
    query_router: QueryRouterSettings = Field(
        default_factory=QueryRouterSettings, description="Query routing configuration"
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
query_router_settings = settings.query_router
