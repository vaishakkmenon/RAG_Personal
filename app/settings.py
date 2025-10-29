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

class QueryRouterSettings(BaseModel):
    """Configuration for the query routing system."""
    
    # General settings
    cumulative_query_boost: int = Field(
        default=int(os.getenv("QUERY_ROUTER_CUMULATIVE_BOOST", "2")),
        description="Boost factor for cumulative queries"
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
        description="Regex patterns for document type detection"
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
        description="Patterns that indicate a broad question"
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
        description="Patterns that indicate a specific question"
    )
    
    cumulative_patterns: List[str] = Field(
        default_factory=lambda: [
            r"\b(overall|cumulative|total|complete|entire)\b",
            r"\b(undergraduate|graduate)\b.*\bgpa\b",
            r"\bhow\s+many\s+total\b",
            r"\bacademic\s+(summary|overview|performance)\b",
        ],
        description="Patterns that indicate a cumulative query"
    )
    
    # Retrieval parameters
    default_top_k: int = Field(
        default=int(os.getenv("DEFAULT_TOP_K", "5")),
        description="Default number of chunks to retrieve"
    )
    
    default_null_threshold: float = Field(
        default=float(os.getenv("DEFAULT_NULL_THRESHOLD", "0.5")),
        description="Default threshold for null responses"
    )
    
    default_max_distance: float = Field(
        default=float(os.getenv("DEFAULT_MAX_DISTANCE", "0.5")),
        description="Default maximum distance for retrieval"
    )
    
    # Validation
    @validator('default_null_threshold', 'default_max_distance')
    def validate_thresholds(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        return v


class IngestSettings(BaseModel):
    """Configuration for document ingestion."""
    
    allowed_extensions: Set[str] = Field(
        default=set(os.getenv("ALLOWED_EXTENSIONS", ".txt,.md,.pdf").split(",")),
        description="File extensions allowed for ingestion"
    )
    
    max_file_size: int = Field(
        default=int(os.getenv("MAX_FILE_SIZE_BYTES", str(5 * 1024 * 1024))),  # 5MB
        description="Maximum file size in bytes"
    )
    
    batch_size: int = Field(
        default=int(os.getenv("INGEST_BATCH_SIZE", "500")),
        description="Number of documents to process in a batch"
    )
    
    chunk_size: int = Field(
        default=int(os.getenv("CHUNK_SIZE", "1000")),
        description="Size of text chunks in characters"
    )
    
    chunk_overlap: int = Field(
        default=int(os.getenv("CHUNK_OVERLAP", "200")),
        description="Overlap between chunks in characters"
    )
    
    # Validation
    @validator('max_file_size')
    def validate_max_file_size(cls, v):
        if v <= 0:
            raise ValueError("Max file size must be positive")
        return v
    
    @validator('batch_size', 'chunk_size', 'chunk_overlap')
    def validate_positive_integer(cls, v):
        if v <= 0:
            raise ValueError("Value must be positive")
        return v


class Settings(BaseModel):
    """
    Global application configuration.
    
    Configuration priority:
    1. Environment variables (.env file or system)
    2. Defaults specified below
    
    Categories:
    - LLM: Ollama connection and model settings
    - Embeddings: Sentence transformer model
    - Storage: ChromaDB and document paths
    - Retrieval: Search and chunking parameters
    - Security: API authentication
    - Query Routing: Query analysis and routing
    - Ingest: Document ingestion settings
    """
    
    # ==================== LLM Settings ====================
    
    ollama_host: str = Field(
        default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        description="URL for the Ollama API server"
    )
    
    ollama_model: str = Field(
        default=os.getenv("OLLAMA_MODEL", "llama3.2:3b-instruct-q4_K_M"),
        description="Ollama model name (with quantization tag)"
    )
    
    num_ctx: int = Field(
        default=int(os.getenv("NUM_CTX", "2048")),
        description="Context window size for Ollama model (tokens)",
        ge=512,  # Minimum context size
        le=16384  # Maximum reasonable context size
    )
    
    ollama_timeout: int = Field(
        default=int(os.getenv("OLLAMA_TIMEOUT", "60")),
        description="Maximum seconds to wait for LLM responses"
    )
    
    # ==================== Embedding Settings ====================
    
    embed_model: str = Field(
        default=os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        description="SentenceTransformer model for embeddings"
    )
    
    # ==================== Storage Settings ====================
    
    chroma_dir: str = Field(
        default=os.getenv("CHROMA_DIR", "./data/chroma"),
        description="ChromaDB persistent storage directory"
    )
    
    docs_dir: str = Field(
        default=os.getenv("DOCS_DIR", "./data/mds"),
        description="Directory containing documents to ingest (resume, transcripts, certs)"
    )
    
    collection_name: str = Field(
        default=os.getenv("COLLECTION_NAME", "personal_rag"),
        description="ChromaDB collection name"
    )
    
    # ==================== Retrieval Settings ====================
    
    top_k: int = Field(
        default=int(os.getenv("TOP_K", "5")),
        description="Default number of chunks to retrieve"
    )
    
    max_distance: float = Field(
        default=float(os.getenv("MAX_DISTANCE", "0.50")),
        description="Maximum cosine distance for retrieval (0-2, lower = more similar)"
    )
    
    null_threshold: float = Field(
        default=float(os.getenv("NULL_THRESHOLD", "0.50")),
        description="Distance threshold for grounding check (refuse if distance > threshold)"
    )
    
    # ==================== Chunking Settings ====================
    
    chunk_size: int = Field(
        default=int(os.getenv("CHUNK_SIZE", "800")),
        description="Target chunk size in characters"
    )
    
    chunk_overlap: int = Field(
        default=int(os.getenv("CHUNK_OVERLAP", "200")),
        description="Overlap between consecutive chunks in characters"
    )
    
    # ==================== Security Settings ====================
    
    api_key: str = Field(
        default=os.getenv("API_KEY", "change-me"),
        description="API key for authentication (change in production!)"
    )
    
    # ==================== HTTP Settings ====================
    
    max_bytes: int = Field(
        default=int(os.getenv("MAX_BYTES", "32768")),
        description="Maximum HTTP request body size in bytes"
    )
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Prevent typos in environment variables
        validate_assignment = True  # Validate on attribute assignment


# Global settings instance
settings = Settings()

# Component-specific settings
query_router_settings = QueryRouterSettings()
ingest_settings = IngestSettings()


# Log critical settings on startup (don't log API key!)
if __name__ != "__main__":  # Only log when imported, not when running directly
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Settings loaded:")
    logger.info(f"  Ollama: {settings.ollama_host}")
    logger.info(f"  Model: {settings.ollama_model}")
    logger.info(f"  Embeddings: {settings.embed_model}")
    logger.info(f"  ChromaDB: {settings.chroma_dir}")
    logger.info(f"  Documents: {settings.docs_dir}")
    logger.info(f"  Collection: {settings.collection_name}")
    logger.info(f"  Retrieval: top_k={settings.top_k}, max_distance={settings.max_distance}")
    logger.info(f"  Chunking: size={settings.chunk_size}, overlap={settings.chunk_overlap}")


# Export
__all__ = ["settings", "Settings"]