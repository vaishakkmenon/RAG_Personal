"""
Application settings for Personal RAG system.

Loads configuration from environment variables (.env file) with sensible defaults.
All settings can be overridden via environment variables.
"""

import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


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
        description="Context window size for Ollama model (tokens)"
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
        default=int(os.getenv("CHUNK_SIZE", "200")),
        description="Target chunk size in characters"
    )
    
    chunk_overlap: int = Field(
        default=int(os.getenv("CHUNK_OVERLAP", "75")),
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