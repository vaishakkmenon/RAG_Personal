"""
Pydantic models for Personal RAG API.

Defines request/response schemas for:
- Document ingestion
- Chat/Q&A endpoints
- Error responses
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class IngestRequest(BaseModel):
    """Request to ingest documents into the vector database."""

    paths: Optional[List[str]] = Field(
        default=None,
        description="List of file or directory paths to ingest. If not provided, uses the default docs_dir.",
        json_schema_extra={"example": ["./data/mds/resume.md"]},
    )


class IngestResponse(BaseModel):
    """Response from document ingestion."""

    ingested_chunks: int = Field(
        description="Number of document chunks successfully ingested and indexed.",
        json_schema_extra={"example": 150},
    )


class ChatRequest(BaseModel):
    """Request to answer a question using RAG."""

    question: str = Field(
        min_length=3,
        max_length=2000,
        description="The user's question to be answered using the ingested documents.",
        json_schema_extra={"example": "What was my graduate GPA?"},
    )

    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for conversation continuity. If not provided, a new session will be created.",
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440000"},
        max_length=64,
        pattern="^[a-zA-Z0-9_-]+$",  # Only alphanumeric, dash, underscore
    )

    @field_validator("question")
    @classmethod
    def strip_and_validate(cls, v: str) -> str:
        """Strip whitespace, check content, and prevent spam/repetition."""
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty or whitespace only")

        # Enforce word count limit to prevent DoS and context overflow
        words = v.split()
        if len(words) > 300:
            raise ValueError("Question too long (max 300 words)")

        # Check for extremely repetitive patterns (potential abuse)
        if len(words) > 10 and len(set(words)) < len(words) / 10:  # >90% repeated words
            raise ValueError("Query contains excessive repetition")

        return v


class ChatSource(BaseModel):
    """A source document chunk used to answer the question."""

    id: str = Field(
        description="Unique identifier of the retrieved chunk",
        json_schema_extra={"example": "resume.md:0"},
    )

    source: str = Field(
        description="Original file path or source label",
        json_schema_extra={"example": "./data/mds/resume.md"},
    )

    text: str = Field(
        description="The actual retrieved chunk text (may be truncated)",
        json_schema_extra={"example": "Graduate GPA: 4.00"},
    )

    distance: Optional[float] = Field(
        default=1.0,
        description="Cosine distance from query (0 = identical, 2 = opposite)",
        json_schema_extra={"example": 0.23},
    )

    citation_index: Optional[int] = Field(
        default=None,
        description="Citation index [1], [2], etc. used in the answer for inline references",
        json_schema_extra={"example": 1},
    )


class AmbiguityMetadata(BaseModel):
    """Metadata describing ambiguity detection for a query."""

    is_ambiguous: bool = Field(
        description="Whether the router classified the question as ambiguous",
        json_schema_extra={"example": True},
    )

    score: float = Field(
        description="Confidence score (0-1) indicating ambiguity strength",
        json_schema_extra={"example": 0.85},
    )

    clarification_requested: bool = Field(
        description="Whether the system asked the user for clarification",
        json_schema_extra={"example": True},
    )


class RewriteMetadata(BaseModel):
    """Metadata about query rewriting performed on the user's question."""

    original_query: str = Field(description="Original user query before rewriting")

    rewritten_query: str = Field(description="Rewritten query after pattern matching")

    pattern_name: str = Field(
        description="Name of the matched pattern (e.g., 'negative_inference')"
    )

    pattern_type: str = Field(
        description="Type of pattern matching used (e.g., 'regex_list', 'keyword_presence')"
    )

    matched_entities: Dict[str, Any] = Field(
        default_factory=dict, description="Entities extracted during pattern matching"
    )

    rewrite_hint: Optional[str] = Field(
        default=None, description="Hint about the rewrite strategy applied"
    )

    metadata_filter_addition: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata filters to apply to retrieval"
    )

    latency_ms: float = Field(
        description="Time taken for pattern matching and rewriting (milliseconds)"
    )

    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score of pattern match (0.0-1.0)"
    )


class ChatResponse(BaseModel):
    """Response containing the answer and supporting sources."""

    answer: str = Field(
        description="The LLM's final answer to the user's question",
        json_schema_extra={"example": ("Your graduate GPA was 4.00.")},
    )

    sources: List[ChatSource] = Field(
        description="List of supporting source chunks with distance scores",
        json_schema_extra={
            "example": [
                {
                    "id": "complete_transcript_analysis.md:3",
                    "source": "./data/mds/complete_transcript_analysis.md",
                    "text": "Graduate GPA: 4.00",
                    "distance": 0.23,
                }
            ]
        },
    )

    grounded: bool = Field(
        description="Whether the answer is grounded in retrieved documents (True) or refused due to low confidence (False)",
        json_schema_extra={"example": True},
    )

    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="LLM's confidence score for the answer (0.0-1.0). Higher is more confident.",
        json_schema_extra={"example": 0.95},
    )

    ambiguity: Optional[AmbiguityMetadata] = Field(
        default=None,
        description="Details about ambiguity detection for this request",
        json_schema_extra={
            "example": {
                "is_ambiguous": True,
                "score": 0.85,
                "clarification_requested": True,
            }
        },
    )

    session_id: str = Field(
        description="Session ID for this conversation. Use this in subsequent requests to maintain conversation history.",
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440000"},
    )

    rewrite_metadata: Optional[RewriteMetadata] = Field(
        default=None,
        description="Query rewriting metadata (if query was rewritten by pattern matching)",
    )


class ErrorResponse(BaseModel):
    """Standard error response format."""

    detail: str = Field(
        description="Error detail message",
        json_schema_extra={"example": "Invalid API key"},
    )


# Export all models
__all__ = [
    "IngestRequest",
    "IngestResponse",
    "ChatRequest",
    "ChatSource",
    "AmbiguityMetadata",
    "ChatResponse",
    "ErrorResponse",
]
