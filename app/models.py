"""
Pydantic models for Personal RAG API.

Defines request/response schemas for:
- Document ingestion
- Chat/Q&A endpoints
- Error responses
"""

from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class IngestRequest(BaseModel):
    """Request to ingest documents into the vector database."""
    
    paths: Optional[List[str]] = Field(
        default=None,
        description="List of file or directory paths to ingest. If not provided, uses the default docs_dir.",
        json_schema_extra={"example": ["./data/mds/resume.md"]}
    )


class IngestResponse(BaseModel):
    """Response from document ingestion."""
    
    ingested_chunks: int = Field(
        description="Number of document chunks successfully ingested and indexed.",
        json_schema_extra={"example": 150}
    )


class ChatRequest(BaseModel):
    """Request to answer a question using RAG."""
    
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="The user's question to be answered using the ingested documents.",
        json_schema_extra={"example": "What was my graduate GPA?"}
    )
    
    @field_validator('question')
    @classmethod
    def strip_and_validate(cls, v: str) -> str:
        """Strip whitespace and ensure question has content."""
        v = v.strip()
        if not v:
            raise ValueError('Question cannot be empty or whitespace only')
        return v


class ChatSource(BaseModel):
    """A source document chunk used to answer the question."""
    
    id: str = Field(
        description="Unique identifier of the retrieved chunk",
        json_schema_extra={"example": "resume.md:0"}
    )
    
    source: str = Field(
        description="Original file path or source label",
        json_schema_extra={"example": "./data/mds/resume.md"}
    )
    
    text: str = Field(
        description="The actual retrieved chunk text (may be truncated)",
        json_schema_extra={"example": "Graduate GPA: 4.00"}
    )
    
    distance: float = Field(
        description="Cosine distance from query (0 = identical, 2 = opposite)",
        json_schema_extra={"example": 0.23}
    )


class ChatResponse(BaseModel):
    """Response containing the answer and supporting sources."""
    
    answer: str = Field(
        description="The LLM's final answer to the user's question",
        json_schema_extra={
            "example": (
                "Your graduate GPA was 4.00."
            )
        }
    )
    
    sources: List[ChatSource] = Field(
        description="List of supporting source chunks with distance scores",
        json_schema_extra={"example": [
            {
                "id": "complete_transcript_analysis.md:3",
                "source": "./data/mds/complete_transcript_analysis.md",
                "text": "Graduate GPA: 4.00",
                "distance": 0.23
            }
        ]}
    )
    
    grounded: bool = Field(
        description="Whether the answer is grounded in retrieved documents (True) or refused due to low confidence (False)",
        json_schema_extra={"example": True}
    )


class ErrorResponse(BaseModel):
    """Standard error response format."""
    
    detail: str = Field(
        description="Error detail message",
        json_schema_extra={"example": "Invalid API key"}
    )


# Export all models
__all__ = [
    "IngestRequest",
    "IngestResponse",
    "ChatRequest",
    "ChatSource",
    "ChatResponse",
    "ErrorResponse",
]