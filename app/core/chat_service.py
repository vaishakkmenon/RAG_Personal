"""
Chat service for Personal RAG system.

Main business logic for handling chat requests, orchestrating retrieval,
prompt building, and LLM generation.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status

from ..models import ChatRequest, ChatResponse, ChatSource, AmbiguityMetadata
from ..monitoring import time_execution_info
from ..prompting import build_clarification_message, create_default_prompt_builder
from ..query_router import route_query
from ..retrieval import search
from ..services.llm import generate_with_ollama
from ..services.reranker import rerank_chunks
from ..settings import settings

logger = logging.getLogger(__name__)

# Optional metrics import
try:
    from ..metrics import rag_retrieval_chunks

    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
    logger.debug("Metrics not available for chat service")


class ChatService:
    """Service for handling RAG chat requests."""

    def __init__(self):
        """Initialize chat service."""
        self.prompt_builder = create_default_prompt_builder()

    def _merge_params(
        self, manual: Dict[str, Any], routed: Dict[str, Any], defaults: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge manual overrides with routed params and defaults.

        Args:
            manual: Manual parameter overrides
            routed: Parameters from query router
            defaults: Default parameters

        Returns:
            Merged parameters dict
        """
        result = defaults.copy()
        result.update(routed)
        result.update({k: v for k, v in manual.items() if v is not None})
        return result

    def _format_sources(self, chunks: List[dict]) -> List[Dict[str, str]]:
        """Format chunks for the prompt builder.

        Args:
            chunks: List of retrieved chunks

        Returns:
            List of formatted sources
        """
        return [
            {"source": c.get("source", "unknown"), "text": c.get("text", "")}
            for c in chunks
            if c.get("text", "").strip()
        ]

    def _build_chat_sources(self, chunks: List[dict]) -> List[ChatSource]:
        """Create output source metadata for chatbot responses.

        Args:
            chunks: List of retrieved chunks

        Returns:
            List of ChatSource objects
        """
        sources: List[ChatSource] = []
        for chunk in chunks:
            text = chunk.get("text", "")
            truncated = f"{text[:200]}..." if len(text) > 200 else text
            sources.append(
                ChatSource(
                    id=chunk.get("id", ""),
                    source=chunk.get("source", ""),
                    text=truncated,
                    distance=chunk.get("distance", 1.0),
                )
            )
        return sources

    @time_execution_info
    def handle_chat(
        self,
        request: ChatRequest,
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
    ) -> ChatResponse:
        """Handle a chat request with RAG.

        Args:
            request: Chat request with question
            grounded_only: Whether to require grounded responses
            null_threshold: Distance threshold for grounding
            max_distance: Maximum distance for retrieval
            top_k: Number of chunks to retrieve
            temperature: LLM temperature
            max_tokens: Maximum tokens to generate
            rerank: Whether to rerank results
            rerank_lex_weight: Weight for lexical reranking
            doc_type: Document type filter
            term_id: Term ID filter
            level: Level filter
            model: LLM model override
            use_router: Whether to use query router

        Returns:
            ChatResponse with answer and metadata
        """
        # Apply defaults from settings if not provided
        temperature = (
            temperature if temperature is not None else settings.retrieval.temperature
        )
        max_tokens = (
            max_tokens if max_tokens is not None else settings.retrieval.max_tokens
        )
        model_name = model or settings.ollama_model

        # Default parameters
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

        # Route query if enabled
        if use_router:
            routed_params = route_query(request.question)

            params = self._merge_params(
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

        # Check for ambiguity from router
        if use_router and params.get("is_ambiguous"):
            logger.info(
                "Ambiguous query detected; prompting user for clarification",
                extra={
                    "question": request.question,
                    "ambiguity_score": params.get("ambiguity_score"),
                },
            )
            clarification = build_clarification_message(
                request.question, self.prompt_builder.config
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

        # Check for heuristic ambiguity
        heuristic_ambiguous, _ = self.prompt_builder.is_ambiguous(request.question)
        if heuristic_ambiguous:
            logger.info(
                "Heuristic ambiguity detected; prompting user for clarification"
            )
            params["is_ambiguous"] = True
            clarification = build_clarification_message(
                request.question, self.prompt_builder.config
            )
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

        if METRICS_ENABLED:
            rag_retrieval_chunks.observe(len(chunks))

        # Grounding check - no chunks
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

        # Grounding check - distance threshold
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

        # Standard RAG flow
        return self._handle_standard_query(request.question, params, chunks)

    def _handle_standard_query(
        self, question: str, params: Dict[str, Any], chunks: List[dict]
    ) -> ChatResponse:
        """Handle standard RAG query with LLM generation.

        Args:
            question: User's question
            params: Query parameters
            chunks: Retrieved chunks

        Returns:
            ChatResponse
        """
        # Prepare context for prompt builder
        formatted_chunks = self._format_sources(chunks)

        # Build and validate prompt
        prompt_result = self.prompt_builder.build_prompt(
            question=question, context_chunks=formatted_chunks
        )

        # Handle ambiguous questions or missing context
        if prompt_result.status == "ambiguous":
            params["is_ambiguous"] = True
            clarification = prompt_result.message or build_clarification_message(
                question, self.prompt_builder.config
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

        # Check for refusal
        if self.prompt_builder.is_refusal(answer):
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

        # Check if clarification is needed
        if params.get("is_ambiguous"):
            if self.prompt_builder.needs_clarification(answer):
                logger.info(
                    "Ambiguity flagged but answer lacked clarification; returning clarification prompt"
                )
                clarification = build_clarification_message(
                    question, self.prompt_builder.config
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

        # Return successful response
        return ChatResponse(
            answer=answer,
            sources=self._build_chat_sources(chunks),
            grounded=True,
            ambiguity=AmbiguityMetadata(
                is_ambiguous=params.get("is_ambiguous", False),
                score=params.get("ambiguity_score", 0.0),
                clarification_requested=False,
            ),
        )


__all__ = ["ChatService"]
