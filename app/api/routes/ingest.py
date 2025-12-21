"""
Document ingestion endpoint for Personal RAG system.
"""

import logging

from fastapi import APIRouter

from app.ingest import ingest_paths
from app.models import IngestRequest, IngestResponse
from app.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """Ingest documents into the vector store.

    Args:
        req: Ingest request with paths to process

    Returns:
        IngestResponse with count of ingested chunks
    """
    paths = req.paths or [settings.docs_dir]
    added = ingest_paths(paths)

    # Automatically rebuild BM25 index to keep it in sync with ChromaDB
    if added > 0:
        try:
            logger.info(f"Ingested {added} chunks, rebuilding BM25 index...")
            from app.retrieval.bm25_search import BM25Index
            from app.retrieval.vector_store import get_vector_store

            # Fetch all documents from VectorStore
            vector_store = get_vector_store()
            documents = vector_store.get_all_documents()

            if documents:
                # Rebuild BM25 index
                bm25_index = BM25Index(
                    index_path="data/chroma/bm25_index.pkl",
                    k1=settings.bm25.k1,
                    b=settings.bm25.b,
                )
                bm25_index.build_index(documents)
                bm25_index.save_index()

                logger.info(
                    f"BM25 index rebuilt successfully with {len(documents)} documents"
                )
            else:
                logger.warning(
                    "No documents found in VectorStore, skipping BM25 rebuild"
                )

        except Exception as e:
            logger.error(f"Failed to rebuild BM25 index: {e}")
            logger.warning("Ingestion completed but BM25 index may be out of sync")

    return IngestResponse(ingested_chunks=added)
