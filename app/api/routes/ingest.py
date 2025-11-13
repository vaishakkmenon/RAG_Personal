"""
Document ingestion endpoint for Personal RAG system.
"""

import logging

from fastapi import APIRouter

from ...ingest import ingest_paths
from ...models import IngestRequest, IngestResponse
from ...settings import settings

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

    return IngestResponse(ingested_chunks=added)
