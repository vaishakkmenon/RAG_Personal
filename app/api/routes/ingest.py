"""
Document ingestion endpoint for Personal RAG system.
"""

import logging

from fastapi import APIRouter

from ...certifications import get_registry
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

    # Reload certification registry after ingestion
    try:
        cert_registry = get_registry()
        cert_registry.reload()
    except Exception as exc:  # pragma: no cover - diagnostic logging only
        logger.warning("Failed to reload certification registry: %s", exc)

    return IngestResponse(ingested_chunks=added)
