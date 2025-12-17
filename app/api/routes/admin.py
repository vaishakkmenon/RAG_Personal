"""
Admin endpoints for Personal RAG system.

Contains administrative operations like clearing ChromaDB.
"""

import shutil
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, HTTPException

from app.settings import settings
from app.retrieval import reset_collection
from app.retrieval.fallback_cache import get_fallback_cache

router = APIRouter()


@router.delete("/chromadb")
async def clear_chromadb() -> Dict[str, Any]:
    """Clear ChromaDB storage completely.

    WARNING: This permanently deletes all documents from the ChromaDB collection.
    Use this when you need to start fresh with a clean database.

    This uses ChromaDB's internal reset mechanism which safely handles the
    collection deletion and recreation without requiring a container restart.

    Returns:
        Dictionary with operation status and details
    """
    try:
        # Use ChromaDB's internal reset mechanism
        # This properly handles the collection deletion and recreation
        reset_collection()

        return {
            "status": "success",
            "message": "Successfully cleared ChromaDB collection",
            "collection_name": settings.collection_name,
            "method": "collection_reset",
            "next_steps": [
                "Run your ingestion script to rebuild the database",
                "Verify no duplicate versions are created"
            ]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during collection reset: {str(e)}"
        )


@router.get("/chromadb/status")
async def chromadb_status() -> Dict[str, Any]:
    """Get ChromaDB storage status.

    Returns information about the current state of ChromaDB storage.

    Returns:
        Dictionary with storage status and statistics
    """
    chroma_path = Path(settings.chroma_dir)

    if not chroma_path.exists():
        return {
            "status": "empty",
            "message": "ChromaDB directory does not exist",
            "path": str(chroma_path.absolute()),
            "exists": False,
            "files_count": 0,
            "dirs_count": 0
        }

    try:
        files_count = sum(1 for _ in chroma_path.rglob('*') if _.is_file())
        dirs_count = sum(1 for _ in chroma_path.rglob('*') if _.is_dir())

        # Get directory size
        total_size = sum(f.stat().st_size for f in chroma_path.rglob('*') if f.is_file())

        # Format size in human-readable format
        if total_size < 1024:
            size_str = f"{total_size} B"
        elif total_size < 1024 ** 2:
            size_str = f"{total_size / 1024:.2f} KB"
        elif total_size < 1024 ** 3:
            size_str = f"{total_size / (1024 ** 2):.2f} MB"
        else:
            size_str = f"{total_size / (1024 ** 3):.2f} GB"

        is_empty = files_count == 0 and dirs_count == 0

        return {
            "status": "empty" if is_empty else "populated",
            "message": "ChromaDB directory is empty" if is_empty else "ChromaDB directory contains data",
            "path": str(chroma_path.absolute()),
            "exists": True,
            "files_count": files_count,
            "dirs_count": dirs_count,
            "total_size_bytes": total_size,
            "total_size": size_str
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error checking status: {str(e)}"
        )


@router.get("/fallback-cache/stats")
async def get_fallback_cache_stats() -> Dict[str, Any]:
    """Get retrieval fallback cache statistics.

    Returns information about the fallback cache used when ChromaDB is unavailable.

    Returns:
        Dictionary with cache statistics including hit rate and size
    """
    try:
        cache = get_fallback_cache()
        stats = cache.get_stats()

        return {
            "status": "active",
            "statistics": stats,
            "description": "Fallback cache provides cached retrieval results when ChromaDB is unavailable"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting cache stats: {str(e)}"
        )


@router.delete("/fallback-cache")
async def clear_fallback_cache() -> Dict[str, Any]:
    """Clear the retrieval fallback cache.

    Removes all cached retrieval results. This is useful when you want to
    ensure fresh results after updating the vector database.

    Returns:
        Dictionary with operation status
    """
    try:
        cache = get_fallback_cache()
        stats_before = cache.get_stats()
        cache.clear()

        return {
            "status": "success",
            "message": "Fallback cache cleared successfully",
            "entries_cleared": stats_before["cache_size"],
            "statistics_reset": {
                "hits": stats_before["hits"],
                "misses": stats_before["misses"],
                "fallback_uses": stats_before["fallback_uses"]
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing cache: {str(e)}"
        )


@router.post("/fallback-cache/cleanup")
async def cleanup_fallback_cache() -> Dict[str, Any]:
    """Clean up expired entries from the fallback cache.

    Removes entries that have exceeded their TTL (time-to-live).

    Returns:
        Dictionary with cleanup results
    """
    try:
        cache = get_fallback_cache()
        removed_count = cache.cleanup_expired()

        return {
            "status": "success",
            "message": "Expired entries cleaned up",
            "entries_removed": removed_count,
            "current_cache_size": cache.get_stats()["cache_size"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error cleaning up cache: {str(e)}"
        )
