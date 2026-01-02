"""
Admin endpoints for Personal RAG system.

Contains administrative operations like clearing ChromaDB and document management.
"""

import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from app.core.auth import get_current_admin_user

from app.settings import settings
from app.retrieval import reset_collection
from app.retrieval.fallback_cache import get_fallback_cache

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(get_current_admin_user)])


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
                "Verify no duplicate versions are created",
            ],
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during collection reset: {str(e)}"
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
            "dirs_count": 0,
        }

    try:
        files_count = sum(1 for _ in chroma_path.rglob("*") if _.is_file())
        dirs_count = sum(1 for _ in chroma_path.rglob("*") if _.is_dir())

        # Get directory size
        total_size = sum(
            f.stat().st_size for f in chroma_path.rglob("*") if f.is_file()
        )

        # Format size in human-readable format
        if total_size < 1024:
            size_str = f"{total_size} B"
        elif total_size < 1024**2:
            size_str = f"{total_size / 1024:.2f} KB"
        elif total_size < 1024**3:
            size_str = f"{total_size / (1024 ** 2):.2f} MB"
        else:
            size_str = f"{total_size / (1024 ** 3):.2f} GB"

        is_empty = files_count == 0 and dirs_count == 0

        return {
            "status": "empty" if is_empty else "populated",
            "message": "ChromaDB directory is empty"
            if is_empty
            else "ChromaDB directory contains data",
            "path": str(chroma_path.absolute()),
            "exists": True,
            "files_count": files_count,
            "dirs_count": dirs_count,
            "total_size_bytes": total_size,
            "total_size": size_str,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking status: {str(e)}")


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
            "description": "Fallback cache provides cached retrieval results when ChromaDB is unavailable",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting cache stats: {str(e)}"
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
                "fallback_uses": stats_before["fallback_uses"],
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")


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
            "current_cache_size": cache.get_stats()["cache_size"],
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error cleaning up cache: {str(e)}"
        )


# =============================================================================
# Document Management Endpoints
# =============================================================================

ALLOWED_EXTENSIONS = {".md", ".txt", ".yaml", ".yml"}
MAX_FILE_SIZE = 1 * 1024 * 1024  # 1MB


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and other attacks."""
    # Remove any path components
    filename = Path(filename).name
    # Replace spaces with hyphens
    filename = filename.replace(" ", "-")
    # Remove any characters that aren't alphanumeric, hyphen, underscore, or dot
    filename = re.sub(r"[^a-zA-Z0-9_\-.]", "", filename)
    # Prevent hidden files
    if filename.startswith("."):
        filename = filename[1:]
    return filename


@router.get("/documents")
async def list_documents() -> Dict[str, Any]:
    """List all documents in the knowledge base directory.

    Returns:
        Dictionary with list of documents and their metadata
    """
    docs_path = Path(settings.docs_dir)

    if not docs_path.exists():
        return {
            "status": "empty",
            "message": "Documents directory does not exist",
            "documents": [],
            "total_count": 0,
        }

    documents = []
    for file_path in docs_path.rglob("*"):
        if file_path.is_file() and file_path.suffix in ALLOWED_EXTENSIONS:
            stat = file_path.stat()
            documents.append(
                {
                    "name": file_path.name,
                    "path": str(file_path.relative_to(docs_path)),
                    "size_bytes": stat.st_size,
                    "size": _format_size(stat.st_size),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "extension": file_path.suffix,
                }
            )

    # Sort by modified date, newest first
    documents.sort(key=lambda x: x["modified"], reverse=True)

    return {
        "status": "success",
        "documents": documents,
        "total_count": len(documents),
        "docs_dir": str(docs_path),
    }


@router.get("/documents/{filename}")
async def get_document(filename: str) -> Dict[str, Any]:
    """Get the content of a specific document.

    Args:
        filename: Name of the document file

    Returns:
        Dictionary with document content and metadata
    """
    safe_filename = sanitize_filename(filename)
    docs_path = Path(settings.docs_dir)
    file_path = docs_path / safe_filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Document '{safe_filename}' not found"
        )

    if file_path.suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, detail=f"File type '{file_path.suffix}' not allowed"
        )

    try:
        content = file_path.read_text(encoding="utf-8")
        stat = file_path.stat()

        return {
            "status": "success",
            "name": safe_filename,
            "content": content,
            "size_bytes": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading document: {str(e)}")


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    overwrite: bool = Form(default=False),
) -> Dict[str, Any]:
    """Upload a document to the knowledge base.

    Args:
        file: The document file to upload (markdown, txt, yaml)
        overwrite: Whether to overwrite existing file with same name

    Returns:
        Dictionary with upload status and file details
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    safe_filename = sanitize_filename(file.filename)
    if not safe_filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Check extension
    ext = Path(safe_filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    docs_path = Path(settings.docs_dir)
    docs_path.mkdir(parents=True, exist_ok=True)
    file_path = docs_path / safe_filename

    # Check if file exists
    if file_path.exists() and not overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"Document '{safe_filename}' already exists. Set overwrite=true to replace.",
        )

    try:
        # Read and validate file size
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {_format_size(MAX_FILE_SIZE)}",
            )

        # Write file
        file_path.write_bytes(content)
        logger.info(f"Document uploaded: {safe_filename} ({len(content)} bytes)")

        return {
            "status": "success",
            "message": f"Document '{safe_filename}' uploaded successfully",
            "name": safe_filename,
            "size_bytes": len(content),
            "size": _format_size(len(content)),
            "path": str(file_path.relative_to(docs_path)),
            "overwritten": file_path.exists() and overwrite,
            "next_steps": [
                "Call POST /ingest to add this document to the knowledge base",
                "Or call POST /admin/reingest to clear and rebuild the entire knowledge base",
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error uploading document: {str(e)}"
        )


@router.delete("/documents/{filename}")
async def delete_document(filename: str) -> Dict[str, Any]:
    """Delete a document from the knowledge base directory.

    Note: This only removes the file. You should reingest to update the vector database.

    Args:
        filename: Name of the document file to delete

    Returns:
        Dictionary with deletion status
    """
    safe_filename = sanitize_filename(filename)
    docs_path = Path(settings.docs_dir)
    file_path = docs_path / safe_filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Document '{safe_filename}' not found"
        )

    try:
        file_path.unlink()
        logger.info(f"Document deleted: {safe_filename}")

        return {
            "status": "success",
            "message": f"Document '{safe_filename}' deleted successfully",
            "name": safe_filename,
            "next_steps": [
                "Call POST /admin/reingest to rebuild the knowledge base without this document",
            ],
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting document: {str(e)}"
        )


@router.post("/reingest")
async def reingest_all() -> Dict[str, Any]:
    """Clear the knowledge base and reingest all documents.

    This performs a full rebuild:
    1. Clears ChromaDB collection
    2. Reingests all documents from docs_dir
    3. Rebuilds BM25 index

    Returns:
        Dictionary with reingest status and statistics
    """
    try:
        # Step 1: Clear ChromaDB
        chroma_path = Path(settings.chroma_dir)
        if chroma_path.exists():
            for item in chroma_path.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            logger.info("ChromaDB directory cleared")

        # Step 2: Ingest documents
        from app.ingest import ingest_paths

        added = ingest_paths([settings.docs_dir])
        logger.info(f"Ingested {added} chunks")

        # Step 3: Rebuild BM25 index
        bm25_info = {"status": "skipped", "message": "No chunks added"}

        if added > 0:
            try:
                from app.retrieval.bm25_search import BM25Index
                from app.retrieval.vector_store import get_vector_store

                vector_store = get_vector_store()
                documents = vector_store.get_all_documents()

                if documents:
                    bm25_index = BM25Index(
                        index_path="data/chroma/bm25_index.pkl",
                        k1=settings.bm25.k1,
                        b=settings.bm25.b,
                    )
                    bm25_index.build_index(documents)
                    bm25_index.save_index()
                    bm25_info = {"status": "rebuilt", "doc_count": len(documents)}
                    logger.info(f"BM25 index rebuilt with {len(documents)} documents")

            except Exception as e:
                logger.error(f"Failed to rebuild BM25 index: {e}")
                bm25_info = {"status": "failed", "error": str(e)}

        return {
            "status": "success",
            "message": "Knowledge base rebuilt successfully",
            "ingested_chunks": added,
            "bm25_stats": bm25_info,
        }

    except Exception as e:
        logger.error(f"Reingest failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reingest failed: {str(e)}")


def _format_size(size_bytes: int) -> str:
    """Format size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"
