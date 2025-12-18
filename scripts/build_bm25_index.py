"""
Build BM25 index from existing ChromaDB collection.

This script extracts all documents from ChromaDB and builds a BM25 index
for keyword-based retrieval.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.retrieval.store import _collection
from app.retrieval.bm25_search import BM25Index

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Build BM25 index from ChromaDB collection."""
    # Import settings
    from app.settings import settings

    logger.info("=" * 80)
    logger.info("Building BM25 Index from ChromaDB")
    logger.info(
        f"Parameters: k1={settings.bm25.k1}, b={settings.bm25.b}, rrf_k={settings.bm25.rrf_k}"
    )
    logger.info("=" * 80)

    # Get all documents from ChromaDB
    logger.info("Fetching all documents from ChromaDB...")
    result = _collection.get(include=["documents", "metadatas"])

    if not result or not result["ids"]:
        logger.error("No documents found in ChromaDB collection!")
        sys.exit(1)

    # Prepare documents for BM25
    documents = []
    for i, doc_id in enumerate(result["ids"]):
        text = result["documents"][i] if result["documents"] else ""
        metadata = result["metadatas"][i] if result["metadatas"] else {}

        documents.append({"id": doc_id, "text": text, "metadata": metadata})

    logger.info(f"Fetched {len(documents)} documents from ChromaDB")

    # Build BM25 index with optimized parameters from settings
    bm25_index = BM25Index(
        index_path="data/chroma/bm25_index.pkl", k1=settings.bm25.k1, b=settings.bm25.b
    )
    bm25_index.build_index(documents)

    # Save index
    bm25_index.save_index()

    logger.info("=" * 80)
    logger.info(f"BM25 index built successfully with {len(documents)} documents")
    logger.info(f"Parameters used: k1={settings.bm25.k1}, b={settings.bm25.b}")
    logger.info("Index saved to: data/chroma/bm25_index.pkl")
    logger.info("=" * 80)

    # Test the index
    logger.info("\nTesting BM25 index with sample queries...")

    test_queries = [
        "CS 350 B grade",
        "AWS certifications",
        "CKA expire",
        "What is my GPA",
    ]

    for query in test_queries:
        results = bm25_index.search(query, k=3)
        logger.info(f"\nQuery: '{query}'")
        logger.info(f"  Found {len(results)} results")
        if results:
            top_result = results[0]
            logger.info(
                f"  Top result: {top_result['id']} (score: {top_result['bm25_score']:.2f})"
            )
            logger.info(f"  Text preview: {top_result['text'][:100]}...")


if __name__ == "__main__":
    main()
