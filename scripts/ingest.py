import sys
import os
import logging

# Add project root to path
sys.path.append(os.getcwd())

from app.ingest import ingest_paths
from app.logging_config import setup_logging
from app.settings import settings


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting manual ingestion script...")
    logger.info(f"Target Directory: {settings.docs_dir}")

    try:
        # Run ingestion
        # ingest_paths defaults to settings.docs_dir if no paths provided
        chunks_added = ingest_paths()

        logger.info(f"Ingestion complete! Added {chunks_added} chunks.")

        # Trigger BM25 rebuild
        if chunks_added > 0 or True:  # Force rebuild check
            logger.info("Rebuilding BM25 index...")
            from app.retrieval.bm25_search import BM25Index
            from app.retrieval.vector_store import get_vector_store

            vector_store = get_vector_store()
            documents = vector_store.get_all_documents()

            if documents:
                bm25_index = BM25Index(
                    index_path=f"{settings.chroma_dir}/bm25_index.pkl",
                    k1=settings.bm25.k1,
                    b=settings.bm25.b,
                )
                bm25_index.build_index(documents)
                bm25_index.save_index()
                logger.info(f"BM25 index rebuilt with {len(documents)} documents.")
            else:
                logger.warning("Vector store empty, skipping BM25 build.")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
