import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from app.retrieval.vector_store import get_vector_store
from app.retrieval.search_engine import get_search_engine
from app.logging_config import setup_logging


def check_bm25_presence(engine, target_file):
    print(f"\n--- Checking BM25 Index for {target_file} ---")
    if not engine.bm25_index:
        print("BM25 index is NOT loaded in search engine.")
        return

    # Check if file is in BM25 documents (using simple string match for now)
    # BM25Index typically stores a list of docs. We need to check their metadata/content source.
    # The actual implementation of BM25Index might vary, assuming it has .documents with metadata
    found_count = 0
    if hasattr(engine.bm25_index, "corpus"):
        # Some implementations store corpus
        pass

    # Let's try to search specifically for a unique string in that file
    # "dc53d66c1c1d46a18e76f862c9a07bdb" is the credential ID
    unique_term = "dc53d66c1c1d46a18e76f862c9a07bdb"
    print(f"Executing BM25 specific search for unique term: {unique_term}")
    results = engine.bm25_index.search(unique_term, k=5)

    if results:
        print(f"Found {len(results)} matches in BM25 for unique term:")
        for res in results:
            src = res["metadata"].get("source", "unknown")
            score = res.get("bm25_score", "N/A")
            print(f"  - {src} (Score: {score})")
            if target_file in src:
                found_count += 1
    else:
        print("No matches found in BM25 for unique term.")

    if found_count > 0:
        print(f"SUCCESS: {target_file} appears to be in BM25 index.")
    else:
        print(f"WARNING: {target_file} might be missing from BM25 index.")


def debug_pipeline(query):
    print("\n--- Debugging Search Pipeline ---")
    print(f"Query: '{query}'")

    engine = get_search_engine()
    vector_store = get_vector_store()

    # 1. Semantic Search Only
    print("\n[1] Semantic Search (Vector Only) - Top 10")
    semantic_results = vector_store.search(query, k=10, max_distance=0.8)
    for i, res in enumerate(semantic_results):
        print(f"  {i+1}. {res['metadata'].get('source')} (Dist: {res['distance']})")

    # 2. BM25 Search Only
    if engine.bm25_index:
        print("\n[2] Keyword Search (BM25 Only) - Top 10")
        bm25_results = engine.bm25_index.search(query, k=10)
        for i, res in enumerate(bm25_results):
            score = res.get("bm25_score", "N/A")
            print(f"  {i+1}. {res['metadata'].get('source')} (Score: {score})")
    else:
        print("\n[2] BM25 not available")

    # 3. Hybrid Search (Pre-Rerank)
    print("\n[3] Hybrid Search (Before Reranking) - Top 10")
    # Access private method to see candidates
    hybrid_results = engine._hybrid_search(query, k=10, max_distance=0.8)
    for i, res in enumerate(hybrid_results):
        score = res.get("rrf_score", res.get("score", "N/A"))
        print(f"  {i+1}. {res['metadata'].get('source')} (Fusion Score: {score})")

    # 4. Final Reranked
    print("\n[4] Final Reranked - Top 5")
    final_results = engine.search(query, k=5)
    for i, res in enumerate(final_results):
        score = res.get("cross_encoder_score", res.get("distance", "N/A"))
        print(f"  {i+1}. {res['metadata'].get('source')} (Final Score: {score})")


def main():
    setup_logging()
    target_file = "certificate--aws-ai-practitioner--2025-06-01.md"
    engine = get_search_engine()

    check_bm25_presence(engine, target_file)

    query = "What certifications have I earned?"
    debug_pipeline(query)


if __name__ == "__main__":
    main()
