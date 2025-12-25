import sys
import os
from pprint import pprint

# Add project root to path
sys.path.append(os.getcwd())

from app.retrieval.vector_store import get_vector_store
from app.retrieval.search_engine import get_search_engine
from app.logging_config import setup_logging


def check_document_presence(filename):
    print(f"Checking for presence of {filename}...")
    vs = get_vector_store()

    # This might be slow if many docs, but okay for debugging
    all_docs = vs.get_all_documents()
    found = False
    count = 0
    for doc in all_docs:
        if filename in doc.metadata.get("source", ""):
            if not found:
                print(f"Found at least one chunk for {filename}")
                pprint(doc.metadata)
                found = True
            count += 1

    print(f"Total chunks found for {filename}: {count}")
    return found


def debug_search(query):
    print(f"\nRunning search for query: '{query}'")
    engine = get_search_engine()
    results = engine.search(query, k=10)

    print(f"Found {len(results)} results:")
    for i, res in enumerate(results):
        print(f"{i+1}. {res['metadata'].get('source')} (Score: {res['score']})")
        print(f"   Snippet: {res['content'][:100]}...")


def main():
    setup_logging()

    target_file = "certificate--aws-ai-practitioner--2025-06-01.md"

    # 1. Check if file is in store
    if not check_document_presence(target_file):
        print(f"WARNING: {target_file} NOT found in vector store.")
    else:
        print(f"SUCCESS: {target_file} is in vector store.")

    # 2. Run the user's query
    query = "I have earned the following certifications: * Certified Kubernetes Administrator (CKA) [1], earned on June 26, 2024, and valid through June 26, 2026 * AWS Certified Cloud Practitioner [2], earned on May 26, 2025, and valid through May 26, 2028 * AWS Certified AI Practitioner [2], earned on [date not specified], and [expiration date not specified]"
    debug_search(query)


if __name__ == "__main__":
    main()
