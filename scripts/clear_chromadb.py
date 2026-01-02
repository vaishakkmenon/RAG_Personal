#!/usr/bin/env python3
"""
Clear ChromaDB Collection

This script safely clears all data from the ChromaDB collection.
Run this before re-ingesting to avoid duplicates.

Usage:
    python scripts/clear_chromadb.py

    # Or with confirmation prompt:
    python scripts/clear_chromadb.py --confirm
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.retrieval.vector_store import VectorStore, COLLECTION_NAME


def clear_collection(confirm: bool = False):
    """Clear the ChromaDB collection."""

    # Get vector store instance
    vector_store = VectorStore()
    client = vector_store.get_client()

    # Get current count
    try:
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()
        print(f"Current collection '{COLLECTION_NAME}' has {count} documents")
    except Exception:
        print(f"Collection '{COLLECTION_NAME}' does not exist or is empty")
        return

    if count == 0:
        print("Collection is already empty. Nothing to clear.")
        return

    # Confirmation
    if confirm:
        response = input(
            f"\nAre you sure you want to delete all {count} documents? (yes/no): "
        )
        if response.lower() != "yes":
            print("Aborted.")
            return

    # Delete the collection
    print(f"\nDeleting collection '{COLLECTION_NAME}'...")
    client.delete_collection(COLLECTION_NAME)
    print("✓ Collection deleted")

    # Don't recreate - let VectorStore create it with proper embedding function
    print(
        f"\nCollection '{COLLECTION_NAME}' deleted. It will be recreated with proper settings on next ingestion."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clear ChromaDB collection")
    parser.add_argument(
        "--confirm", action="store_true", help="Prompt for confirmation before deleting"
    )

    args = parser.parse_args()
    clear_collection(confirm=args.confirm)
