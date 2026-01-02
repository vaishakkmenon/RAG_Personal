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
import shutil
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.settings import settings


def clear_collection(confirm: bool = False):
    """Clear the ChromaDB collection by wiping the data directory."""

    chroma_path = Path(settings.chroma_dir)

    if not chroma_path.exists():
        print(f"ChromaDB directory '{chroma_path}' does not exist. Nothing to clear.")
        return

    # Count files to give user an idea of what's there
    file_count = sum(1 for _ in chroma_path.rglob("*") if _.is_file())
    print(f"ChromaDB directory '{chroma_path}' contains {file_count} files")

    if file_count == 0:
        print("Directory is already empty. Nothing to clear.")
        return

    # Confirmation
    if confirm:
        response = input(
            "\nAre you sure you want to delete all ChromaDB data? (yes/no): "
        )
        if response.lower() != "yes":
            print("Aborted.")
            return

    # Wipe the directory contents (but keep the directory itself)
    print(f"\nClearing ChromaDB directory '{chroma_path}'...")
    for item in chroma_path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    print("✓ ChromaDB data cleared")
    print("\nCollection will be recreated with proper settings on next ingestion.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clear ChromaDB collection")
    parser.add_argument(
        "--confirm", action="store_true", help="Prompt for confirmation before deleting"
    )

    args = parser.parse_args()
    clear_collection(confirm=args.confirm)
