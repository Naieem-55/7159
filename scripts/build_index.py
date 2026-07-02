"""CLI entry point: `python -m scripts.build_index`

Rebuilds the ChromaDB vector index from data/book.txt.
"""
from backend.indexing import build_index

if __name__ == "__main__":
    count = build_index()
    print(f"Indexed {count} chunks.")
