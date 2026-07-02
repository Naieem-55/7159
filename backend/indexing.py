"""Builds a searchable vector index from the course book.

Pipeline: load book text -> split into chapter-aware chunks -> embed each
chunk with the OpenAI embedding model -> persist in a local ChromaDB
collection. Re-running this script rebuilds the index from scratch, so it
is safe to call whenever the book content changes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import chromadb
from openai import OpenAI

from backend.config import settings

COLLECTION_NAME = "course_book"


@dataclass
class Chunk:
    """A chunk of book text with metadata preserved for citation."""

    text: str
    chapter: str
    chunk_id: str


def _split_by_chapter(raw_text: str) -> list[tuple[str, str]]:
    """Split the book into (chapter_title, chapter_body) pairs.

    The book file uses Markdown-style '# Chapter N: ...' headings, which we
    use as natural semantic boundaries before further sub-chunking.
    """
    pattern = re.compile(r"^# (Chapter .+)$", re.MULTILINE)
    matches = list(pattern.finditer(raw_text))
    if not matches:
        return [("Book", raw_text)]

    sections = []
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        body = raw_text[start:end].strip()
        sections.append((title, body))
    return sections


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-based chunks (simple semantic chunking).

    Splitting on paragraph boundaries first keeps related sentences together;
    paragraphs longer than `size` are further split with a sliding window.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buffer = ""

    for para in paragraphs:
        candidate = f"{buffer}\n\n{para}".strip() if buffer else para
        if len(candidate.split()) <= size:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer)
            words = para.split()
            if len(words) <= size:
                buffer = para
            else:
                step = max(size - overlap, 1)
                for start in range(0, len(words), step):
                    piece = " ".join(words[start : start + size])
                    if piece:
                        chunks.append(piece)
                buffer = ""
    if buffer:
        chunks.append(buffer)
    return chunks


def load_and_chunk_book() -> list[Chunk]:
    """Read the book file and produce chapter-tagged, semantically chunked pieces."""
    raw_text = settings.book_path.read_text(encoding="utf-8")
    chunks: list[Chunk] = []
    for chapter_title, body in _split_by_chapter(raw_text):
        pieces = _chunk_text(body, settings.chunk_size, settings.chunk_overlap)
        for idx, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    text=piece,
                    chapter=chapter_title,
                    chunk_id=f"{chapter_title}-{idx}",
                )
            )
    return chunks


def _embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Call the OpenAI embeddings API in a single batch request."""
    response = client.embeddings.create(model=settings.embedding_model, input=texts)
    return [item.embedding for item in response.data]


def build_index() -> int:
    """Build (or rebuild) the persistent vector store from the book.

    Returns the number of chunks indexed.
    """
    settings.validate()
    chunks = load_and_chunk_book()
    if not chunks:
        raise RuntimeError("No chunks produced from the book file — is it empty?")

    client = OpenAI(api_key=settings.openai_api_key)
    embeddings = _embed_texts(client, [c.text for c in chunks])

    chroma_client = chromadb.PersistentClient(path=str(settings.vector_store_dir))
    # Drop and recreate so re-indexing never mixes stale + fresh chunks.
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma_client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[{"chapter": c.chapter} for c in chunks],
    )
    return len(chunks)


if __name__ == "__main__":
    n = build_index()
    print(f"Indexed {n} chunks into '{settings.vector_store_dir}'.")
