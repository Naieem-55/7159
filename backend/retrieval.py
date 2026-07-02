"""Semantic retrieval of relevant book chunks for a given query."""
from __future__ import annotations

import chromadb
from openai import OpenAI

from backend.config import settings
from backend.indexing import COLLECTION_NAME
from backend.models import Evidence


def _get_collection():
    chroma_client = chromadb.PersistentClient(path=str(settings.vector_store_dir))
    try:
        return chroma_client.get_collection(COLLECTION_NAME)
    except Exception as exc:  # collection missing -> index was never built
        raise RuntimeError(
            "Vector index not found. Run `python -m scripts.build_index` first."
        ) from exc


def retrieve(query: str, top_k: int | None = None) -> list[Evidence]:
    """Return the top-k most relevant book chunks for a query string.

    Similarity is reported as 1 - cosine_distance (Chroma's default metric),
    so higher is more relevant.
    """
    settings.validate()
    k = top_k or settings.top_k
    collection = _get_collection()

    client = OpenAI(api_key=settings.openai_api_key)
    query_embedding = client.embeddings.create(
        model=settings.embedding_model, input=[query]
    ).data[0].embedding

    results = collection.query(query_embeddings=[query_embedding], n_results=k)

    evidence: list[Evidence] = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        evidence.append(
            Evidence(
                chapter=meta.get("chapter", "Unknown"),
                text=doc,
                similarity=round(1 - dist, 4),
            )
        )
    return evidence
