import chromadb

from opssleuth.core.config import settings
from opssleuth.domain.models import Chunk


class VectorStore:
    def __init__(self, persist_dir: str | None = None, collection_name: str | None = None):
        self.client = chromadb.PersistentClient(path=persist_dir or settings.chroma_persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name or settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.to_metadata() for c in chunks],
        )

    def query(self, query_embedding: list[float], top_k: int = 10) -> list[dict]:
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        hits = []
        for chunk_id, text, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": metadata,
                    "distance": distance,
                    "score": 1 - distance,
                }
            )
        return hits

    def get_by_ids(self, chunk_ids: list[str]) -> dict[str, dict]:
        """Fetch text/metadata for known chunk ids, e.g. to backfill metadata for
        hits that came from a retriever (like BM25) that doesn't store it."""
        if not chunk_ids:
            return {}
        result = self.collection.get(ids=chunk_ids, include=["documents", "metadatas"])
        return {
            chunk_id: {"text": text, "metadata": metadata}
            for chunk_id, text, metadata in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        }

    def count(self) -> int:
        return self.collection.count()
