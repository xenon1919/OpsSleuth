from opssleuth.domain.models import RetrievedChunk
from opssleuth.retrieval.embeddings import embed_query
from opssleuth.retrieval.vector_store import VectorStore

_store: VectorStore | None = None


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def dense_retrieve(question: str, top_k: int = 5) -> list[RetrievedChunk]:
    query_embedding = embed_query(question)
    hits = _get_store().query(query_embedding, top_k=top_k)
    return [
        RetrievedChunk(
            chunk_id=hit["chunk_id"],
            document_id=hit["metadata"]["document_id"],
            text=hit["text"],
            score=hit["score"],
            metadata=hit["metadata"],
        )
        for hit in hits
    ]
