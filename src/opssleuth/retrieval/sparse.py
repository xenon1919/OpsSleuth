from pathlib import Path

from opssleuth.core.config import settings
from opssleuth.domain.models import RetrievedChunk
from opssleuth.retrieval.bm25_index import BM25Index

_index: BM25Index | None = None


def _get_index() -> BM25Index:
    global _index
    if _index is None:
        _index = BM25Index.load(Path(settings.artifacts_dir) / "bm25_index.pkl")
    return _index


def sparse_retrieve(question: str, top_k: int = 10) -> list[RetrievedChunk]:
    hits = _get_index().query(question, top_k=top_k)
    return [
        RetrievedChunk(
            chunk_id=hit["chunk_id"],
            document_id=hit["chunk_id"].split("::")[0],
            text=hit["text"],
            score=hit["score"],
            metadata={},
        )
        for hit in hits
    ]
