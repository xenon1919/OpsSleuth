from opssleuth.domain.models import RetrievedChunk
from opssleuth.retrieval.dense import dense_retrieve
from opssleuth.retrieval.sparse import sparse_retrieve
from opssleuth.retrieval.vector_store import VectorStore

RRF_K = 60


def reciprocal_rank_fusion(ranked_id_lists: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Standard RRF: score(d) = sum over retrievers of 1 / (k + rank(d)), rank starting at 1.
    A chunk missing from a retriever's ranked list simply contributes 0 from it."""
    scores: dict[str, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def hybrid_retrieve(question: str, top_k: int = 5, candidate_k: int = 20) -> list[RetrievedChunk]:
    """Dense + BM25, fused with Reciprocal Rank Fusion. Each retriever contributes its
    own top candidate_k; the fused ranking is truncated to top_k."""
    dense_hits = dense_retrieve(question, top_k=candidate_k)
    sparse_hits = sparse_retrieve(question, top_k=candidate_k)

    by_id: dict[str, RetrievedChunk] = {c.chunk_id: c for c in dense_hits}
    for chunk in sparse_hits:
        by_id.setdefault(chunk.chunk_id, chunk)

    fused_scores = reciprocal_rank_fusion(
        [[c.chunk_id for c in dense_hits], [c.chunk_id for c in sparse_hits]]
    )

    # BM25 hits don't carry Chroma metadata; backfill it for anything dense didn't already find.
    missing_ids = [cid for cid, chunk in by_id.items() if not chunk.metadata]
    if missing_ids:
        fetched = VectorStore().get_by_ids(missing_ids)
        for chunk_id, data in fetched.items():
            chunk = by_id[chunk_id]
            by_id[chunk_id] = chunk.model_copy(
                update={
                    "metadata": data["metadata"],
                    "document_id": data["metadata"].get("document_id", chunk.document_id),
                }
            )

    ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)[:top_k]
    return [by_id[cid].model_copy(update={"score": fused_scores[cid]}) for cid in ranked_ids]
