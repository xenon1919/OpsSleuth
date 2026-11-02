import math

from opssleuth.domain.models import RetrievedChunk


def distinct_doc_ids(chunks: list[RetrievedChunk]) -> list[str]:
    """Ranked, deduplicated document ids from a chunk list (first occurrence wins).
    Retrieval metrics are document-level per the dataset's gold set, not chunk-level -
    a hit is any chunk from a relevant document, so duplicate chunks from the same doc
    shouldn't double-count or shift rank."""
    seen: list[str] = []
    for chunk in chunks:
        if chunk.document_id not in seen:
            seen.append(chunk.document_id)
    return seen


def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str]) -> float:
    """Fraction of the gold relevant documents that appear anywhere in the ranked
    retrieved list (not just the top k - the list passed in is already the retrieved
    set, whatever size it is)."""
    if not relevant_doc_ids:
        return 0.0
    hits = len(set(retrieved_doc_ids) & relevant_doc_ids)
    return hits / len(relevant_doc_ids)


def mrr(retrieved_doc_ids: list[str], relevant_doc_ids: set[str]) -> float:
    """Reciprocal rank of the first relevant document in the ranked list, 0 if none."""
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    """Binary-relevance NDCG@k: how close the ranking of relevant documents within the
    top k is to the ideal ranking (all relevant documents first)."""
    ranked = retrieved_doc_ids[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked, start=1)
        if doc_id in relevant_doc_ids
    )
    ideal_hits = min(len(relevant_doc_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def context_precision(retrieved_doc_ids: list[str], relevant_doc_ids: set[str]) -> float:
    """Fraction of the retrieved documents that are actually relevant."""
    if not retrieved_doc_ids:
        return 0.0
    distinct = set(retrieved_doc_ids)
    hits = len(distinct & relevant_doc_ids)
    return hits / len(distinct)
