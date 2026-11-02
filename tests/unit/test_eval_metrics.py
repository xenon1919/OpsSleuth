import math

from opssleuth.domain.models import RetrievedChunk
from opssleuth.eval.metrics import context_precision, distinct_doc_ids, mrr, ndcg_at_k, recall_at_k


def make_chunk(doc_id: str, chunk_index: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{doc_id}::chunk{chunk_index}", document_id=doc_id, text="t", score=1.0, metadata={}
    )


def test_distinct_doc_ids_dedupes_preserving_rank_order():
    chunks = [make_chunk("DOC-1"), make_chunk("DOC-2"), make_chunk("DOC-1", 1), make_chunk("DOC-3")]
    assert distinct_doc_ids(chunks) == ["DOC-1", "DOC-2", "DOC-3"]


def test_recall_at_k_full_and_partial_and_none():
    relevant = {"DOC-1", "DOC-2"}
    assert recall_at_k(["DOC-1", "DOC-2", "DOC-3"], relevant) == 1.0
    assert recall_at_k(["DOC-1", "DOC-3"], relevant) == 0.5
    assert recall_at_k(["DOC-3", "DOC-4"], relevant) == 0.0


def test_recall_at_k_with_no_relevant_docs_is_zero():
    assert recall_at_k(["DOC-1"], set()) == 0.0


def test_mrr_rewards_earlier_hits():
    relevant = {"DOC-2"}
    assert mrr(["DOC-2", "DOC-1"], relevant) == 1.0
    assert mrr(["DOC-1", "DOC-2"], relevant) == 0.5
    assert mrr(["DOC-1", "DOC-3"], relevant) == 0.0


def test_ndcg_perfect_ranking_is_one():
    relevant = {"DOC-1", "DOC-2"}
    assert ndcg_at_k(["DOC-1", "DOC-2", "DOC-3"], relevant, k=3) == 1.0


def test_ndcg_worse_ranking_scores_lower_than_perfect():
    relevant = {"DOC-1", "DOC-2"}
    perfect = ndcg_at_k(["DOC-1", "DOC-2", "DOC-3"], relevant, k=3)
    worse = ndcg_at_k(["DOC-3", "DOC-1", "DOC-2"], relevant, k=3)
    assert worse < perfect
    assert worse == (1.0 / math.log2(3) + 1.0 / math.log2(4)) / (1.0 / math.log2(2) + 1.0 / math.log2(3))


def test_ndcg_no_relevant_docs_retrieved_is_zero():
    assert ndcg_at_k(["DOC-9"], {"DOC-1"}, k=5) == 0.0


def test_context_precision():
    relevant = {"DOC-1"}
    assert context_precision(["DOC-1", "DOC-2", "DOC-3"], relevant) == 1 / 3
    assert context_precision([], relevant) == 0.0
