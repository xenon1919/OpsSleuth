from opssleuth.domain.models import RetrievedChunk
from opssleuth.retrieval import hybrid as hybrid_module
from opssleuth.retrieval.hybrid import hybrid_retrieve, reciprocal_rank_fusion


def test_rrf_rewards_consensus_over_a_single_top_rank():
    dense_ranked = ["A", "B", "C", "D", "E"]
    sparse_ranked = ["F", "B", "G", "H", "I"]
    scores = reciprocal_rank_fusion([dense_ranked, sparse_ranked], k=60)

    # B is #2 in both lists; A and F are #1 but only in one list each.
    # Being found by both retrievers should outweigh being top-ranked by just one.
    assert scores["B"] > scores["A"]
    assert scores["B"] > scores["F"]
    assert set(scores) == {"A", "B", "C", "D", "E", "F", "G", "H", "I"}


def test_rrf_empty_lists_produce_empty_scores():
    assert reciprocal_rank_fusion([[], []]) == {}


def make_chunk(chunk_id: str, score: float, with_metadata: bool) -> RetrievedChunk:
    doc_id = chunk_id.split("::")[0]
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        text=f"text for {chunk_id}",
        score=score,
        metadata={"document_id": doc_id, "title": "T"} if with_metadata else {},
    )


def test_hybrid_retrieve_fuses_and_backfills_metadata(monkeypatch):
    dense_hits = [make_chunk("DOC-1::chunk0", 0.9, with_metadata=True)]
    sparse_hits = [
        make_chunk("DOC-1::chunk0", 5.0, with_metadata=False),
        make_chunk("DOC-2::chunk0", 3.0, with_metadata=False),
    ]

    monkeypatch.setattr(hybrid_module, "dense_retrieve", lambda q, top_k: dense_hits)
    monkeypatch.setattr(hybrid_module, "sparse_retrieve", lambda q, top_k: sparse_hits)

    class FakeVectorStore:
        def get_by_ids(self, ids):
            return {
                cid: {"text": f"text for {cid}", "metadata": {"document_id": cid.split("::")[0], "title": "backfilled"}}
                for cid in ids
            }

    monkeypatch.setattr(hybrid_module, "VectorStore", FakeVectorStore)

    results = hybrid_retrieve("why did it fail?", top_k=2, candidate_k=10)

    assert [r.chunk_id for r in results] == ["DOC-1::chunk0", "DOC-2::chunk0"]
    # DOC-1 already had metadata from dense; DOC-2 only came from sparse and must be backfilled.
    assert results[0].metadata["title"] == "T"
    assert results[1].metadata["title"] == "backfilled"
    # scores are fused RRF scores, not the raw dense/sparse scores.
    assert results[0].score != 0.9
