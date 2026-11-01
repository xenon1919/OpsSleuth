from opssleuth.domain.models import RetrievedChunk
from opssleuth.retrieval import reranker as reranker_module
from opssleuth.retrieval.reranker import rerank


def make_chunk(chunk_id: str, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=chunk_id.split("::")[0],
        text=f"text for {chunk_id}",
        score=score,
        metadata={},
    )


class FakeCrossEncoder:
    def __init__(self, scores_by_chunk_id: dict[str, float]):
        self.scores_by_chunk_id = scores_by_chunk_id

    def predict(self, pairs):
        return [self.scores_by_chunk_id[text.split(" ")[-1]] for _, text in pairs]


def test_rerank_reorders_by_cross_encoder_score(monkeypatch):
    chunks = [make_chunk("A", score=0.9), make_chunk("B", score=0.1), make_chunk("C", score=0.5)]
    # Cross-encoder disagrees with the input order/scores: C should end up first.
    fake_model = FakeCrossEncoder({"A": 0.1, "B": 0.2, "C": 0.9})
    monkeypatch.setattr(reranker_module, "_get_model", lambda: fake_model)

    results = rerank("some question", chunks, top_k=3)

    assert [r.chunk_id for r in results] == ["C", "B", "A"]
    assert results[0].score == 0.9


def test_rerank_truncates_to_top_k(monkeypatch):
    chunks = [make_chunk("A"), make_chunk("B"), make_chunk("C")]
    fake_model = FakeCrossEncoder({"A": 0.1, "B": 0.5, "C": 0.9})
    monkeypatch.setattr(reranker_module, "_get_model", lambda: fake_model)

    results = rerank("q", chunks, top_k=2)

    assert len(results) == 2
    assert [r.chunk_id for r in results] == ["C", "B"]


def test_rerank_empty_input_returns_empty():
    assert rerank("q", [], top_k=5) == []
