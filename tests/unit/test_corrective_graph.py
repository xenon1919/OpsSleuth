from opssleuth.domain.models import QueryRewrite, RetrievalGrade, RetrievedChunk
from opssleuth.graph import corrective as corrective_module
from opssleuth.graph.corrective import _route_after_grade, run_corrective_retrieval


def make_chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, document_id=chunk_id.split("::")[0], text="t", score=1.0, metadata={}
    )


def sufficient_grade() -> RetrievalGrade:
    return RetrievalGrade(chunk_grades=[], sufficiency="sufficient", missing_evidence="")


def insufficient_grade(missing: str = "need more") -> RetrievalGrade:
    return RetrievalGrade(chunk_grades=[], sufficiency="insufficient", missing_evidence=missing)


def test_route_after_grade_done_when_sufficient():
    state = {"grade": sufficient_grade(), "attempt": 1, "max_attempts": 3}
    assert _route_after_grade(state) == "done"


def test_route_after_grade_done_when_attempts_exhausted():
    state = {"grade": insufficient_grade(), "attempt": 3, "max_attempts": 3}
    assert _route_after_grade(state) == "done"


def test_route_after_grade_rewrite_when_insufficient_and_attempts_remain():
    state = {"grade": insufficient_grade(), "attempt": 1, "max_attempts": 3}
    assert _route_after_grade(state) == "rewrite"


def _patch_retrieval(monkeypatch, grades_by_attempt, rewritten_queries):
    """grades_by_attempt: list of RetrievalGrade, one per retrieve+grade call.
    rewritten_queries: list of rewritten query strings, one per rewrite call."""
    call_count = {"retrieve": 0, "grade": 0, "rewrite": 0}

    def fake_hybrid_retrieve(query, top_k, candidate_k):
        return [make_chunk(f"DOC-{call_count['retrieve']}")]

    def fake_rerank(query, chunks, top_k):
        return chunks

    def fake_grade_retrieval(question, chunks):
        grade = grades_by_attempt[call_count["grade"]]
        call_count["grade"] += 1
        return grade

    def fake_rewrite_query(question, missing_evidence, previous_queries):
        query = rewritten_queries[call_count["rewrite"]]
        call_count["rewrite"] += 1
        return QueryRewrite(rewritten_query=query, rationale="because")

    def counted_retrieve(query, top_k, candidate_k):
        call_count["retrieve"] += 1
        return fake_hybrid_retrieve(query, top_k, candidate_k)

    monkeypatch.setattr(corrective_module, "hybrid_retrieve", counted_retrieve)
    monkeypatch.setattr(corrective_module, "rerank", fake_rerank)
    monkeypatch.setattr(corrective_module, "grade_retrieval", fake_grade_retrieval)
    monkeypatch.setattr(corrective_module, "rewrite_query", fake_rewrite_query)
    return call_count


def test_stops_immediately_when_first_attempt_is_sufficient(monkeypatch):
    call_count = _patch_retrieval(monkeypatch, [sufficient_grade()], [])

    final_state = run_corrective_retrieval("why did it fail?", max_attempts=3)

    assert call_count["retrieve"] == 1
    assert call_count["rewrite"] == 0
    assert len(final_state["history"]) == 1
    assert final_state["grade"].sufficiency == "sufficient"


def test_rewrites_once_then_succeeds(monkeypatch):
    call_count = _patch_retrieval(
        monkeypatch,
        [insufficient_grade("missing X"), sufficient_grade()],
        ["better query"],
    )

    final_state = run_corrective_retrieval("why did it fail?", max_attempts=3)

    assert call_count["retrieve"] == 2
    assert call_count["rewrite"] == 1
    assert final_state["search_query"] == "better query"
    assert len(final_state["history"]) == 2
    assert final_state["history"][0].sufficiency == "insufficient"
    assert final_state["history"][1].sufficiency == "sufficient"


def test_stops_after_max_attempts_even_if_still_insufficient(monkeypatch):
    call_count = _patch_retrieval(
        monkeypatch,
        [insufficient_grade(), insufficient_grade()],
        ["retry 1"],
    )

    final_state = run_corrective_retrieval("why did it fail?", max_attempts=2)

    assert call_count["retrieve"] == 2
    assert call_count["rewrite"] == 1
    assert final_state["grade"].sufficiency == "insufficient"
    assert len(final_state["history"]) == 2
