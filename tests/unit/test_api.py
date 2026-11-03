from fastapi.testclient import TestClient

from opssleuth.api import app as app_module
from opssleuth.domain.models import (
    ClaimVerdict,
    CorrectiveRagAnswer,
    FaithfulnessReport,
    RagAnswer,
    RetrievedChunk,
    RewriteAttempt,
)


def make_chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, document_id=chunk_id.split("::")[0], text="t", score=0.9, metadata={}
    )


def test_health():
    client = TestClient(app_module.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_pipelines():
    client = TestClient(app_module.app)
    response = client.get("/pipelines")
    assert response.status_code == 200
    assert set(response.json()) == {"basic", "hybrid", "advanced", "corrective"}


def test_investigate_unknown_pipeline_is_400():
    client = TestClient(app_module.app)
    response = client.post("/investigate", json={"question": "why?", "pipeline": "nonexistent"})
    assert response.status_code == 400


def test_investigate_basic_pipeline(monkeypatch):
    fake_answer = RagAnswer(
        question="why did it fail?",
        answer="It failed because of X [DOC-1].",
        pipeline="basic",
        cited_document_ids=["DOC-1"],
        retrieved_chunks=[make_chunk("DOC-1::chunk0")],
    )
    monkeypatch.setitem(app_module.PIPELINES, "basic", lambda question, top_k: fake_answer)

    client = TestClient(app_module.app)
    response = client.post("/investigate", json={"question": "why did it fail?", "pipeline": "basic"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "It failed because of X [DOC-1]."
    assert body["cited_document_ids"] == ["DOC-1"]
    assert body["retrieved_chunks"] == [{"chunk_id": "DOC-1::chunk0", "document_id": "DOC-1", "score": 0.9}]
    assert body["attempts"] is None
    assert body["final_sufficiency"] is None
    assert body["faithfulness"] is None


def test_investigate_corrective_pipeline_includes_attempts(monkeypatch):
    fake_answer = CorrectiveRagAnswer(
        question="why did it fail?",
        answer="Insufficient evidence.",
        pipeline="corrective",
        cited_document_ids=[],
        retrieved_chunks=[],
        attempts=[
            RewriteAttempt(attempt=1, query="why did it fail?", sufficiency="insufficient", missing_evidence="need more"),
            RewriteAttempt(attempt=2, query="rewritten query", sufficiency="insufficient", missing_evidence="still missing"),
        ],
        final_sufficiency="insufficient",
    )
    monkeypatch.setitem(app_module.PIPELINES, "corrective", lambda question, top_k: fake_answer)

    client = TestClient(app_module.app)
    response = client.post("/investigate", json={"question": "why did it fail?", "pipeline": "corrective"})

    assert response.status_code == 200
    body = response.json()
    assert body["final_sufficiency"] == "insufficient"
    assert len(body["attempts"]) == 2
    assert body["attempts"][1]["query"] == "rewritten query"


def test_investigate_with_verify_includes_faithfulness(monkeypatch):
    fake_answer = RagAnswer(
        question="why?",
        answer="X happened [DOC-1].",
        pipeline="basic",
        cited_document_ids=["DOC-1"],
        retrieved_chunks=[make_chunk("DOC-1::chunk0")],
    )
    fake_report = FaithfulnessReport(
        claims=[
            ClaimVerdict(claim="X happened", verdict="unsupported", supporting_chunk_ids=[], reason="not in evidence"),
        ]
    )
    monkeypatch.setitem(app_module.PIPELINES, "basic", lambda question, top_k: fake_answer)
    monkeypatch.setattr(app_module, "verify_faithfulness", lambda answer, chunks: fake_report)

    client = TestClient(app_module.app)
    response = client.post(
        "/investigate", json={"question": "why?", "pipeline": "basic", "verify": True}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["faithfulness"]["is_faithful"] is False
    assert body["faithfulness"]["num_unsupported_claims"] == 1
    assert body["faithfulness"]["unsupported_claims"] == ["X happened"]
