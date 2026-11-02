from opssleuth.domain.models import RetrievedChunk
from opssleuth.grading import faithfulness_grader as grader_module
from opssleuth.grading.faithfulness_grader import _build_verification_prompt, verify_faithfulness


def make_chunk(chunk_id: str) -> RetrievedChunk:
    doc_id = chunk_id.split("::")[0]
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        text=f"text for {chunk_id}",
        score=0.9,
        metadata={"document_id": doc_id},
    )


def test_build_verification_prompt_includes_answer_and_evidence():
    chunks = [make_chunk("DOC-1::chunk0")]
    prompt = _build_verification_prompt("The pool was exhausted [DOC-1].", chunks)
    assert "The pool was exhausted [DOC-1]." in prompt
    assert "text for DOC-1::chunk0" in prompt


def test_verify_faithfulness_all_supported_is_faithful(monkeypatch):
    chunks = [make_chunk("DOC-1::chunk0")]
    fake_response = {
        "claims": [
            {
                "claim": "The pool was exhausted",
                "verdict": "supported",
                "supporting_chunk_ids": ["DOC-1::chunk0"],
                "reason": "directly stated",
            }
        ]
    }
    monkeypatch.setattr(grader_module, "generate_json", lambda *a, **k: fake_response)

    report = verify_faithfulness("The pool was exhausted [DOC-1].", chunks)

    assert report.is_faithful is True
    assert report.unsupported_claims == []


def test_verify_faithfulness_flags_unsupported_claims(monkeypatch):
    chunks = [make_chunk("DOC-1::chunk0")]
    fake_response = {
        "claims": [
            {
                "claim": "The pool was exhausted",
                "verdict": "supported",
                "supporting_chunk_ids": ["DOC-1::chunk0"],
                "reason": "directly stated",
            },
            {
                "claim": "The on-call engineer was notified within 2 minutes",
                "verdict": "unsupported",
                "supporting_chunk_ids": [],
                "reason": "no evidence mentions notification timing",
            },
        ]
    }
    monkeypatch.setattr(grader_module, "generate_json", lambda *a, **k: fake_response)

    report = verify_faithfulness("...", chunks)

    assert report.is_faithful is False
    assert len(report.unsupported_claims) == 1
    assert report.unsupported_claims[0].claim == "The on-call engineer was notified within 2 minutes"


def test_verify_faithfulness_empty_claims_is_vacuously_faithful(monkeypatch):
    monkeypatch.setattr(grader_module, "generate_json", lambda *a, **k: {"claims": []})

    report = verify_faithfulness("Insufficient evidence to answer.", [make_chunk("DOC-1::chunk0")])

    assert report.is_faithful is True
    assert report.claims == []
