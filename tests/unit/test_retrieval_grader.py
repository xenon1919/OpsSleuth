from opssleuth.domain.models import RetrievedChunk
from opssleuth.grading import retrieval_grader as grader_module
from opssleuth.grading.retrieval_grader import _build_grading_prompt, grade_retrieval


def make_chunk(chunk_id: str) -> RetrievedChunk:
    doc_id = chunk_id.split("::")[0]
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        text=f"text for {chunk_id}",
        score=0.9,
        metadata={"document_id": doc_id},
    )


def test_grade_retrieval_with_no_chunks_is_insufficient_without_calling_llm(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("should not call the LLM when there are no chunks")

    monkeypatch.setattr(grader_module, "generate_json", fail)

    grade = grade_retrieval("why did it fail?", [])

    assert grade.sufficiency == "insufficient"
    assert grade.chunk_grades == []
    assert grade.relevant_chunk_ids == []


def test_build_grading_prompt_lists_all_chunk_ids():
    chunks = [make_chunk("DOC-1::chunk0"), make_chunk("EVT-2::chunk0")]
    prompt = _build_grading_prompt("why did it fail?", chunks)
    assert "DOC-1::chunk0" in prompt
    assert "EVT-2::chunk0" in prompt
    assert "why did it fail?" in prompt


def test_grade_retrieval_parses_llm_response(monkeypatch):
    chunks = [make_chunk("DOC-1::chunk0"), make_chunk("DOC-2::chunk0")]
    fake_response = {
        "chunk_grades": [
            {"chunk_id": "DOC-1::chunk0", "verdict": "relevant", "reason": "directly answers it"},
            {"chunk_id": "DOC-2::chunk0", "verdict": "irrelevant", "reason": "different incident"},
        ],
        "sufficiency": "sufficient",
        "missing_evidence": "",
    }
    monkeypatch.setattr(grader_module, "generate_json", lambda *a, **k: fake_response)

    grade = grade_retrieval("why did it fail?", chunks)

    assert grade.sufficiency == "sufficient"
    assert grade.relevant_chunk_ids == ["DOC-1::chunk0"]
    assert len(grade.chunk_grades) == 2
