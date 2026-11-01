from opssleuth.domain.models import RetrievedChunk
from opssleuth.pipelines.common import CITATION_RE, build_prompt, format_context


def make_chunk(document_id: str, text: str, **meta) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{document_id}::chunk0",
        document_id=document_id,
        text=text,
        score=0.9,
        metadata={"document_id": document_id, **meta},
    )


def test_format_context_includes_id_and_metadata():
    chunks = [make_chunk("DOC-001", "Deploy with blue-green.", title="Runbook", status="authoritative")]
    context = format_context(chunks)
    assert "[DOC-001]" in context
    assert "Runbook" in context
    assert "authoritative" in context
    assert "Deploy with blue-green." in context


def test_build_prompt_includes_question_and_context():
    chunks = [make_chunk("DOC-001", "Some evidence text.")]
    prompt = build_prompt("Why did it fail?", chunks)
    assert "Why did it fail?" in prompt
    assert "Some evidence text." in prompt


def test_citation_regex_extracts_doc_and_evt_ids():
    text = "The probe failed [EVT-1001], per [DOC-017], and again [DOC-017]."
    assert sorted(set(CITATION_RE.findall(text))) == ["DOC-017", "EVT-1001"]


def test_citation_regex_handles_multi_id_brackets():
    text = "Caused by a leak [DOC-007, DOC-024, EVT-1004]."
    assert sorted(set(CITATION_RE.findall(text))) == ["DOC-007", "DOC-024", "EVT-1004"]


def test_citation_regex_ignores_text_without_ids():
    text = "This sentence has no citation at all."
    assert CITATION_RE.findall(text) == []
