from opssleuth.domain.models import RetrievedChunk
from opssleuth.retrieval.formatting import format_context


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
