from opssleuth.domain.models import Document, TelemetryEvent
from opssleuth.ingestion.chunking import chunk_document, telemetry_to_chunk


def make_doc(word_count: int) -> Document:
    return Document(
        document_id="DOC-TEST",
        title="Test doc",
        document_type="runbook",
        last_updated="2026-01-01",
        version="v1",
        status="authoritative",
        source="test",
        text=" ".join(f"word{i}" for i in range(word_count)),
    )


def test_chunk_document_single_chunk_for_short_text():
    doc = make_doc(100)
    chunks = chunk_document(doc, chunk_size_words=450, overlap_words=75)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "DOC-TEST::chunk0"
    assert chunks[0].document_id == "DOC-TEST"


def test_chunk_document_splits_long_text_with_overlap():
    doc = make_doc(1000)
    chunks = chunk_document(doc, chunk_size_words=450, overlap_words=75)
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i
    # overlap: last word of chunk 0 word window should reappear near start of chunk 1
    words0 = chunks[0].text.split()
    words1 = chunks[1].text.split()
    assert words0[-1] in words1


def test_telemetry_to_chunk():
    event = TelemetryEvent(
        event_id="EVT-1",
        timestamp="2026-01-01T00:00:00Z",
        service="payment-api",
        environment="production",
        event_type="readiness_probe_failure",
        details={"status_code": 404},
        severity="warning",
    )
    chunk = telemetry_to_chunk(event)
    assert chunk.source_type == "telemetry"
    assert "EVT-1" in chunk.text
    assert "404" in chunk.text
