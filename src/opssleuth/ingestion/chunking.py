from opssleuth.domain.models import Chunk, Document, TelemetryEvent


def chunk_document(doc: Document, chunk_size_words: int, overlap_words: int) -> list[Chunk]:
    words = doc.text.split()
    if not words:
        return []

    step = max(chunk_size_words - overlap_words, 1)
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(words):
        piece = words[start : start + chunk_size_words]
        chunks.append(
            Chunk(
                chunk_id=f"{doc.document_id}::chunk{index}",
                document_id=doc.document_id,
                source_type="document",
                chunk_index=index,
                text=" ".join(piece),
                title=doc.title,
                document_type=doc.document_type,
                last_updated=doc.last_updated,
                version=doc.version,
                status=doc.status,
            )
        )
        index += 1
        if start + chunk_size_words >= len(words):
            break
        start += step
    return chunks


def telemetry_to_text(event: TelemetryEvent) -> str:
    detail_str = ", ".join(f"{k}={v}" for k, v in event.details.items())
    return (
        f"Telemetry event {event.event_id} at {event.timestamp}: "
        f"service={event.service} environment={event.environment} "
        f"type={event.event_type} severity={event.severity}. Details: {detail_str}."
    )


def telemetry_to_chunk(event: TelemetryEvent) -> Chunk:
    return Chunk(
        chunk_id=f"{event.event_id}::chunk0",
        document_id=event.event_id,
        source_type="telemetry",
        chunk_index=0,
        text=telemetry_to_text(event),
        title=f"{event.service} {event.event_type}",
        document_type="telemetry_event",
        last_updated=event.timestamp,
        version="",
        status=event.severity,
    )
