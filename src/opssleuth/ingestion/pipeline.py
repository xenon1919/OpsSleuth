from pathlib import Path

from opssleuth.config import settings
from opssleuth.ingestion.bm25_index import BM25Index
from opssleuth.ingestion.chunking import chunk_document, telemetry_to_chunk
from opssleuth.ingestion.embed import embed_texts
from opssleuth.ingestion.loader import load_documents, load_telemetry_events
from opssleuth.ingestion.vector_store import VectorStore
from opssleuth.models import Chunk


def build_chunks(data_dir: str | Path) -> list[Chunk]:
    documents = load_documents(data_dir)
    events = load_telemetry_events(data_dir)

    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(
            chunk_document(doc, settings.chunk_size_words, settings.chunk_overlap_words)
        )
    for event in events:
        chunks.append(telemetry_to_chunk(event))
    return chunks


def ingest_all(data_dir: str | None = None, reset: bool = True) -> dict:
    data_dir = data_dir or settings.data_dir
    chunks = build_chunks(data_dir)

    store = VectorStore()
    if reset:
        store.reset()

    embeddings = embed_texts([c.text for c in chunks], task_type="RETRIEVAL_DOCUMENT")
    store.add_chunks(chunks, embeddings)

    bm25 = BM25Index.build(chunks)
    artifacts_dir = Path(settings.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    bm25.save(artifacts_dir / "bm25_index.pkl")

    return {
        "num_chunks": len(chunks),
        "num_document_chunks": sum(1 for c in chunks if c.source_type == "document"),
        "num_telemetry_chunks": sum(1 for c in chunks if c.source_type == "telemetry"),
        "vector_store_count": store.count(),
    }
