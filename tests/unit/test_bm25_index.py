from opssleuth.domain.models import Chunk
from opssleuth.retrieval.bm25_index import BM25Index


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=chunk_id.split("::")[0],
        source_type="document",
        chunk_index=0,
        text=text,
    )


def test_bm25_ranks_relevant_chunk_first():
    chunks = [
        make_chunk("DOC-1::chunk0", "The payment API readiness probe failed after deployment."),
        make_chunk("DOC-2::chunk0", "The weather forecast predicts rain tomorrow."),
        make_chunk("DOC-3::chunk0", "Database connection pool exhaustion caused timeouts."),
    ]
    index = BM25Index.build(chunks)
    results = index.query("Why did the readiness probe fail?", top_k=2)
    assert results[0]["chunk_id"] == "DOC-1::chunk0"


def test_bm25_save_and_load_roundtrip(tmp_path):
    chunks = [
        make_chunk("DOC-1::chunk0", "payment api readiness probe failure"),
        make_chunk("DOC-2::chunk0", "unrelated weather forecast report"),
        make_chunk("DOC-3::chunk0", "database connection pool exhaustion"),
    ]
    index = BM25Index.build(chunks)
    path = tmp_path / "bm25.pkl"
    index.save(path)
    loaded = BM25Index.load(path)
    results = loaded.query("readiness probe", top_k=1)
    assert results[0]["chunk_id"] == "DOC-1::chunk0"
