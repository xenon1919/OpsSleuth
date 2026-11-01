import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from opssleuth.domain.models import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, chunk_ids: list[str], chunk_texts: dict[str, str], bm25: BM25Okapi):
        self.chunk_ids = chunk_ids
        self.chunk_texts = chunk_texts
        self.bm25 = bm25

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "BM25Index":
        chunk_ids = [c.chunk_id for c in chunks]
        chunk_texts = {c.chunk_id: c.text for c in chunks}
        tokenized_corpus = [tokenize(c.text) for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        return cls(chunk_ids, chunk_texts, bm25)

    def query(self, query_text: str, top_k: int = 10) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query_text))
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {"chunk_id": cid, "text": self.chunk_texts[cid], "score": float(score)}
            for cid, score in ranked
            if score > 0
        ]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"chunk_ids": self.chunk_ids, "chunk_texts": self.chunk_texts, "bm25": self.bm25}, f)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["chunk_ids"], data["chunk_texts"], data["bm25"])
