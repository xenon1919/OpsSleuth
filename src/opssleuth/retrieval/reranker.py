from sentence_transformers import CrossEncoder

from opssleuth.core.config import settings
from opssleuth.core.logging import get_logger
from opssleuth.domain.models import RetrievedChunk

logger = get_logger(__name__)

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        logger.info("Loading reranker model %s (first use downloads it)", settings.reranker_model)
        _model = CrossEncoder(settings.reranker_model)
    return _model


def rerank(question: str, chunks: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
    """Cross-encoder reranking: scores each (question, chunk) pair jointly, which is
    far more precise than the independent similarity/BM25 scores used to assemble the
    candidate set, at the cost of being too slow to run over the whole corpus."""
    if not chunks:
        return []

    model = _get_model()
    pairs = [(question, chunk.text) for chunk in chunks]
    raw_scores = model.predict(pairs)

    reranked = sorted(zip(chunks, raw_scores), key=lambda pair: pair[1], reverse=True)
    return [
        chunk.model_copy(update={"score": float(score)})
        for chunk, score in reranked[:top_k]
    ]
