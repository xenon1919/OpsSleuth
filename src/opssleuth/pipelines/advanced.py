from opssleuth.core.logging import get_logger
from opssleuth.domain.models import RagAnswer
from opssleuth.pipelines.common import build_rag_answer
from opssleuth.retrieval.hybrid import hybrid_retrieve
from opssleuth.retrieval.reranker import rerank

logger = get_logger(__name__)


def run_advanced_rag(question: str, top_k: int = 5, candidate_k: int = 20) -> RagAnswer:
    """Hybrid retrieval (BM25 + dense, RRF-fused) widened to candidate_k, then
    cross-encoder reranked down to top_k. This is the "Advanced RAG" baseline that
    the corrective-retrieval loop (milestone 5+) gets compared against."""
    logger.info("advanced_rag: retrieving for question=%r", question)
    candidates = hybrid_retrieve(question, top_k=candidate_k, candidate_k=candidate_k)
    chunks = rerank(question, candidates, top_k=top_k)
    return build_rag_answer(question, chunks, pipeline="advanced")
