from opssleuth.core.logging import get_logger
from opssleuth.domain.models import RagAnswer
from opssleuth.pipelines.common import build_rag_answer
from opssleuth.retrieval.hybrid import hybrid_retrieve

logger = get_logger(__name__)


def run_hybrid_rag(question: str, top_k: int = 5, candidate_k: int = 20) -> RagAnswer:
    """BM25 + dense retrieval fused with Reciprocal Rank Fusion, then a grounded
    answer. No reranking yet (milestone 3)."""
    logger.info("hybrid_rag: retrieving for question=%r", question)
    chunks = hybrid_retrieve(question, top_k=top_k, candidate_k=candidate_k)
    return build_rag_answer(question, chunks, pipeline="hybrid")
