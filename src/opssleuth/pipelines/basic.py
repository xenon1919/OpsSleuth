from opssleuth.core.logging import get_logger
from opssleuth.domain.models import RagAnswer
from opssleuth.pipelines.common import build_rag_answer
from opssleuth.retrieval.dense import dense_retrieve

logger = get_logger(__name__)


def run_basic_rag(question: str, top_k: int = 5) -> RagAnswer:
    """Dense-retrieval-only RAG: embed the question, retrieve top_k chunks, generate
    a grounded answer. No hybrid search, no reranking, no correction loop."""
    logger.info("basic_rag: retrieving for question=%r", question)
    chunks = dense_retrieve(question, top_k=top_k)
    return build_rag_answer(question, chunks, pipeline="basic")
