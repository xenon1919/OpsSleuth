from opssleuth.core.logging import get_logger
from opssleuth.domain.models import CorrectiveRagAnswer
from opssleuth.graph.corrective import run_corrective_retrieval
from opssleuth.pipelines.common import generate_grounded_answer

logger = get_logger(__name__)


def run_corrective_rag(
    question: str, top_k: int = 5, candidate_k: int = 20, max_attempts: int = 2
) -> CorrectiveRagAnswer:
    """Corrective RAG: hybrid retrieval + reranking, graded for sufficiency, with
    query rewriting and retries when evidence is insufficient (milestone 5). Falls
    through to generation using whatever was retrieved on the final attempt, so the
    model's own "insufficient evidence" abstention rule is the last line of defense
    if correction still doesn't find enough."""
    final_state = run_corrective_retrieval(
        question, top_k=top_k, candidate_k=candidate_k, max_attempts=max_attempts
    )
    chunks = final_state["chunks"]
    answer, cited_ids = generate_grounded_answer(question, chunks)

    return CorrectiveRagAnswer(
        question=question,
        answer=answer,
        pipeline="corrective",
        cited_document_ids=cited_ids,
        retrieved_chunks=chunks,
        attempts=final_state["history"],
        final_sufficiency=final_state["grade"].sufficiency,
    )
