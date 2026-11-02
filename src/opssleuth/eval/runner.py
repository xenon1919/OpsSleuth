import time
from dataclasses import dataclass

from opssleuth.core.config import settings
from opssleuth.core.logging import get_logger
from opssleuth.domain.models import EvalQuestion
from opssleuth.eval.metrics import context_precision, distinct_doc_ids, mrr, ndcg_at_k, recall_at_k
from opssleuth.grading.faithfulness_grader import verify_faithfulness
from opssleuth.ingestion.loader import load_eval_questions
from opssleuth.pipelines.advanced import run_advanced_rag
from opssleuth.pipelines.basic import run_basic_rag
from opssleuth.pipelines.corrective import run_corrective_rag
from opssleuth.pipelines.hybrid import run_hybrid_rag

logger = get_logger(__name__)

PIPELINES = {
    "basic": run_basic_rag,
    "hybrid": run_hybrid_rag,
    "advanced": run_advanced_rag,
    "corrective": run_corrective_rag,
}


@dataclass
class QuestionResult:
    question_id: str
    pipeline: str
    challenge_type: str
    latency_seconds: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    context_precision: float
    is_faithful: bool | None
    num_claims: int
    num_unsupported_claims: int
    attempts: int
    final_sufficiency: str | None
    answer: str
    retrieved_doc_ids: list[str]
    relevant_doc_ids: list[str]
    error: str | None = None


def run_one(
    question: EvalQuestion, pipeline_name: str, top_k: int, check_faithfulness: bool
) -> QuestionResult:
    run_fn = PIPELINES[pipeline_name]
    relevant = set(question.relevant_document_ids)

    start = time.monotonic()
    result = run_fn(question.question, top_k=top_k)
    latency = time.monotonic() - start

    retrieved_doc_ids = distinct_doc_ids(result.retrieved_chunks)

    is_faithful = None
    num_claims = 0
    num_unsupported = 0
    if check_faithfulness:
        report = verify_faithfulness(result.answer, result.retrieved_chunks)
        is_faithful = report.is_faithful
        num_claims = len(report.claims)
        num_unsupported = len(report.unsupported_claims)

    attempts = getattr(result, "attempts", None)
    final_sufficiency = getattr(result, "final_sufficiency", None)

    return QuestionResult(
        question_id=question.question_id,
        pipeline=pipeline_name,
        challenge_type=question.challenge_type,
        latency_seconds=latency,
        recall_at_k=recall_at_k(retrieved_doc_ids, relevant),
        mrr=mrr(retrieved_doc_ids, relevant),
        ndcg_at_k=ndcg_at_k(retrieved_doc_ids, relevant, k=top_k),
        context_precision=context_precision(retrieved_doc_ids, relevant),
        is_faithful=is_faithful,
        num_claims=num_claims,
        num_unsupported_claims=num_unsupported,
        attempts=len(attempts) if attempts is not None else 1,
        final_sufficiency=final_sufficiency,
        answer=result.answer,
        retrieved_doc_ids=retrieved_doc_ids,
        relevant_doc_ids=sorted(relevant),
    )


def run_evaluation(
    pipeline_names: list[str] | None = None,
    question_ids: list[str] | None = None,
    top_k: int = 5,
    check_faithfulness: bool = True,
    on_result=None,
) -> list[QuestionResult]:
    """Run every (question, pipeline) combination and collect metrics. A single
    pipeline/question failure is recorded as an error row rather than aborting the
    whole run - a 90-minute live-API sweep shouldn't be lost to one bad call."""
    pipeline_names = pipeline_names or ["basic", "advanced", "corrective"]
    questions = load_eval_questions(settings.data_dir)
    if question_ids:
        wanted = set(question_ids)
        questions = [q for q in questions if q.question_id in wanted]

    results: list[QuestionResult] = []
    total = len(questions) * len(pipeline_names)
    done = 0
    for question in questions:
        for pipeline_name in pipeline_names:
            done += 1
            logger.info("eval [%d/%d]: %s / %s", done, total, question.question_id, pipeline_name)
            try:
                row = run_one(question, pipeline_name, top_k, check_faithfulness)
            except Exception as exc:
                logger.exception("eval: failed on %s/%s", question.question_id, pipeline_name)
                row = QuestionResult(
                    question_id=question.question_id,
                    pipeline=pipeline_name,
                    challenge_type=question.challenge_type,
                    latency_seconds=0.0,
                    recall_at_k=0.0,
                    mrr=0.0,
                    ndcg_at_k=0.0,
                    context_precision=0.0,
                    is_faithful=None,
                    num_claims=0,
                    num_unsupported_claims=0,
                    attempts=0,
                    final_sufficiency=None,
                    answer="",
                    retrieved_doc_ids=[],
                    relevant_doc_ids=sorted(question.relevant_document_ids),
                    error=str(exc),
                )
            results.append(row)
            if on_result:
                on_result(row)
    return results
