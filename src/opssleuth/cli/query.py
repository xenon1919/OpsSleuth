import argparse
import sys

from opssleuth.core.config import settings
from opssleuth.core.logging import get_logger
from opssleuth.domain.models import CorrectiveRagAnswer
from opssleuth.grading.faithfulness_grader import verify_faithfulness
from opssleuth.grading.retrieval_grader import grade_retrieval
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


def _print_answer(question: str, result, grade: bool, verify: bool) -> None:
    print(f"\n=== {question} ===")
    print(result.answer)
    print(f"-- cited: {result.cited_document_ids}")
    print(f"-- retrieved: {[c.chunk_id for c in result.retrieved_chunks]}")

    if isinstance(result, CorrectiveRagAnswer):
        print(f"-- final sufficiency: {result.final_sufficiency}")
        for a in result.attempts:
            print(f"   attempt {a.attempt} query={a.query!r} -> {a.sufficiency}")
            if a.missing_evidence:
                print(f"      missing: {a.missing_evidence}")
    elif grade:
        retrieval_grade = grade_retrieval(question, result.retrieved_chunks)
        print(f"-- grading: sufficiency={retrieval_grade.sufficiency}")
        for g in retrieval_grade.chunk_grades:
            print(f"   [{g.verdict}] {g.chunk_id}: {g.reason}")
        if retrieval_grade.missing_evidence:
            print(f"-- missing evidence: {retrieval_grade.missing_evidence}")

    if verify:
        report = verify_faithfulness(result.answer, result.retrieved_chunks)
        print(f"-- faithfulness: {'faithful' if report.is_faithful else 'UNFAITHFUL'}")
        for c in report.claims:
            print(f"   [{c.verdict}] {c.claim}")
            if c.verdict == "unsupported":
                print(f"      reason: {c.reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a RAG pipeline against a question.")
    parser.add_argument("--question", "-q", help="Ask an ad-hoc question instead of running eval IDs.")
    parser.add_argument(
        "--pipeline",
        "-p",
        choices=sorted(PIPELINES),
        default="basic",
        help="Which pipeline variant to run.",
    )
    parser.add_argument(
        "--eval-ids",
        nargs="*",
        default=None,
        help="Question IDs from evaluation_questions.jsonl to run (default: Q-001 Q-002).",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--grade",
        action="store_true",
        help="Also run LLM-as-judge retrieval-quality grading on the retrieved chunks.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Also run LLM-as-judge claim verification (hallucination detection) on the answer.",
    )
    args = parser.parse_args()

    run_pipeline = PIPELINES[args.pipeline]

    if args.question:
        result = run_pipeline(args.question, top_k=args.top_k)
        _print_answer(args.question, result, args.grade, args.verify)
        return

    eval_ids = args.eval_ids or ["Q-001", "Q-002"]
    questions = {q.question_id: q for q in load_eval_questions(settings.data_dir)}
    for qid in eval_ids:
        eval_question = questions[qid]
        result = run_pipeline(eval_question.question, top_k=args.top_k)
        _print_answer(f"{qid}: {eval_question.question}", result, args.grade, args.verify)
        print(f"-- reference: {eval_question.reference_answer}")
        print(f"-- expected relevant docs: {eval_question.relevant_document_ids}")


if __name__ == "__main__":
    sys.exit(main())
