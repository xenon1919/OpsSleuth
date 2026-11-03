from fastapi import FastAPI, HTTPException

from opssleuth.api.schemas import (
    ChunkRef,
    FaithfulnessSummary,
    InvestigateRequest,
    InvestigateResponse,
    RetrievalAttempt,
)
from opssleuth.core.logging import get_logger
from opssleuth.domain.models import CorrectiveRagAnswer
from opssleuth.grading.faithfulness_grader import verify_faithfulness
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

app = FastAPI(
    title="OpsSleuth",
    description="Agentic incident-investigation engine with Corrective RAG.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/pipelines")
def list_pipelines() -> list[str]:
    return sorted(PIPELINES)


def _build_response(result, faithfulness_report=None) -> InvestigateResponse:
    attempts = None
    final_sufficiency = None
    if isinstance(result, CorrectiveRagAnswer):
        attempts = [
            RetrievalAttempt(
                attempt=a.attempt,
                query=a.query,
                sufficiency=a.sufficiency,
                missing_evidence=a.missing_evidence,
            )
            for a in result.attempts
        ]
        final_sufficiency = result.final_sufficiency

    faithfulness = None
    if faithfulness_report is not None:
        faithfulness = FaithfulnessSummary(
            is_faithful=faithfulness_report.is_faithful,
            num_claims=len(faithfulness_report.claims),
            num_unsupported_claims=len(faithfulness_report.unsupported_claims),
            unsupported_claims=[c.claim for c in faithfulness_report.unsupported_claims],
        )

    return InvestigateResponse(
        question=result.question,
        pipeline=result.pipeline,
        answer=result.answer,
        cited_document_ids=result.cited_document_ids,
        retrieved_chunks=[
            ChunkRef(chunk_id=c.chunk_id, document_id=c.document_id, score=c.score)
            for c in result.retrieved_chunks
        ],
        attempts=attempts,
        final_sufficiency=final_sufficiency,
        faithfulness=faithfulness,
    )


@app.post("/investigate", response_model=InvestigateResponse)
def investigate(request: InvestigateRequest) -> InvestigateResponse:
    run_pipeline = PIPELINES.get(request.pipeline)
    if run_pipeline is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown pipeline '{request.pipeline}'. Choose from {sorted(PIPELINES)}.",
        )

    logger.info("investigate: pipeline=%s question=%r", request.pipeline, request.question)
    result = run_pipeline(request.question, top_k=request.top_k)

    faithfulness_report = None
    if request.verify:
        faithfulness_report = verify_faithfulness(result.answer, result.retrieved_chunks)

    return _build_response(result, faithfulness_report)
