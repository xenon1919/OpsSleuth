from pydantic import BaseModel, Field


class InvestigateRequest(BaseModel):
    question: str = Field(..., min_length=1)
    pipeline: str = "corrective"
    top_k: int = Field(default=5, ge=1, le=20)
    verify: bool = False


class ChunkRef(BaseModel):
    chunk_id: str
    document_id: str
    score: float


class RetrievalAttempt(BaseModel):
    attempt: int
    query: str
    sufficiency: str
    missing_evidence: str = ""


class FaithfulnessSummary(BaseModel):
    is_faithful: bool
    num_claims: int
    num_unsupported_claims: int
    unsupported_claims: list[str]


class InvestigateResponse(BaseModel):
    question: str
    pipeline: str
    answer: str
    cited_document_ids: list[str]
    retrieved_chunks: list[ChunkRef]
    # Only populated for the corrective pipeline.
    attempts: list[RetrievalAttempt] | None = None
    final_sufficiency: str | None = None
    # Only populated when the request asked for verify=true.
    faithfulness: FaithfulnessSummary | None = None
