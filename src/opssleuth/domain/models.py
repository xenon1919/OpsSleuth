from typing import Any, Literal

from pydantic import BaseModel


class Document(BaseModel):
    document_id: str
    title: str
    document_type: str
    last_updated: str
    version: str
    status: str
    source: str
    text: str


class TelemetryEvent(BaseModel):
    event_id: str
    timestamp: str
    service: str
    environment: str
    event_type: str
    details: dict[str, Any]
    severity: str


class EvalQuestion(BaseModel):
    question_id: str
    question: str
    task_type: str
    relevant_document_ids: list[str]
    reference_answer: str
    expected_confidence: str
    challenge_type: str


class Chunk(BaseModel):
    """A retrievable unit: either a slice of a Document or a whole TelemetryEvent."""

    chunk_id: str
    document_id: str
    source_type: Literal["document", "telemetry"]
    chunk_index: int
    text: str
    title: str = ""
    document_type: str = ""
    last_updated: str = ""
    version: str = ""
    status: str = ""

    def to_metadata(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_type": self.source_type,
            "chunk_index": self.chunk_index,
            "title": self.title,
            "document_type": self.document_type,
            "last_updated": self.last_updated,
            "version": self.version,
            "status": self.status,
        }


class RetrievedChunk(BaseModel):
    """A chunk returned by a retriever, with its retrieval score."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict[str, Any] = {}


class RagAnswer(BaseModel):
    """Output of a RAG pipeline run (basic, advanced, or corrective)."""

    question: str
    answer: str
    pipeline: str
    cited_document_ids: list[str]
    retrieved_chunks: list[RetrievedChunk]
