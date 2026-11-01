from opssleuth.domain.models import ChunkGrade, RetrievalGrade, RetrievedChunk
from opssleuth.llm.gemini import generate_json
from opssleuth.retrieval.formatting import format_context

GRADER_SYSTEM_PROMPT = """You are a strict retrieval-quality grader for an incident-investigation \
RAG system. You are given a question and a set of retrieved evidence chunks (documentation \
excerpts and telemetry events), each labeled with its id in square brackets.

For each chunk, decide:
- "relevant": the chunk contains information that is actually useful for answering the \
question (a direct fact, a contributing cause, corroborating telemetry, or evidence needed to \
resolve a conflict between sources).
- "irrelevant": the chunk is about a different topic, a different incident, or otherwise does \
not help answer this specific question, even if it mentions the same service name.

Then, considering ONLY the chunks you marked "relevant", decide overall:
- "sufficient": the relevant chunks, taken together, let someone confidently and completely \
answer the question.
- "insufficient": necessary information is still missing (e.g. only part of the causal chain is \
covered, or the evidence conflicts and there's no way to tell which source is authoritative), or \
no chunk was marked relevant at all.

If insufficient, describe concretely what evidence is missing in "missing_evidence" (empty \
string otherwise). Be skeptical: evidence that is merely topically related but doesn't actually \
answer the question should be graded "irrelevant", not "relevant"."""

_GRADE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "chunk_grades": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "chunk_id": {"type": "STRING"},
                    "verdict": {"type": "STRING", "enum": ["relevant", "irrelevant"]},
                    "reason": {"type": "STRING"},
                },
                "required": ["chunk_id", "verdict", "reason"],
            },
        },
        "sufficiency": {"type": "STRING", "enum": ["sufficient", "insufficient"]},
        "missing_evidence": {"type": "STRING"},
    },
    "required": ["chunk_grades", "sufficiency", "missing_evidence"],
}


def _build_grading_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = format_context(chunks)
    chunk_ids = ", ".join(chunk.chunk_id for chunk in chunks)
    return (
        f"Question: {question}\n\n"
        f"Retrieved evidence:\n{context}\n\n"
        f"Grade every chunk. Use these exact chunk_id values (one grade per id): {chunk_ids}"
    )


def grade_retrieval(question: str, chunks: list[RetrievedChunk]) -> RetrievalGrade:
    """LLM-as-judge grading of a retrieved set: per-chunk relevance, plus an overall
    sufficiency verdict. This is the signal the corrective-retrieval loop acts on -
    insufficient triggers query rewriting / alternative retrieval (milestone 5)."""
    if not chunks:
        return RetrievalGrade(
            chunk_grades=[],
            sufficiency="insufficient",
            missing_evidence="No chunks were retrieved at all.",
        )

    prompt = _build_grading_prompt(question, chunks)
    raw = generate_json(prompt, response_schema=_GRADE_SCHEMA, system=GRADER_SYSTEM_PROMPT)

    return RetrievalGrade(
        chunk_grades=[ChunkGrade.model_validate(g) for g in raw["chunk_grades"]],
        sufficiency=raw["sufficiency"],
        missing_evidence=raw.get("missing_evidence", ""),
    )
