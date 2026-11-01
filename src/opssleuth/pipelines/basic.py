import re

from opssleuth.core.logging import get_logger
from opssleuth.domain.models import RagAnswer, RetrievedChunk
from opssleuth.llm.gemini import generate
from opssleuth.retrieval.dense import dense_retrieve

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are OpsSleuth, an incident-investigation assistant for a production \
engineering team. You answer questions using ONLY the evidence provided to you below \
(documentation excerpts and telemetry events). You never use outside knowledge.

Rules:
- Cite the document/event ID for every factual claim, inline, like [DOC-008] or [EVT-1003].
- If the evidence conflicts (e.g. two documents disagree), point out the conflict and prefer \
the source with the more recent last_updated/version, explaining why.
- If the evidence is insufficient to answer the question, say so explicitly and state what \
evidence is missing. Do not guess or invent an answer.
"""

# Matches DOC-xxx / EVT-xxx ids anywhere, including multi-id brackets like
# "[DOC-007, DOC-024, EVT-1004]" which a single-id-per-bracket pattern would miss.
_CITATION_RE = re.compile(r"(?:DOC|EVT)-[A-Za-z0-9]+")


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for chunk in chunks:
        meta = chunk.metadata
        header = f"[{chunk.document_id}] title={meta.get('title', '')} " \
            f"status={meta.get('status', '')} last_updated={meta.get('last_updated', '')} " \
            f"version={meta.get('version', '')}"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = _format_context(chunks)
    return (
        f"Evidence:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the evidence above, with inline citations."
    )


def run_basic_rag(question: str, top_k: int = 5) -> RagAnswer:
    """Dense-retrieval-only RAG: embed the question, retrieve top_k chunks, generate
    a grounded answer. No reranking, no hybrid search, no correction loop."""
    logger.info("basic_rag: retrieving for question=%r", question)
    chunks = dense_retrieve(question, top_k=top_k)

    prompt = _build_prompt(question, chunks)
    answer = generate(prompt, system=SYSTEM_PROMPT, temperature=0.0)
    cited_ids = sorted(set(_CITATION_RE.findall(answer)))

    return RagAnswer(
        question=question,
        answer=answer,
        pipeline="basic",
        cited_document_ids=cited_ids,
        retrieved_chunks=chunks,
    )
