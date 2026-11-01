import re

from opssleuth.domain.models import RagAnswer, RetrievedChunk
from opssleuth.llm.gemini import generate
from opssleuth.retrieval.formatting import format_context

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
CITATION_RE = re.compile(r"(?:DOC|EVT)-[A-Za-z0-9]+")


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = format_context(chunks)
    return (
        f"Evidence:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the evidence above, with inline citations."
    )


def build_rag_answer(question: str, chunks: list[RetrievedChunk], pipeline: str) -> RagAnswer:
    """Generate a grounded answer for already-retrieved chunks and package it as a
    RagAnswer. Shared by every pipeline variant (basic, hybrid, corrective, ...) so they
    only differ in how they retrieve, not how they prompt or parse citations."""
    prompt = build_prompt(question, chunks)
    answer = generate(prompt, system=SYSTEM_PROMPT, temperature=0.0)
    cited_ids = sorted(set(CITATION_RE.findall(answer)))

    return RagAnswer(
        question=question,
        answer=answer,
        pipeline=pipeline,
        cited_document_ids=cited_ids,
        retrieved_chunks=chunks,
    )
