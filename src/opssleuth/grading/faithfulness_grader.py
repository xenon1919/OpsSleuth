from opssleuth.domain.models import ClaimVerdict, FaithfulnessReport, RetrievedChunk
from opssleuth.llm.gemini import generate_json
from opssleuth.retrieval.formatting import format_context

VERIFIER_SYSTEM_PROMPT = """You are a strict fact-checker for an incident-investigation RAG \
system. You are given a generated answer and the evidence chunks (documentation excerpts and \
telemetry events, each labeled with its id in square brackets) that the answer was supposed to \
be grounded in.

Break the answer down into its individual atomic factual claims (one specific fact per claim - \
split compound sentences). For each claim, decide:
- "supported": the evidence chunks directly state or clearly imply this claim. List which \
chunk id(s) actually support it in "supporting_chunk_ids" - don't just trust the answer's own \
citation, check the evidence yourself.
- "unsupported": the claim is not stated or implied by any evidence chunk, even if it sounds \
plausible or the answer cited a chunk for it. This includes outside/world knowledge not present \
in the evidence, and claims that overstate or misread what a chunk actually says.

Ignore purely organizational phrases ("the evidence shows", "in summary") - only grade concrete \
factual claims. If the answer explicitly abstains (states the evidence is insufficient) and makes \
no other factual claims, return an empty claims list."""

_VERIFY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "claims": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "claim": {"type": "STRING"},
                    "verdict": {"type": "STRING", "enum": ["supported", "unsupported"]},
                    "supporting_chunk_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "reason": {"type": "STRING"},
                },
                "required": ["claim", "verdict", "supporting_chunk_ids", "reason"],
            },
        },
    },
    "required": ["claims"],
}


def _build_verification_prompt(answer: str, chunks: list[RetrievedChunk]) -> str:
    context = format_context(chunks)
    return f"Evidence:\n{context}\n\nGenerated answer to verify:\n{answer}\n\nList and grade every atomic claim in the answer."


def verify_faithfulness(answer: str, chunks: list[RetrievedChunk]) -> FaithfulnessReport:
    """Hallucination detection: decompose the answer into claims and check each one
    against the evidence independently, rather than trusting the generator's own
    inline citations (which the citation regex alone can't verify)."""
    prompt = _build_verification_prompt(answer, chunks)
    raw = generate_json(prompt, response_schema=_VERIFY_SCHEMA, system=VERIFIER_SYSTEM_PROMPT)
    return FaithfulnessReport(claims=[ClaimVerdict.model_validate(c) for c in raw["claims"]])
