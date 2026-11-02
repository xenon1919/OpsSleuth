from opssleuth.domain.models import QueryRewrite
from opssleuth.llm.gemini import generate_json

_REWRITE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "rewritten_query": {"type": "STRING"},
        "rationale": {"type": "STRING"},
    },
    "required": ["rewritten_query", "rationale"],
}

REWRITER_SYSTEM_PROMPT = """You rewrite search queries for an incident-investigation retrieval \
system. You are given the user's original question, what evidence is still missing after a \
retrieval attempt, and the queries already tried. Produce ONE new search query that is more \
likely to retrieve the missing evidence: pull in specific terms the missing-evidence description \
implies (error codes, config names, build/version numbers, component names) rather than just \
rephrasing the question. Do not repeat a query already tried."""


def _build_rewrite_prompt(
    question: str, missing_evidence: str, previous_queries: list[str]
) -> str:
    tried = "\n".join(f"- {q}" for q in previous_queries) or "(none yet)"
    return (
        f"Original question: {question}\n\n"
        f"Missing evidence after the last retrieval attempt: {missing_evidence}\n\n"
        f"Queries already tried:\n{tried}\n\n"
        "Produce a new search query."
    )


def rewrite_query(question: str, missing_evidence: str, previous_queries: list[str]) -> QueryRewrite:
    prompt = _build_rewrite_prompt(question, missing_evidence, previous_queries)
    raw = generate_json(prompt, response_schema=_REWRITE_SCHEMA, system=REWRITER_SYSTEM_PROMPT)
    return QueryRewrite.model_validate(raw)
