from typing import TypedDict

from langgraph.graph import END, StateGraph

from opssleuth.core.logging import get_logger
from opssleuth.domain.models import RetrievalGrade, RetrievedChunk, RewriteAttempt
from opssleuth.grading.retrieval_grader import grade_retrieval
from opssleuth.retrieval.hybrid import hybrid_retrieve
from opssleuth.retrieval.query_rewrite import rewrite_query
from opssleuth.retrieval.reranker import rerank

logger = get_logger(__name__)


class CorrectiveState(TypedDict, total=False):
    question: str
    search_query: str
    top_k: int
    candidate_k: int
    attempt: int
    max_attempts: int
    chunks: list[RetrievedChunk]
    grade: RetrievalGrade
    previous_queries: list[str]
    history: list[RewriteAttempt]


def _retrieve_node(state: CorrectiveState) -> dict:
    query = state["search_query"]
    logger.info("corrective: attempt %d retrieving for query=%r", state["attempt"], query)
    candidates = hybrid_retrieve(query, top_k=state["candidate_k"], candidate_k=state["candidate_k"])
    chunks = rerank(query, candidates, top_k=state["top_k"])
    return {"chunks": chunks}


def _grade_node(state: CorrectiveState) -> dict:
    # Always grade against the ORIGINAL question - that's what must ultimately be
    # answered, regardless of what search query produced the candidates.
    grade = grade_retrieval(state["question"], state["chunks"])
    history = state["history"] + [
        RewriteAttempt(
            attempt=state["attempt"],
            query=state["search_query"],
            sufficiency=grade.sufficiency,
            missing_evidence=grade.missing_evidence,
        )
    ]
    logger.info("corrective: attempt %d sufficiency=%s", state["attempt"], grade.sufficiency)
    return {"grade": grade, "history": history}


def _route_after_grade(state: CorrectiveState) -> str:
    if state["grade"].sufficiency == "sufficient":
        return "done"
    if state["attempt"] >= state["max_attempts"]:
        return "done"
    return "rewrite"


def _rewrite_node(state: CorrectiveState) -> dict:
    rewrite = rewrite_query(
        question=state["question"],
        missing_evidence=state["grade"].missing_evidence,
        previous_queries=state["previous_queries"],
    )
    logger.info("corrective: rewriting query -> %r (%s)", rewrite.rewritten_query, rewrite.rationale)
    return {
        "search_query": rewrite.rewritten_query,
        "previous_queries": state["previous_queries"] + [rewrite.rewritten_query],
        "attempt": state["attempt"] + 1,
    }


_graph = None


def _build_graph():
    graph = StateGraph(CorrectiveState)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("grade", _grade_node)
    graph.add_node("rewrite", _rewrite_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", _route_after_grade, {"done": END, "rewrite": "rewrite"})
    graph.add_edge("rewrite", "retrieve")

    return graph.compile()


def _get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


def run_corrective_retrieval(
    question: str, top_k: int = 5, candidate_k: int = 20, max_attempts: int = 2
) -> CorrectiveState:
    """Retrieve -> grade -> (if insufficient and attempts remain) rewrite query and
    retry. Grading is always against the original question; only the search query
    fed to retrieval changes across attempts."""
    initial_state: CorrectiveState = {
        "question": question,
        "search_query": question,
        "top_k": top_k,
        "candidate_k": candidate_k,
        "attempt": 1,
        "max_attempts": max_attempts,
        "previous_queries": [question],
        "history": [],
    }
    return _get_graph().invoke(initial_state)
