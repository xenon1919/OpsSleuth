"""Live smoke tests against real Vertex AI + local Chroma/BM25 indexes.

Excluded from the default test run (see `addopts` in pyproject.toml). Run explicitly
with `pytest -m integration`. Requires `opssleuth-ingest` to have been run first
(populates data/chroma and data/artifacts) and working Vertex AI credentials.
"""

import pytest

from opssleuth.api.app import app
from opssleuth.pipelines.basic import run_basic_rag
from opssleuth.pipelines.corrective import run_corrective_rag

pytestmark = pytest.mark.integration


def test_basic_rag_answers_a_known_question():
    result = run_basic_rag("What is the current payment API readiness endpoint and port?", top_k=5)

    assert result.answer
    assert "DOC-017" in result.cited_document_ids or "DOC-001" in result.cited_document_ids
    assert any(c.document_id in ("DOC-001", "DOC-017") for c in result.retrieved_chunks)


def test_corrective_rag_abstains_when_evidence_is_genuinely_missing():
    result = run_corrective_rag("What exact signing key should I use for the production JWT?", top_k=5)

    assert result.final_sufficiency == "insufficient"
    answer_lower = result.answer.lower()
    assert "insufficient" in answer_lower or "cannot" in answer_lower or "do not" in answer_lower


def test_investigate_endpoint_end_to_end():
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post(
        "/investigate",
        json={
            "question": "What is the current payment API readiness endpoint and port?",
            "pipeline": "basic",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["retrieved_chunks"]
