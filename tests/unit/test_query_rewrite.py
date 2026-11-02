from opssleuth.retrieval.query_rewrite import _build_rewrite_prompt


def test_build_rewrite_prompt_includes_question_and_missing_evidence():
    prompt = _build_rewrite_prompt(
        "why did it fail?", "no telemetry for the connection pool", []
    )
    assert "why did it fail?" in prompt
    assert "no telemetry for the connection pool" in prompt
    assert "(none yet)" in prompt


def test_build_rewrite_prompt_lists_previous_queries():
    prompt = _build_rewrite_prompt("why?", "missing", ["query one", "query two"])
    assert "query one" in prompt
    assert "query two" in prompt
