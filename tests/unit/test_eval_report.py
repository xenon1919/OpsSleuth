from opssleuth.eval.report import format_markdown_table, summarize
from opssleuth.eval.runner import QuestionResult


def make_result(**overrides) -> QuestionResult:
    defaults = dict(
        question_id="Q-1",
        pipeline="basic",
        challenge_type="simple",
        latency_seconds=1.0,
        recall_at_k=1.0,
        mrr=1.0,
        ndcg_at_k=1.0,
        context_precision=1.0,
        is_faithful=True,
        num_claims=2,
        num_unsupported_claims=0,
        attempts=1,
        final_sufficiency=None,
        answer="answer",
        retrieved_doc_ids=["DOC-1"],
        relevant_doc_ids=["DOC-1"],
        error=None,
    )
    defaults.update(overrides)
    return QuestionResult(**defaults)


def test_summarize_averages_metrics_per_pipeline():
    results = [
        make_result(pipeline="basic", recall_at_k=1.0),
        make_result(pipeline="basic", recall_at_k=0.0),
        make_result(pipeline="advanced", recall_at_k=1.0),
    ]
    summary = summarize(results)

    assert summary["basic"]["n"] == 2
    assert summary["basic"]["recall_at_k"] == 0.5
    assert summary["advanced"]["recall_at_k"] == 1.0


def test_summarize_excludes_error_rows_from_metric_averages():
    results = [
        make_result(pipeline="basic", recall_at_k=1.0),
        make_result(pipeline="basic", error="boom", recall_at_k=0.0),
    ]
    summary = summarize(results)

    assert summary["basic"]["n"] == 2
    assert summary["basic"]["n_errors"] == 1
    assert summary["basic"]["recall_at_k"] == 1.0


def test_summarize_correction_rates_only_for_corrective_rows():
    results = [
        make_result(pipeline="corrective", attempts=1, final_sufficiency="sufficient"),
        make_result(pipeline="corrective", attempts=2, final_sufficiency="sufficient"),
        make_result(pipeline="corrective", attempts=2, final_sufficiency="insufficient"),
        make_result(pipeline="basic", attempts=1, final_sufficiency=None),
    ]
    summary = summarize(results)

    assert summary["basic"]["correction_trigger_rate"] is None
    # 2 of 3 corrective rows needed more than one attempt.
    assert summary["corrective"]["correction_trigger_rate"] == 2 / 3
    # of the 2 that needed correction, 1 ended up sufficient.
    assert summary["corrective"]["correction_success_rate"] == 1 / 2


def test_format_markdown_table_includes_all_pipelines():
    summary = summarize([make_result(pipeline="basic"), make_result(pipeline="advanced")])
    table = format_markdown_table(summary)
    assert "basic" in table
    assert "advanced" in table
    assert table.startswith("| pipeline |")
