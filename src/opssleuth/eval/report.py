import statistics
from collections import defaultdict
from typing import Any

from opssleuth.eval.runner import QuestionResult


def summarize(results: list[QuestionResult]) -> dict[str, dict[str, Any]]:
    by_pipeline: dict[str, list[QuestionResult]] = defaultdict(list)
    for row in results:
        by_pipeline[row.pipeline].append(row)

    summary: dict[str, dict[str, Any]] = {}
    for pipeline, rows in by_pipeline.items():
        ok_rows = [r for r in rows if r.error is None]
        faith_rows = [r for r in ok_rows if r.is_faithful is not None]
        # "corrective_rows" is really just ok_rows for the corrective pipeline, but
        # final_sufficiency is None for other pipelines, so this stays generic.
        graded_rows = [r for r in ok_rows if r.final_sufficiency is not None]
        corrected_rows = [r for r in graded_rows if r.attempts > 1]

        summary[pipeline] = {
            "n": len(rows),
            "n_errors": len(rows) - len(ok_rows),
            "recall_at_k": statistics.mean(r.recall_at_k for r in ok_rows) if ok_rows else None,
            "mrr": statistics.mean(r.mrr for r in ok_rows) if ok_rows else None,
            "ndcg_at_k": statistics.mean(r.ndcg_at_k for r in ok_rows) if ok_rows else None,
            "context_precision": statistics.mean(r.context_precision for r in ok_rows) if ok_rows else None,
            "faithfulness_rate": (
                sum(r.is_faithful for r in faith_rows) / len(faith_rows) if faith_rows else None
            ),
            "avg_latency_seconds": statistics.mean(r.latency_seconds for r in ok_rows) if ok_rows else None,
            "correction_trigger_rate": (
                len(corrected_rows) / len(graded_rows) if graded_rows else None
            ),
            "correction_success_rate": (
                sum(1 for r in corrected_rows if r.final_sufficiency == "sufficient") / len(corrected_rows)
                if corrected_rows
                else None
            ),
        }
    return summary


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


_COLUMNS = [
    ("n", "n"),
    ("n_errors", "errors"),
    ("recall_at_k", "recall@k"),
    ("mrr", "mrr"),
    ("ndcg_at_k", "ndcg@k"),
    ("context_precision", "ctx_precision"),
    ("faithfulness_rate", "faithfulness"),
    ("avg_latency_seconds", "avg_latency_s"),
    ("correction_trigger_rate", "correction_trigger"),
    ("correction_success_rate", "correction_success"),
]


def format_markdown_table(summary: dict[str, dict[str, Any]]) -> str:
    headers = ["pipeline"] + [label for _, label in _COLUMNS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for pipeline, metrics in summary.items():
        row = [pipeline] + [_fmt(metrics[key]) for key, _ in _COLUMNS]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
