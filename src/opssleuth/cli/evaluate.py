import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from opssleuth.core.config import settings
from opssleuth.core.logging import get_logger
from opssleuth.eval.report import format_markdown_table, summarize
from opssleuth.eval.runner import PIPELINES, run_evaluation

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Basic vs Advanced vs Corrective RAG comparison over evaluation_questions.jsonl."
    )
    parser.add_argument(
        "--pipelines",
        nargs="*",
        default=["basic", "advanced", "corrective"],
        choices=sorted(PIPELINES),
        help="Which pipelines to compare (default: basic advanced corrective).",
    )
    parser.add_argument(
        "--question-ids",
        nargs="*",
        default=None,
        help="Limit to specific question ids (default: all questions in the dataset).",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--no-faithfulness",
        action="store_true",
        help="Skip the per-answer LLM-judge faithfulness check (faster/cheaper).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Where to write eval_results.json and eval_report.md (default: artifacts dir).",
    )
    args = parser.parse_args()

    def on_result(row):
        status = f"error: {row.error}" if row.error else f"recall={row.recall_at_k:.2f}"
        print(f"  {row.question_id} / {row.pipeline}: {status}")

    results = run_evaluation(
        pipeline_names=args.pipelines,
        question_ids=args.question_ids,
        top_k=args.top_k,
        check_faithfulness=not args.no_faithfulness,
        on_result=on_result,
    )

    summary = summarize(results)
    table = format_markdown_table(summary)
    print("\n" + table)

    out_dir = Path(args.out_dir or settings.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_results.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
    )
    (out_dir / "eval_report.md").write_text(table + "\n", encoding="utf-8")
    print(f"\nSaved {len(results)} raw results and report to {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
