import json
from pathlib import Path

from opssleuth.models import Document, EvalQuestion, TelemetryEvent


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_documents(data_dir: str | Path) -> list[Document]:
    path = Path(data_dir) / "documents.jsonl"
    return [Document.model_validate(row) for row in _read_jsonl(path)]


def load_telemetry_events(data_dir: str | Path) -> list[TelemetryEvent]:
    path = Path(data_dir) / "telemetry_events.jsonl"
    return [TelemetryEvent.model_validate(row) for row in _read_jsonl(path)]


def load_eval_questions(data_dir: str | Path) -> list[EvalQuestion]:
    path = Path(data_dir) / "evaluation_questions.jsonl"
    return [EvalQuestion.model_validate(row) for row in _read_jsonl(path)]
