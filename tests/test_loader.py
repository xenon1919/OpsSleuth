from opssleuth.ingestion.loader import load_documents, load_eval_questions, load_telemetry_events

DATA_DIR = "opssleuth_starter_dataset"


def test_load_documents():
    docs = load_documents(DATA_DIR)
    assert len(docs) == 30
    assert docs[0].document_id == "DOC-001"


def test_load_telemetry_events():
    events = load_telemetry_events(DATA_DIR)
    assert len(events) == 6
    assert events[0].event_id == "EVT-1001"


def test_load_eval_questions():
    questions = load_eval_questions(DATA_DIR)
    assert len(questions) == 25
    assert questions[0].question_id == "Q-001"
