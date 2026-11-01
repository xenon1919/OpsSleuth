.PHONY: install test ingest lint

install:
	python -m venv .venv
	.venv/Scripts/pip install -e ".[dev]"

test:
	.venv/Scripts/python -m pytest tests/ -v

ingest:
	.venv/Scripts/python scripts/ingest.py
