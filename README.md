# OpsSleuth — Agentic Incident Investigation Engine

An autonomous system that investigates production incidents using Corrective RAG:
hybrid retrieval (BM25 + dense), reranking, retrieval-quality grading, query
rewriting, and evidence-backed root-cause reports — instead of a plain
RAG-chatbot-over-docs.

## Stack

- LLM + embeddings: Gemini via Vertex AI (`google-cloud-aiplatform`)
- Dense vector store: Chroma (local, persistent)
- Keyword search: BM25 (`rank-bm25`)
- Orchestration: LangGraph (added in the corrective-retrieval milestone)
- API: FastAPI (added in the deployment milestone)
- Eval: Ragas / DeepEval

## Project layout

```
src/opssleuth/
  config.py           settings (env-driven)
  models.py            Document, TelemetryEvent, EvalQuestion, Chunk
  ingestion/
    loader.py          parse the starter dataset's jsonl files
    chunking.py         word-window chunking + telemetry-to-text
    embed.py            Vertex AI embeddings wrapper
    vector_store.py     Chroma wrapper
    bm25_index.py        BM25 index build/save/load
    pipeline.py          ingest_all(): load -> chunk -> embed -> store
  llm/
    gemini.py            Vertex AI Gemini generation wrapper
scripts/
  ingest.py              CLI: run the full ingestion pipeline
tests/                   unit tests (no GCP calls required)
opssleuth_starter_dataset/   synthetic documents, telemetry, eval questions
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
copy .env.example .env   # fill in GOOGLE_CLOUD_PROJECT
gcloud auth application-default login
```

Run tests (no GCP credentials needed):

```powershell
.venv\Scripts\python -m pytest tests/ -v
```

Run ingestion (requires GCP credentials + a project with Vertex AI enabled):

```powershell
.venv\Scripts\python scripts\ingest.py
```

This chunks `documents.jsonl`, converts `telemetry_events.jsonl` into
retrievable text records, embeds everything with the configured Vertex AI
embedding model, upserts into a local Chroma collection, and builds a BM25
index saved to `artifacts/bm25_index.pkl`.

## Build order (from the project plan)

1. Basic RAG pipeline (dense retrieval only) — **next milestone**
2. Hybrid retrieval (BM25 + dense, fused with Reciprocal Rank Fusion)
3. Reranking
4. Retrieval quality grading
5. Corrective retrieval loop (query rewriting, alternative retrieval)
6. Evidence / claim verification against retrieved context
7. RAG evaluation dashboard (Basic vs. Advanced vs. Corrective RAG, on
   `evaluation_questions.jsonl`)
8. FastAPI deployment

First milestone: correctly investigate `Q-001` and `Q-002` in
`evaluation_questions.jsonl` end to end before moving on to the agentic
corrective loop.

## Dataset

`opssleuth_starter_dataset/` is synthetic (fictional services, incidents,
telemetry) and designed to contain conflicting/stale documentation and
questions with insufficient evidence — see its own `README.md` for details.
