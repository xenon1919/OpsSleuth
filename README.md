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

Organized by architectural layer, not by milestone, so each layer can grow
independently as the corrective-RAG loop, API, and eval harness are added:

```
src/opssleuth/
  core/                 cross-cutting concerns
    config.py            settings (env-driven, pydantic-settings)
    logging.py           stdlib logging setup
  domain/                shared types
    models.py             Document, TelemetryEvent, EvalQuestion, Chunk
  ingestion/             ETL: raw dataset -> Chunk objects
    loader.py             parse the starter dataset's jsonl files
    chunking.py           word-window chunking + telemetry-to-text
    pipeline.py            ingest_all(): load -> chunk -> embed -> store
  retrieval/             storage + search over Chunks
    embeddings.py          Vertex AI embeddings wrapper
    vector_store.py        Chroma wrapper
    bm25_index.py            BM25 index build/save/load
    (hybrid.py, reranker.py added in milestones 2-3)
  llm/
    gemini.py              Vertex AI Gemini generation wrapper
  graph/                 LangGraph corrective-retrieval orchestration (milestone 5+)
  eval/                  Basic vs Advanced vs Corrective RAG eval harness (milestone 7+)
  api/                   FastAPI service (milestone 8+)
  cli/
    ingest.py             `opssleuth-ingest` console script

data/
  starter_dataset/       synthetic documents, telemetry, eval questions (committed)
  chroma/                generated Chroma persistence dir (gitignored)
  artifacts/             generated BM25 index etc. (gitignored)

scripts/
  ingest.py              thin wrapper around opssleuth.cli.ingest for non-installed use

tests/
  unit/                  no GCP calls required
  (integration/ added once there are tests that need real GCP credentials)
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

Run ingestion (requires GCP credentials + a project with Vertex AI enabled —
this makes real, billed Vertex AI embedding calls):

```powershell
.venv\Scripts\python scripts\ingest.py
# or, once installed: opssleuth-ingest
```

This chunks `documents.jsonl`, converts `telemetry_events.jsonl` into
retrievable text records, embeds everything with the configured Vertex AI
embedding model, upserts into a local Chroma collection (`data/chroma/`), and
builds a BM25 index saved to `data/artifacts/bm25_index.pkl`.

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

`data/starter_dataset/` is synthetic (fictional services, incidents,
telemetry) and designed to contain conflicting/stale documentation and
questions with insufficient evidence — see its own `README.md` for details.
