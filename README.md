# OpsSleuth — Agentic Incident Investigation Engine

OpsSleuth investigates production incidents the way an on-call engineer would:
it retrieves evidence, grades whether that evidence is actually good enough,
and — if it isn't — rewrites its own search query and tries again, instead of
confidently answering from whatever it happened to find first.

> Why did `payment-api` start returning 503s after the latest deployment?

A plain RAG chatbot embeds that question, grabs the top-k nearest documents,
and generates an answer — even if the retrieved evidence is about a different
incident, is outdated, or doesn't actually contain the answer. OpsSleuth is
built to catch and correct exactly those failure modes, and to **measure**
whether each added technique actually helps rather than just assuming it does.

## The centerpiece: measured, not assumed

The project implements four pipeline variants of increasing sophistication
and evaluates all of them against the same 25 held-out questions — including
ones with conflicting documentation, multi-hop reasoning, and questions that
have no answer in the corpus at all (`opssleuth-evaluate`):

| pipeline | recall@k | mrr | ndcg@k | ctx precision | faithfulness | avg latency |
| --- | --- | --- | --- | --- | --- | --- |
| **basic** — dense vector retrieval only | **0.769** | 0.901 | **0.756** | 0.344 | **1.000** | 4.6s |
| **advanced** — hybrid (BM25+dense) + reranking | 0.751 | 0.907 | 0.721 | **0.352** | 0.960 | 6.5s |
| **corrective** — advanced + grading + query-rewrite retry loop | 0.751 | 0.907 | 0.721 | **0.352** | 0.960 | 14.5s |

**The honest finding:** hybrid retrieval + reranking did *not* uniformly win.
Basic dense-only retrieval actually edged out the fancier pipelines on
recall@k and ndcg@k in this run. Tracing it down: on a "stale document"
question (is the old deployment guide still valid?), the reranker dropped
*both* gold documents that dense-only retrieval had kept in its top 5,
causing advanced/corrective to wrongly answer "insufficient evidence" where
basic correctly identified the superseded doc. Advanced/corrective did win on
context precision (less retrieved noise) and reached the right answer on
every multi-hop question, but introduced one unsupported claim basic didn't
have, and cost 3x the latency.

The corrective loop itself worked as designed: of the two questions in the
set with literally no answer in the corpus (e.g. "what's the production JWT
signing key?"), it correctly triggered a query rewrite, retried with a
sharper query, and after exhausting its retry budget still **abstained**
rather than inventing a secret — a 0% "correction success rate" there is the
right outcome, not a bug, since there was never a right answer to find.

Full per-question results: `data/artifacts/eval_results.json` (regenerate
with `opssleuth-evaluate`).

## How it works

```
question
   │
   ▼
query embedding ──► dense retrieval ─┐
                                      ├─► Reciprocal Rank Fusion ─► cross-encoder rerank ─► top-k chunks
BM25 keyword search ─────────────────┘                                                         │
                                                                                                 ▼
                                                                          LLM-as-judge: relevant? sufficient?
                                                                                 │                  │
                                                                      insufficient,                 │ sufficient
                                                                      attempts left                 │
                                                                                 │                   ▼
                                                                      rewrite query            generate grounded
                                                                      (missing-evidence             answer
                                                                       aware) and retry              │
                                                                                                       ▼
                                                                                        LLM-as-judge: is every
                                                                                        claim actually backed
                                                                                        by the evidence?
```

- **Hybrid retrieval**: dense (Vertex AI embeddings + Chroma) and BM25
  (`rank-bm25`) each retrieve independently; results are merged with
  Reciprocal Rank Fusion, which rewards chunks both retrievers agree on.
- **Reranking**: a local cross-encoder (`BAAI/bge-reranker-base`, no API
  cost) re-scores the candidate set by looking at the question and each
  chunk *together*, which is slower but far more precise than independent
  similarity scores.
- **Retrieval grading**: Gemini grades each retrieved chunk relevant/
  irrelevant for *this specific question*, then judges whether the relevant
  subset is actually sufficient to answer confidently — this is the CRAG-style
  correction signal, done via structured JSON output rather than regex
  parsing.
- **Corrective loop** (LangGraph `StateGraph`): retrieve → grade →
  (insufficient and attempts remain? rewrite the query using what's missing,
  then retry) → generate. Grading always runs against the *original*
  question even as the search query evolves, so correction can't drift
  off-task.
- **Claim verification**: after generation, a separate LLM-as-judge pass
  decomposes the answer into atomic claims and checks each one against the
  evidence independently — catching claims the generator cited a source for
  but that source doesn't actually support.

## Stack

| component | choice | why |
| --- | --- | --- |
| LLM + embeddings | Gemini via Vertex AI (`google-cloud-aiplatform`) | already had Vertex AI set up |
| Dense vector store | Chroma (local, persistent) | no infra to stand up |
| Keyword search | BM25 (`rank-bm25`) | lightweight, no service |
| Reranker | local `sentence-transformers` CrossEncoder | no extra paid API call |
| Orchestration | LangGraph | explicit state machine for the retry loop |
| API | FastAPI + uvicorn | |
| Eval | custom (see note below) | |

**Note on eval tooling**: the original plan called for Ragas/DeepEval.
Ragas 0.4.3 (latest at time of writing) unconditionally imports a
`langchain-community` module at package-import time that no longer exists in
current `langchain-community` releases; the only fix is downgrading the
entire `langchain-core` stack, which breaks LangGraph 1.x (a load-bearing
dependency for the corrective loop). Rather than force a brittle, unmaintained
dependency chain into the project, the eval metrics (`eval/metrics.py`:
Recall@K, MRR, NDCG, Context Precision) and the LLM-as-judge grading/
faithfulness checks are implemented directly — arguably a stronger
demonstration of understanding *how* these metrics work than calling a
library that computes them opaquely.

## Project layout

```
src/opssleuth/
  core/                 settings (pydantic-settings) + logging
  domain/                shared pydantic models (Chunk, RagAnswer, RetrievalGrade, ...)
  ingestion/             ETL: raw dataset -> Chunk objects (loader, chunking, pipeline)
  retrieval/             dense, BM25, hybrid (RRF), reranker, query rewriting
  llm/                   Vertex AI Gemini wrapper (generate + structured generate_json)
  grading/               LLM-as-judge: retrieval-quality grading, faithfulness/claim verification
  graph/                 LangGraph corrective-retrieval loop
  pipelines/             the four end-to-end variants: basic, hybrid, advanced, corrective
  eval/                  retrieval metrics + the pipeline-comparison harness
  api/                   FastAPI app + request/response schemas
  cli/                   console scripts (ingest, query, evaluate, serve)

data/
  starter_dataset/       synthetic docs, telemetry, eval questions (committed)
  chroma/, artifacts/    generated at ingest time (gitignored)

tests/
  unit/                  no network calls - everything mocked (52 tests)
  integration/           hits live Vertex AI; skipped by default, run with `-m integration`
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
copy .env.example .env   # fill in GOOGLE_CLOUD_PROJECT
gcloud auth application-default login
```

For an exact reproduction of the environment this was built/tested in, use
the pinned lockfile instead: `pip install -r requirements-lock.txt && pip
install -e . --no-deps`.

Run unit tests (fast, no GCP calls, mocked throughout):

```powershell
.venv\Scripts\python -m pytest tests/unit -v
```

Run the live integration tests (hits real Vertex AI; requires ingestion to
have been run first):

```powershell
.venv\Scripts\python -m pytest tests/integration -m integration -v
```

Run ingestion (makes real, billed Vertex AI embedding calls):

```powershell
opssleuth-ingest
```

Ask a question through any pipeline variant:

```powershell
opssleuth-query --pipeline corrective --question "Why did payment-api return 503s?"
opssleuth-query --pipeline advanced --grade --verify --eval-ids Q-001 Q-019
```

Run the full Basic vs Advanced vs Corrective comparison:

```powershell
opssleuth-evaluate
```

Serve the HTTP API (`POST /investigate`, `GET /health`, `GET /pipelines`):

```powershell
opssleuth-serve --port 8000
```

## Dataset

`data/starter_dataset/` is synthetic (fictional services, incidents,
telemetry) and deliberately contains conflicting/stale documentation and
questions with no answer in the corpus at all — see its own `README.md` for
details.

## License

MIT — see `LICENSE`.
