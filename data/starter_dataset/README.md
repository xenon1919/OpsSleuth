# OpsSleuth synthetic starter dataset

This package is a self-contained starter corpus for building an incident-investigation RAG / Corrective RAG prototype.

## Files
- `documents.jsonl`: 30 synthetic operational documents. Each line is one JSON document with stable `document_id`, metadata, and `text`.
- `evaluation_questions.jsonl`: 25 questions with task type, relevant document IDs, reference answers, expected confidence, and challenge labels.
- `telemetry_events.jsonl`: 6 structured synthetic monitoring/log events. Convert each event to text or index its serialized JSON as a separate retrieval record.

## Important
All organizations, services, incident records, versions, and telemetry in this dataset are fictional and synthetic. Do not treat it as real production guidance.

## Suggested ingestion
1. Load `documents.jsonl` and `telemetry_events.jsonl`.
2. Preserve IDs and metadata in your vector store.
3. Chunk document text (start with 400-600 tokens and 60-100 token overlap; keep each source document ID on every chunk).
4. Build dense retrieval plus BM25.
5. Fuse results with Reciprocal Rank Fusion, then rerank.
6. Use `evaluation_questions.jsonl` as a held-out test set. Avoid putting reference answers into the retrieval corpus.
7. For each question, compare baseline RAG, hybrid + reranking, and Corrective RAG.

## Suggested Corrective RAG behavior
- Grade each retrieved chunk for relevance to the question.
- If evidence is weak, rewrite the query and retrieve again.
- Detect stale/superseded documents using `status`, `last_updated`, and `version`.
- If sources conflict, prefer current authoritative sources only when metadata supports that choice; explain the conflict.
- If evidence remains insufficient, abstain and identify what evidence is missing.
- Cite document IDs for every factual claim.

## Evaluation notes
`relevant_document_ids` is a document-level gold set, not a strict chunk-level answer key. For retrieval metrics, count a hit when any chunk from a relevant document is retrieved. For answer evaluation, compare generated answers with `reference_answer` and inspect citations/faithfulness.
