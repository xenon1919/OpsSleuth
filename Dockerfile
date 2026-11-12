FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

COPY data/starter_dataset/ ./data/starter_dataset/

# Pre-ingested indexes (run `opssleuth-ingest` locally first to populate these -
# needs GCP credentials - so the image serves immediately with no cold-start ingestion).
COPY data/chroma/ ./data/chroma/
COPY data/artifacts/ ./data/artifacts/

# Pre-download the reranker so the first request isn't stuck fetching ~1GB from HF.
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"

EXPOSE 8000

CMD ["opssleuth-serve", "--host", "0.0.0.0"]
