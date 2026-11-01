FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

COPY data/starter_dataset/ ./data/starter_dataset/

# Overridden for the API service once it exists (milestone 8); defaults to ingestion.
ENTRYPOINT ["opssleuth-ingest"]
