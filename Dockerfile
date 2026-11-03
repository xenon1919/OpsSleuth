FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

COPY data/starter_dataset/ ./data/starter_dataset/

EXPOSE 8000

# Run `docker run <image> opssleuth-ingest` first (needs GCP credentials mounted/
# passed in) to populate data/chroma and data/artifacts before serving.
CMD ["opssleuth-serve", "--host", "0.0.0.0"]
