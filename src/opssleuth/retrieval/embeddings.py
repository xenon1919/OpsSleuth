import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from opssleuth.core.config import settings

_model: TextEmbeddingModel | None = None
_initialized = False


def _ensure_init() -> None:
    global _initialized
    if not _initialized:
        vertexai.init(
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
        )
        _initialized = True


def _get_model() -> TextEmbeddingModel:
    global _model
    _ensure_init()
    if _model is None:
        _model = TextEmbeddingModel.from_pretrained(settings.embedding_model)
    return _model


def embed_texts(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    batch_size: int = 16,
) -> list[list[float]]:
    """Embed a list of texts via Vertex AI. task_type is RETRIEVAL_DOCUMENT for
    indexing and RETRIEVAL_QUERY for querying."""
    model = _get_model()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = [TextEmbeddingInput(text=t, task_type=task_type) for t in batch]
        embeddings = model.get_embeddings(inputs)
        vectors.extend(e.values for e in embeddings)
    return vectors


def embed_query(text: str) -> list[float]:
    return embed_texts([text], task_type="RETRIEVAL_QUERY")[0]
