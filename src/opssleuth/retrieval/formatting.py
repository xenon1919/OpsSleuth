from opssleuth.domain.models import RetrievedChunk


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as labeled evidence blocks, shared by generation and
    grading prompts so both see the same id/metadata framing."""
    blocks = []
    for chunk in chunks:
        meta = chunk.metadata
        header = (
            f"[{chunk.document_id}] title={meta.get('title', '')} "
            f"status={meta.get('status', '')} last_updated={meta.get('last_updated', '')} "
            f"version={meta.get('version', '')}"
        )
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)
