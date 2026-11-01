import json
from typing import Any

import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

from opssleuth.core.config import settings

_initialized = False
_models: dict[str, GenerativeModel] = {}


def _ensure_init() -> None:
    global _initialized
    if not _initialized:
        vertexai.init(
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
        )
        _initialized = True


def _get_model(model_name: str, system_instruction: str | None) -> GenerativeModel:
    _ensure_init()
    key = f"{model_name}::{system_instruction or ''}"
    if key not in _models:
        _models[key] = GenerativeModel(model_name, system_instruction=system_instruction)
    return _models[key]


def generate(
    prompt: str,
    system: str | None = None,
    temperature: float = 0.0,
    model_name: str | None = None,
) -> str:
    model = _get_model(model_name or settings.gemini_model, system)
    response = model.generate_content(
        prompt,
        generation_config=GenerationConfig(temperature=temperature),
    )
    return response.text


def generate_json(
    prompt: str,
    response_schema: dict[str, Any],
    system: str | None = None,
    temperature: float = 0.0,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Like generate(), but constrains output to the given JSON schema and parses it.
    Used for structured steps (grading, query rewriting) where free text would need
    fragile regex parsing."""
    model = _get_model(model_name or settings.gemini_model, system)
    response = model.generate_content(
        prompt,
        generation_config=GenerationConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=response_schema,
        ),
    )
    return json.loads(response.text)
