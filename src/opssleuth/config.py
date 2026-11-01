from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_cloud_project: str = ""
    google_cloud_location: str = "us-central1"
    gemini_model: str = "gemini-2.0-flash"
    embedding_model: str = "text-embedding-005"

    data_dir: str = "opssleuth_starter_dataset"
    chroma_persist_dir: str = "chroma_data"
    chroma_collection: str = "opssleuth_docs"
    artifacts_dir: str = "artifacts"

    chunk_size_words: int = 450
    chunk_overlap_words: int = 75


settings = Settings()
