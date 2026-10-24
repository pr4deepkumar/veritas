"""Runtime configuration, read from environment variables and an optional ``.env`` file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    # extra="ignore": .env also holds variables meant for other tools (AWS_ACCESS_KEY_ID for
    # boto3, ...). Without it, pydantic-settings rejects those keys and the app fails to start.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "VERITAS API"
    log_level: str = "INFO"
    # Directory holding the web console (index.html, app.js, style.css). Auto-detected when unset.
    web_dir: Path | None = None

    # Providers. "local" runs on this machine with open models and needs no API keys
    # (pip install -e ".[local]"); "claude" and "pinecone" use the managed services.
    llm_provider: Literal["local", "claude"] = "local"
    vector_store: Literal["local", "pinecone"] = "local"

    # Local models (Hugging Face Hub ids; downloaded once on first use)
    local_llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    local_embedding_model: str = "intfloat/multilingual-e5-small"
    local_device: str = "auto"  # where the answer and NLI models run: auto, cpu, mps or cuda
    local_max_new_tokens: int = 512
    local_index_dir: Path = Path("data/index")
    documents_dir: Path = Path("data/documents")

    # Claude
    anthropic_api_key: str = ""
    chat_model: str = "claude-sonnet-5-5"
    verifier_model: str = ""  # empty: use chat_model
    answer_effort: Effort = "medium"
    verifier_effort: Effort = "medium"
    llm_timeout_seconds: float = 180.0
    # Re-run a request on a fallback model when the primary model declines it on policy grounds.
    # Turn off for models that don't support server-side fallbacks.
    refusal_fallback: bool = True

    # Model behind the free, offline "local-nli" verification mode (Hugging Face Hub id).
    local_nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index: str = "veritas-research"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    embedding_model: str = "multilingual-e5-large"
    embedding_dimension: int = 1024

    # Chunking, in characters. multilingual-e5-large embeds at most ~507 tokens per input and
    # silently drops the rest, so chunks stay well under that.
    chunk_size: int = 1500
    chunk_overlap: int = 200

    # AWS / S3 ingestion
    aws_region: str = "us-east-1"
    s3_bucket: str = ""
    s3_input_prefix: str = "input_pdfs/"
    s3_output_prefix: str = "parsed_json/"

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        return value.upper()

    @model_validator(mode="after")
    def _check_chunking(self) -> Settings:
        if self.chunk_size <= 0 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_SIZE must be positive and CHUNK_OVERLAP in [0, CHUNK_SIZE)")
        return self

    @property
    def verifier_model_name(self) -> str:
        return self.verifier_model or self.chat_model

    @property
    def claude_available(self) -> bool:
        return bool(self.anthropic_api_key)


settings = Settings()
