from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from veritas.core.settings import Settings

ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"


def test_env_example_loads_even_though_it_holds_keys_for_other_tools(monkeypatch):
    # .env.example carries AWS_* variables that boto3 reads; Settings must ignore them.
    for name in ("CHAT_MODEL", "LOG_LEVEL", "CHUNK_SIZE", "CHUNK_OVERLAP"):
        monkeypatch.delenv(name, raising=False)
    loaded = Settings(_env_file=ENV_EXAMPLE)
    assert loaded.pinecone_index == "veritas-research"
    assert loaded.chat_model == "claude-sonnet-5-5"


def test_verifier_model_defaults_to_chat_model():
    assert Settings(_env_file=None, chat_model="claude-opus-5-5").verifier_model_name == "claude-opus-5-5"
    assert (
        Settings(_env_file=None, verifier_model="claude-haiku-5-5").verifier_model_name == "claude-haiku-5-5"
    )


def test_log_level_is_case_insensitive():
    assert Settings(_env_file=None, log_level="debug").log_level == "DEBUG"


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (500, 500), (500, -1)])
def test_chunking_settings_are_validated(size, overlap):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_size=size, chunk_overlap=overlap)


def test_effort_is_validated():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, answer_effort="extreme")


def test_providers_default_to_local_and_are_validated(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("VECTOR_STORE", raising=False)
    loaded = Settings(_env_file=None)
    assert (loaded.llm_provider, loaded.vector_store) == ("local", "local")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_provider="openai")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, vector_store="faiss")
