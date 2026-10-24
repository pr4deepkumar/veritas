from __future__ import annotations

import sys

import pytest

from veritas.core.errors import ConfigurationError, LLMOutputError
from veritas.local.llm import HuggingFaceChat, LocalLLM
from veritas.local.runtime import device_lock, local_models_installed


class RecordingBackend:
    def __init__(self, reply: str = "Revenue grew 9% [1].") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str, list]] = []

    def generate(self, system, prompt, examples=()):
        self.calls.append((system, prompt, list(examples)))
        return self.reply


def test_generate_text_passes_the_conversation_to_the_model():
    backend = RecordingBackend()
    llm = LocalLLM("org/model", backend=backend)

    reply = llm.generate_text(system="sys", prompt="question", examples=[("example q", "example a")])

    assert reply == "Revenue grew 9% [1]."
    assert backend.calls == [("sys", "question", [("example q", "example a")])]


def test_an_empty_reply_is_an_output_error():
    with pytest.raises(LLMOutputError):
        LocalLLM("org/model", backend=RecordingBackend(reply="")).generate_text(system="s", prompt="p")


def test_without_the_local_extra_the_error_says_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ConfigurationError, match=r"\.\[local\]"):
        HuggingFaceChat("org/model")


def test_the_model_loads_lazily(monkeypatch):
    loaded = []
    monkeypatch.setattr(
        "veritas.local.llm.HuggingFaceChat", lambda *a, **k: loaded.append(a) or RecordingBackend()
    )
    llm = LocalLLM("org/model")
    assert loaded == []
    llm.generate_text(system="s", prompt="p")
    llm.generate_text(system="s", prompt="p")
    assert loaded == [("org/model",)]


def test_the_install_check_needs_both_torch_and_transformers(monkeypatch):
    monkeypatch.setattr("veritas.local.runtime.find_spec", lambda name: object())
    assert local_models_installed()
    monkeypatch.setattr("veritas.local.runtime.find_spec", lambda name: None if name == "torch" else object())
    assert not local_models_installed()


def test_gpu_work_is_serialized_across_models_but_cpu_work_is_not():
    assert device_lock("mps") is device_lock("cuda")  # one lock for the answer and NLI models
    with device_lock("cpu"), device_lock("cpu"):  # a no-op: nested CPU sections don't deadlock
        pass
