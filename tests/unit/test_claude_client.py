from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from veritas.core.clients.claude import FALLBACK_BETA, MAX_TOKENS, ClaudeClient
from veritas.core.errors import ConfigurationError, LLMOutputError, LLMRefusalError
from veritas.core.settings import settings


class Verdict(BaseModel):
    ok: bool


class FakeMessages:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    def create(self, **kwargs):
        return self._respond("create", kwargs)

    def parse(self, **kwargs):
        return self._respond("parse", kwargs)

    def _respond(self, method: str, kwargs: dict):
        self.calls.append((method, kwargs))
        if self.error is not None:
            raise self.error
        return self.response


def make_client(response=None, error: Exception | None = None) -> tuple[ClaudeClient, FakeMessages]:
    messages = FakeMessages(response, error)
    sdk = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return ClaudeClient(client=sdk), messages


def response(*content, stop_reason: str = "end_turn", category: str | None = None, parsed=None):
    details = SimpleNamespace(category=category) if category else None
    return SimpleNamespace(
        stop_reason=stop_reason, stop_details=details, content=list(content), parsed_output=parsed
    )


def text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


def generate(client: ClaudeClient) -> str:
    return client.generate_text(system="sys", prompt="question", model="claude-sonnet-5-5", effort="medium")


def test_requires_an_api_key():
    with pytest.raises(ConfigurationError):
        ClaudeClient()


def test_text_is_read_from_text_blocks_only():
    client, _ = make_client(
        response(
            SimpleNamespace(type="thinking", thinking=""),
            text("Revenue grew [1]."),
            SimpleNamespace(type="fallback"),
            text(" Costs fell [2]."),
        )
    )
    assert generate(client) == "Revenue grew [1]. Costs fell [2]."


def test_request_uses_current_api_surface():
    client, messages = make_client(response(text("ok")))
    generate(client)

    method, kwargs = messages.calls[0]
    assert method == "create"
    assert kwargs["model"] == "claude-sonnet-5-5"
    assert kwargs["max_tokens"] == MAX_TOKENS
    assert kwargs["system"] == "sys"
    assert kwargs["messages"] == [{"role": "user", "content": "question"}]
    assert kwargs["output_config"] == {"effort": "medium"}
    assert kwargs["betas"] == [FALLBACK_BETA]
    assert kwargs["fallbacks"] == "default"
    # Current models reject sampling parameters and forced tool choice.
    assert not {"temperature", "top_p", "top_k", "tool_choice"} & kwargs.keys()


def test_refusal_fallback_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(settings, "refusal_fallback", False)
    client, messages = make_client(response(text("ok")))
    generate(client)
    kwargs = messages.calls[0][1]
    assert "fallbacks" not in kwargs
    assert "betas" not in kwargs


def test_refusal_raises_with_its_category():
    client, _ = make_client(response(stop_reason="refusal", category="cyber"))
    with pytest.raises(LLMRefusalError) as excinfo:
        generate(client)
    assert excinfo.value.category == "cyber"


def test_refusal_without_details():
    client, _ = make_client(response(stop_reason="refusal"))
    with pytest.raises(LLMRefusalError) as excinfo:
        generate(client)
    assert excinfo.value.category is None


def test_empty_reply_is_an_output_error():
    client, _ = make_client(response())
    with pytest.raises(LLMOutputError):
        generate(client)


def test_truncated_reply_is_returned_with_a_warning(caplog):
    client, _ = make_client(response(text("partial answer"), stop_reason="max_tokens"))
    with caplog.at_level(logging.WARNING):
        assert generate(client) == "partial answer"
    assert "max_tokens" in caplog.text


def test_structured_output_is_validated_by_the_sdk_parse_helper():
    client, messages = make_client(response(parsed=Verdict(ok=True)))

    result = client.generate_structured(
        system="s", prompt="p", output_model=Verdict, model="claude-sonnet-5-5", effort="high"
    )

    assert result == Verdict(ok=True)
    method, kwargs = messages.calls[0]
    assert method == "parse"
    assert kwargs["output_format"] is Verdict
    assert kwargs["output_config"] == {"effort": "high"}


def test_schema_mismatch_becomes_an_output_error():
    try:
        Verdict.model_validate_json('{"ok": tr')
    except ValidationError as exc:
        error = exc
    client, _ = make_client(error=error)
    with pytest.raises(LLMOutputError):
        client.generate_structured(system="s", prompt="p", output_model=Verdict, model="m", effort="low")


def test_missing_structured_output_is_an_output_error():
    client, _ = make_client(response())
    with pytest.raises(LLMOutputError):
        client.generate_structured(system="s", prompt="p", output_model=Verdict, model="m", effort="low")


def test_structured_refusal_is_reported_as_a_refusal():
    client, _ = make_client(response(stop_reason="refusal", category="bio"))
    with pytest.raises(LLMRefusalError):
        client.generate_structured(system="s", prompt="p", output_model=Verdict, model="m", effort="low")


def test_examples_are_sent_as_earlier_conversation_turns():
    client, messages = make_client(response(text("ok")))
    client.generate_text(system="s", prompt="question", examples=[("example q", "example a")])
    assert messages.calls[0][1]["messages"] == [
        {"role": "user", "content": "example q"},
        {"role": "assistant", "content": "example a"},
        {"role": "user", "content": "question"},
    ]


def test_answer_model_and_effort_default_to_settings(monkeypatch):
    monkeypatch.setattr(settings, "chat_model", "claude-opus-5-5")
    monkeypatch.setattr(settings, "answer_effort", "high")
    client, messages = make_client(response(text("ok")))
    client.generate_text(system="s", prompt="p")
    kwargs = messages.calls[0][1]
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["output_config"] == {"effort": "high"}
