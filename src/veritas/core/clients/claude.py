"""Thin wrapper around the Anthropic SDK for answer synthesis and claim verification."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, TypeVar

from anthropic import Anthropic
from pydantic import BaseModel, ValidationError

from veritas.core.errors import ConfigurationError, LLMOutputError, LLMRefusalError
from veritas.core.settings import Effort, settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Thinking tokens count toward max_tokens. 16k leaves room for both while keeping
# non-streaming requests inside the SDK's HTTP timeout.
MAX_TOKENS = 16_000
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeClient:
    def __init__(self, client: Anthropic | None = None) -> None:
        if client is None:
            if not settings.anthropic_api_key:
                raise ConfigurationError("ANTHROPIC_API_KEY is required")
            client = Anthropic(api_key=settings.anthropic_api_key, timeout=settings.llm_timeout_seconds)
        self._client = client

    def generate_text(
        self,
        *,
        system: str,
        prompt: str,
        examples: Sequence[tuple[str, str]] = (),
        model: str | None = None,
        effort: Effort | None = None,
    ) -> str:
        """Same call shape as the local LLM; model and effort default to the answer settings.
        ``examples`` are (user, assistant) pairs sent as earlier conversation turns."""
        turns: list[dict[str, str]] = []
        for user, assistant in examples:
            turns += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        response = self._client.beta.messages.create(
            **_request(system, prompt, model or settings.chat_model, effort or settings.answer_effort, turns)
        )
        _raise_if_refused(response)
        if response.stop_reason == "max_tokens":
            logger.warning("Answer reached max_tokens=%d and may be cut off", MAX_TOKENS)
        # Read by block type: responses can also carry thinking and fallback blocks.
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            raise LLMOutputError(f"The model returned no text (stop_reason={response.stop_reason}).")
        return text

    def generate_structured(
        self, *, system: str, prompt: str, output_model: type[T], model: str, effort: Effort
    ) -> T:
        """Return the model's reply validated against ``output_model`` (structured outputs)."""
        try:
            response = self._client.beta.messages.parse(
                **_request(system, prompt, model, effort), output_format=output_model
            )
        except ValidationError as exc:
            raise LLMOutputError(
                "The model's structured output did not match the schema; it may have been cut off "
                "at max_tokens."
            ) from exc
        _raise_if_refused(response)
        parsed = response.parsed_output
        if parsed is None:
            raise LLMOutputError(
                f"The model returned no structured output (stop_reason={response.stop_reason})."
            )
        return parsed


def _request(
    system: str, prompt: str, model: str, effort: Effort, turns: Sequence[dict[str, str]] = ()
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": [*turns, {"role": "user", "content": prompt}],
        "output_config": {"effort": effort},
    }
    if settings.refusal_fallback:
        # On a policy decline, the API re-runs the request on a fallback model within the same call.
        request["betas"] = [FALLBACK_BETA]
        request["fallbacks"] = "default"
    return request


def _raise_if_refused(response: Any) -> None:
    # A decline is an HTTP 200 with stop_reason "refusal"; content is empty or partial.
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        raise LLMRefusalError(getattr(details, "category", None))
