"""Exception types shared across the VERITAS layers."""

from __future__ import annotations


class VeritasError(Exception):
    """Base class for errors raised by VERITAS itself."""


class ConfigurationError(VeritasError):
    """A required setting (API key, bucket name, ...) is missing or invalid."""


class PipelineError(VeritasError):
    """One or more items in an ingestion or indexing run failed."""


class LLMError(VeritasError):
    """A model call completed but produced no usable result."""


class LLMRefusalError(LLMError):
    """The model (and any configured fallback) declined the request on policy grounds."""

    def __init__(self, category: str | None = None) -> None:
        self.category = category
        suffix = f" (category: {category})" if category else ""
        super().__init__(f"The model declined this request{suffix}.")


class LLMOutputError(LLMError):
    """The model's output was empty, cut off, or did not match the requested schema."""
