"""Sentence embeddings from an open encoder model, computed locally."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

import numpy as np

from veritas.local.runtime import import_torch, load_failure, pick_device


class LocalEmbedder:
    """Mean-pooled, L2-normalized embeddings from a Hugging Face encoder.

    Models of the e5 family were trained with "query: " and "passage: " prefixes (the same
    convention as the multilingual-e5-large model Pinecone hosts), so they are added here.
    """

    def __init__(self, model_name: str, *, batch_size: int = 32, device: str = "cpu") -> None:
        # CPU by default: a small encoder embeds a query in milliseconds, and keeping it off
        # the GPU leaves that free for the answer model.
        self.model_name = model_name
        self.batch_size = batch_size
        self._device_preference = device
        self._loaded: tuple[Any, Any, Any, str] | None = None
        self._lock = threading.Lock()

    def warm_up(self) -> None:
        self._load()

    def _load(self) -> tuple[Any, Any, Any, str]:
        with self._lock:
            if self._loaded is None:
                torch = import_torch()
                from transformers import AutoModel, AutoTokenizer

                try:
                    tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    model = AutoModel.from_pretrained(self.model_name).eval()
                except OSError as exc:
                    raise load_failure(self.model_name, exc) from exc
                device = pick_device(torch, self._device_preference)
                self._loaded = (torch, tokenizer, model.to(device), device)
            return self._loaded

    def _prefix(self, input_type: str) -> str:
        if "e5" not in self.model_name.lower():
            return ""
        return "query: " if input_type == "query" else "passage: "

    def embed(self, texts: Sequence[str], input_type: str = "passage") -> np.ndarray:
        """One unit-length float32 row per text."""
        torch, tokenizer, model, device = self._load()
        prefix = self._prefix(input_type)
        rows = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = [prefix + text for text in texts[start : start + self.batch_size]]
                encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                encoded = encoded.to(device)
                hidden = model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                rows.append(torch.nn.functional.normalize(pooled, dim=-1).float().cpu().numpy())
        if not rows:
            return np.zeros((0, model.config.hidden_size), dtype=np.float32)
        return np.concatenate(rows).astype(np.float32)
