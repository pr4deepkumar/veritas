"""Answer synthesis with an open-weights instruct model run in-process with transformers."""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from typing import Protocol

from veritas.core.errors import LLMOutputError
from veritas.local.runtime import device_lock, import_torch, load_failure, pick_device

logger = logging.getLogger(__name__)


Example = tuple[str, str]  # (user message, ideal assistant reply)


class ChatBackend(Protocol):
    def generate(self, system: str, prompt: str, examples: Sequence[Example] = ()) -> str: ...


class HuggingFaceChat:
    """Greedy decoding with a chat model's own template. Runs on CUDA or the Apple Silicon GPU
    when available, else the CPU."""

    def __init__(self, model_name: str, *, device: str = "auto", max_new_tokens: int = 512) -> None:
        torch = import_torch()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.device = pick_device(torch, device)
        # bfloat16 halves memory on GPUs and is what Qwen-family models are trained in; the
        # CPU path stays in float32, which every CPU supports at full speed.
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        except OSError as exc:
            raise load_failure(model_name, exc) from exc
        self.model = model.to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        logger.info("Loaded %s on %s", model_name, self.device)

    def generate(self, system: str, prompt: str, examples: Sequence[Example] = ()) -> str:
        messages = [{"role": "system", "content": system}]
        for user, assistant in examples:
            messages += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        messages.append({"role": "user", "content": prompt})
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)
        with device_lock(self.device), self._torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # greedy: the same evidence always yields the same answer
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class LocalLLM:
    """Drop-in replacement for the Claude client's ``generate_text``. The model loads on first
    use, and calls are serialized because a single model instance isn't safe to run
    concurrently."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        max_new_tokens: int = 512,
        backend: ChatBackend | None = None,
    ) -> None:
        self.model_name = model_name
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._backend = backend
        self._lock = threading.Lock()

    def warm_up(self) -> None:
        with self._lock:
            self._load()

    def _load(self) -> ChatBackend:
        if self._backend is None:
            self._backend = HuggingFaceChat(
                self.model_name, device=self._device, max_new_tokens=self._max_new_tokens
            )
        return self._backend

    def generate_text(self, *, system: str, prompt: str, examples: Sequence[Example] = ()) -> str:
        with self._lock:
            text = self._load().generate(system, prompt, examples)
        if not text:
            raise LLMOutputError("The local model returned no text.")
        return text
