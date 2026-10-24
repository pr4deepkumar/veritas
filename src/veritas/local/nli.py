"""Offline claim verification with an open-source NLI model: no API key, no cost.

The answer is split into claims by rule (one per sentence, or per clause where a citation
marker closes a clause mid-sentence). Each claim is then checked against the passages it
cites by a natural language inference (NLI) cross-encoder running locally (GPU or CPU):

- supported: some cited passage entails the claim.
- unsupported: a cited passage contradicts it, it cites a passage that does not exist, or
  no passage covers it.
- partially supported: the passages are neutral but share most of the claim's content words,
  i.e. they discuss it without entailing all of it.

The default model, MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli (MIT licence, trained on
MNLI, FEVER and ANLI), is downloaded from the Hugging Face Hub on first use (about 0.4 GB).
Requires the optional dependencies: ``pip install -e ".[local]"``.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from veritas.core.errors import ConfigurationError
from veritas.core.evidence import Evidence
from veritas.core.models import ClaimVerdict
from veritas.local.runtime import device_lock, import_torch, load_failure, pick_device

logger = logging.getLogger(__name__)

# (premise, hypothesis) pairs -> one {"entailment": p, "neutral": p, "contradiction": p} each.
Classifier = Callable[[list[tuple[str, str]]], list[dict[str, float]]]

NLI_LABELS = {"entailment", "neutral", "contradiction"}
MIN_CLAIM_WORDS = 3
# Share of a claim's content words found in a passage at which a neutral verdict counts as
# "partially supported". It only splits partial from unsupported; both are grounding issues.
PARTIAL_OVERLAP = 0.5

_MARKER_GROUP = re.compile(r"(?:\s*\[\d+\])+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9\[])|\n+")
_LIST_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_LEADING_CONNECTOR = re.compile(
    r"^(?:[,;:]\s*)*(?:and|but|while|whereas|although|though|also)\b[\s,]*", re.IGNORECASE
)
# Sentences that only describe what the evidence lacks ("The label doesn't say whether it is
# waterproof.") are not factual claims. Only reporting verbs count: "The warranty does not cover
# water damage." is a claim about the warranty, not about the sources.
_NOT = r"(?:\s+not|n['\N{RIGHT SINGLE QUOTATION MARK}]t)\b"
_META_SENTENCE = re.compile(
    rf"\b(?:do|does|did){_NOT}\s+(?:\w+\s+)?"
    r"(?:say|state|mention|specify|report|disclose|discuss|describe|explain|indicate|clarify|detail)\b"
    rf"|\b(?:is|are|was|were){_NOT}\s+(?:\w+\s+)?"
    r"(?:mentioned|stated|specified|reported|disclosed|discussed|described|explained|indicated)\b"
    rf"|\b(?:sources?|documents?|passages?|evidence|context)\b[^.]*\b(?:do|does|did){_NOT}"
    r"|\bno (?:information|evidence|details?|data|mention)\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[^\W_]+")
_PREMISE_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
WINDOW_SENTENCES = 2
MAX_WINDOWS_PER_PASSAGE = 24
_STOPWORDS = frozenset(
    """a an and are as at be been by for from had has have in into is it its of on or that the
    their this to was were will with""".split()  # noqa: SIM905
)


@dataclass
class ClaimSpan:
    text: str  # verbatim from the answer, without citation markers
    citations: list[int] = field(default_factory=list)


def split_claims(answer: str) -> list[ClaimSpan]:
    claims: list[ClaimSpan] = []
    for raw_sentence in _SENTENCE_BREAK.split(answer):
        sentence = _LIST_BULLET.sub("", raw_sentence).strip()
        if not sentence:
            continue
        if _is_meta(sentence):
            leading = _MARKER_GROUP.match(sentence)
            if leading and claims:  # "... grew 12%. [1] The report doesn't say why."
                claims[-1].citations.extend(int(n) for n in re.findall(r"\d+", leading.group()))
            continue
        cursor = 0
        for group in _MARKER_GROUP.finditer(sentence):
            citations = [int(n) for n in re.findall(r"\d+", group.group())]
            if group.start() == 0 and claims:
                claims[-1].citations.extend(citations)  # "... grew 12%. [1]" cites the previous sentence
            else:
                _append_claim(claims, sentence[cursor : group.start()], citations)
            cursor = group.end()
        _append_claim(claims, sentence[cursor:], [])
    return claims


def _is_meta(sentence: str) -> bool:
    """A sentence that only says what the evidence doesn't cover. Citing a passage makes it a claim."""
    leading = _MARKER_GROUP.match(sentence)  # markers that belong to the previous sentence
    rest = sentence[leading.end() :] if leading else sentence
    return not _MARKER_GROUP.search(rest) and _META_SENTENCE.search(rest) is not None


def _append_claim(claims: list[ClaimSpan], segment: str, citations: list[int]) -> None:
    text = _LEADING_CONNECTOR.sub("", segment.strip()).strip(" ,;:")
    text = re.sub(r"[.!?]+$", "", text).strip()
    if len(_WORD.findall(text)) >= MIN_CLAIM_WORDS:  # drops the "." left after a final marker
        claims.append(ClaimSpan(text, citations))


def premise_windows(text: str) -> list[str]:
    """Overlapping two-sentence windows of a passage. NLI models are trained mostly on short
    premises and read long chunks less reliably, so each claim is also scored against the few
    sentences that state it (the approach of SummaC, Laban et al., 2022). Passages of two
    sentences or fewer need no windows."""
    sentences = [s.strip() for s in _PREMISE_SENTENCE.split(text) if s.strip()]
    if len(sentences) <= WINDOW_SENTENCES:
        return []
    windows = [
        " ".join(sentences[i : i + WINDOW_SENTENCES]) for i in range(len(sentences) - WINDOW_SENTENCES + 1)
    ]
    return windows[:MAX_WINDOWS_PER_PASSAGE]


def _context(title: str) -> str:
    """A passage's title as plain words: ``reports/Q2_2026-results`` -> ``Q2 2026 results``."""
    return re.sub(r"[_-]+", " ", title.rsplit("/", 1)[-1]).strip()


def _headed(passage: Evidence, text: str) -> str:
    """Prefix a premise with its source's title. A window such as "hardware revenue declined
    2%" doesn't say whose revenue or when, while answers do ("Northwind's hardware revenue ...
    in Q2 2026"), so without the title the model calls correct claims neutral."""
    context = _context(passage.title)
    return f"{context}: {text}" if context else text


def _premises(passage: Evidence) -> list[str]:
    """The whole passage, then its two-sentence windows, each headed by the passage's title."""
    return [_headed(passage, text) for text in (passage.text, *premise_windows(passage.text))]


def _hypothesis(text: str) -> str:
    # A clause split off mid-sentence can lack its subject ("will retail at $199").
    return text if text[:1].isupper() or text[:1].isdigit() else f"It {text}"


def _content_words(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.lower()) if word not in _STOPWORDS}


def _overlap(claim: str, passage: str) -> float:
    words = _content_words(claim)
    return len(words & _content_words(passage)) / len(words) if words else 0.0


def _refs(indices: Sequence[int]) -> str:
    return "".join(f"[{i}]" for i in indices)


def _argmax(scores: dict[str, float]) -> str:
    return max(scores, key=scores.__getitem__)


class LocalNLIVerifier:
    def __init__(self, classifier: Classifier) -> None:
        self.classify = classifier

    def judge(self, answer: str, evidence: Sequence[Evidence]) -> list[ClaimVerdict]:
        passages = {p.index: p for p in evidence}
        plans: list[tuple[ClaimSpan, int, list[tuple[list[int], str]]]] = []
        pairs: list[tuple[str, str]] = []
        for claim in split_claims(answer):
            cited = [i for i in dict.fromkeys(claim.citations) if i in passages]
            # Uncited claims are checked against all passages; a citation to a missing passage gives no support.
            refs = cited or ([] if claim.citations else list(passages))
            premises: list[tuple[list[int], str]] = []
            if (
                len(refs) > 1
            ):  # all cited passages together, then each alone (in case the combination gets truncated)
                premises.append((refs, "\n\n".join(_headed(passages[i], passages[i].text) for i in refs)))
            premises += [([i], premise) for i in refs for premise in _premises(passages[i])]
            plans.append((claim, len(pairs), premises))
            pairs.extend((premise, _hypothesis(claim.text)) for _, premise in premises)

        scores = self.classify(pairs) if pairs else []
        verdicts = [
            _decide(claim, premises, scores[start : start + len(premises)])
            for claim, start, premises in plans
        ]
        return self._explain_wrong_citations([claim for claim, _, _ in plans], verdicts, passages)

    def _explain_wrong_citations(
        self, claims: list[ClaimSpan], verdicts: list[ClaimVerdict], passages: dict[int, Evidence]
    ) -> list[ClaimVerdict]:
        """For a cited claim that fails, check the passages it doesn't cite. If one entails it,
        the fact is fine and the citation is wrong. The verdict stands (the cited passage still
        doesn't support the claim), but the reason says which passage does."""
        checks: list[tuple[int, int, str]] = []  # (claim position, passage number, premise)
        for n, claim in enumerate(claims):
            if not claim.citations or verdicts[n].verdict == "supported":
                continue
            for i, passage in passages.items():
                if i not in claim.citations:
                    checks += [(n, i, premise) for premise in _premises(passage)]
        if not checks:
            return verdicts
        scores = self.classify([(premise, _hypothesis(claims[n].text)) for n, _, premise in checks])
        best: dict[int, tuple[float, int]] = {}
        for (n, i, _), score in zip(checks, scores, strict=True):
            if _argmax(score) == "entailment" and score["entailment"] > best.get(n, (0.0, 0))[0]:
                best[n] = (score["entailment"], i)
        for n, (probability, i) in best.items():
            reason = (
                f"{verdicts[n].reason} Passage [{i}] does support it (p={probability:.2f}), "
                "so the citation is likely wrong."
            )
            verdicts[n] = verdicts[n].model_copy(update={"reason": reason})
        return verdicts


def _decide(
    claim: ClaimSpan, premises: list[tuple[list[int], str]], scores: list[dict[str, float]]
) -> ClaimVerdict:
    def verdict(label: str, reason: str) -> ClaimVerdict:
        return ClaimVerdict(
            claim_text=claim.text, citation_indices=claim.citations, verdict=label, reason=reason
        )

    if not premises:
        return verdict("unsupported", f"Cites passage {_refs(claim.citations)}, which does not exist.")

    entailed = [k for k, s in enumerate(scores) if _argmax(s) == "entailment"]
    if entailed:
        k = max(entailed, key=lambda i: scores[i]["entailment"])
        return verdict(
            "supported", f"Entailed by passage {_refs(premises[k][0])} (p={scores[k]['entailment']:.2f})."
        )

    contradicted = [k for k, s in enumerate(scores) if _argmax(s) == "contradiction"]
    if contradicted:
        k = max(contradicted, key=lambda i: scores[i]["contradiction"])
        return verdict(
            "unsupported",
            f"Contradicted by passage {_refs(premises[k][0])} (p={scores[k]['contradiction']:.2f}).",
        )

    k = max(range(len(scores)), key=lambda i: scores[i]["entailment"])
    refs, premise = premises[k]
    if _overlap(claim.text, premise) >= PARTIAL_OVERLAP:
        return verdict(
            "partially_supported",
            f"Passage {_refs(refs)} covers this but does not entail all of it (p={scores[k]['entailment']:.2f}).",
        )
    subject = f"Cited passage {_refs(claim.citations)} doesn't" if claim.citations else "No passage does"
    return verdict("unsupported", f"{subject} support this claim (best p={scores[k]['entailment']:.2f}).")


class HuggingFaceNLI:
    """A three-way NLI cross-encoder from the Hugging Face Hub, run with PyTorch."""

    def __init__(self, model_name: str, batch_size: int = 16, device: str = "cpu") -> None:
        torch = import_torch()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = pick_device(torch, device)
        logger.info("Loading local NLI model %s on %s (downloaded on first use)", model_name, self.device)
        self._torch = torch
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        except OSError as exc:
            raise load_failure(model_name, exc) from exc
        self.model = model.to(self.device).eval()  # float32 on every device
        config = self.model.config
        self.labels = [config.id2label[i].lower() for i in range(config.num_labels)]
        if set(self.labels) != NLI_LABELS:
            raise ConfigurationError(f"{model_name} is not a three-way NLI model (labels: {self.labels})")
        self.batch_size = batch_size

    def __call__(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        # Batch pairs of similar length together so short windows aren't padded to a full chunk.
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        results: list[dict[str, float]] = [{} for _ in pairs]
        with device_lock(self.device), self._torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                indices = order[start : start + self.batch_size]
                batch = [pairs[i] for i in indices]
                encoded = self.tokenizer(
                    [premise for premise, _ in batch],
                    [hypothesis for _, hypothesis in batch],
                    truncation="only_first",
                    max_length=512,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                probabilities = self.model(**encoded).logits.float().softmax(dim=-1).tolist()
                for i, row in zip(indices, probabilities, strict=True):
                    results[i] = dict(zip(self.labels, row, strict=True))
        return results


_cache: dict[tuple[str, str], LocalNLIVerifier] = {}
_cache_lock = threading.Lock()


def load_local_verifier(model_name: str, device: str = "auto") -> LocalNLIVerifier:
    """Load (once per process) the verifier backed by ``model_name``. On the benchmarks the
    verdicts are identical on the CPU and the Apple GPU; the GPU is about 10x faster."""
    with _cache_lock:
        if (model_name, device) not in _cache:
            _cache[model_name, device] = LocalNLIVerifier(HuggingFaceNLI(model_name, device=device))
        return _cache[model_name, device]
