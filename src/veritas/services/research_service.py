"""Orchestrates retrieval, answer synthesis and grounding verification for one question."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from veritas.agents.arxiv_agent import ArxivAgent
from veritas.agents.document_selection_agent import DocumentSelectionAgent
from veritas.agents.rag_agent import RagAgent
from veritas.agents.web_search_agent import WebSearchAgent
from veritas.core.clients.claude import ClaudeClient
from veritas.core.clients.pinecone import PineconeStore
from veritas.core.errors import ConfigurationError
from veritas.core.evidence import build_evidence
from veritas.core.grounding import GroundingVerifier
from veritas.core.models import AskRequest, AskResponse, SourceItem, VerificationMode
from veritas.core.settings import settings
from veritas.local.ingest import document_name, index_file, open_default_store
from veritas.local.llm import LocalLLM
from veritas.local.store import LocalVectorStore

EXTERNAL_RESULTS_PER_SOURCE = 3
NO_EVIDENCE_ANSWER = (
    "No relevant evidence was found in the indexed documents or the enabled external sources, "
    "so this question cannot be answered from sources. Try rephrasing it, choosing a different "
    "document, or enabling web and arXiv search."
)


def _results(future: Future[list[SourceItem]] | None) -> list[SourceItem]:
    return future.result() if future is not None else []


class ResearchService:
    def __init__(
        self,
        *,
        store: Any,
        llm: Any,
        verifier: GroundingVerifier | None = None,
        web_agent: WebSearchAgent | None = None,
        arxiv_agent: ArxivAgent | None = None,
    ) -> None:
        self.store = store
        self.llm = llm
        self.verifier = verifier if verifier is not None else GroundingVerifier(None)
        self.doc_agent = DocumentSelectionAgent(store)
        self.rag_agent = RagAgent(store, llm)
        self.web_agent = web_agent if web_agent is not None else WebSearchAgent()
        self.arxiv_agent = arxiv_agent if arxiv_agent is not None else ArxivAgent()

    @classmethod
    def from_settings(cls) -> ResearchService:
        """Wire the configured backends: local models by default, Claude and Pinecone on request."""
        claude = ClaudeClient() if settings.claude_available else None
        if settings.llm_provider == "claude":
            if claude is None:
                raise ConfigurationError("LLM_PROVIDER=claude needs ANTHROPIC_API_KEY")
            llm: Any = claude
        else:
            llm = LocalLLM(
                settings.local_llm_model,
                device=settings.local_device,
                max_new_tokens=settings.local_max_new_tokens,
            )
        store = PineconeStore() if settings.vector_store == "pinecone" else open_default_store()
        return cls(store=store, llm=llm, verifier=GroundingVerifier(claude))

    @property
    def default_verification_mode(self) -> VerificationMode:
        return "nli" if self.verifier.client is not None else "local-nli"

    def warm_up(self) -> None:
        """Load local models ahead of the first question (each step is a no-op for cloud backends)."""
        for component in (self.store, self.llm, self.verifier):
            warm = getattr(component, "warm_up", None)
            if warm is not None:
                warm()

    def available_documents(self) -> list[str]:
        return self.doc_agent.fetch_documents()

    def add_document(self, path: Path) -> tuple[str, int]:
        """Index an uploaded file into the local store; returns (document name, chunks)."""
        if not isinstance(self.store, LocalVectorStore):
            raise ConfigurationError("Uploading documents needs VECTOR_STORE=local")
        chunks = index_file(self.store, path, settings.documents_dir, force=True) or 0
        self.store.save()
        return document_name(path, settings.documents_dir), chunks

    def ask(self, payload: AskRequest) -> AskResponse:
        question = payload.question
        # The three retrievals are independent; run them concurrently.
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="veritas-retrieval") as pool:
            hits_future = pool.submit(
                self.rag_agent.retrieve, question, payload.selected_document, payload.top_k
            )
            web_future = (
                pool.submit(self.web_agent.search, question, EXTERNAL_RESULTS_PER_SOURCE)
                if payload.include_web
                else None
            )
            arxiv_future = (
                pool.submit(self.arxiv_agent.search, question, EXTERNAL_RESULTS_PER_SOURCE)
                if payload.include_arxiv
                else None
            )
            hits = hits_future.result()
            sources = [*_results(web_future), *_results(arxiv_future)]

        # One numbering for synthesis and verification: chunks are [1]..[n], external sources follow.
        evidence = build_evidence(hits, sources)
        if evidence.passages:
            answer = self.rag_agent.answer(question, evidence.passages)
        else:
            answer = NO_EVIDENCE_ANSWER

        grounding = None
        if payload.verify_grounding:
            mode = payload.verification_mode or self.default_verification_mode
            grounding = self.verifier.verify(answer, evidence.passages, mode)
        return AskResponse(
            answer=answer, sources=evidence.sources, retrieval=evidence.hits, grounding=grounding
        )
