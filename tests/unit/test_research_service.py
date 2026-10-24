from __future__ import annotations

from types import SimpleNamespace

import pytest
from pinecone.errors import PineconeError

import veritas.services.research_service as research_module
from veritas.agents.rag_agent import CITATION_EXAMPLE, RagAgent
from veritas.core.errors import ConfigurationError
from veritas.core.grounding import GroundingVerifier, _ClaimJudgment, _GroundingJudgment
from veritas.core.models import AskRequest, SourceItem
from veritas.core.settings import settings
from veritas.local.store import LocalVectorStore
from veritas.services.research_service import NO_EVIDENCE_ANSWER, ResearchService


def match(id_: str, document: str, page: int, content: str) -> SimpleNamespace:
    # Pinecone returns numeric metadata as floats.
    return SimpleNamespace(
        id=id_, score=0.9, metadata={"document": document, "page_num": float(page), "content": content}
    )


class FakeStore:
    def __init__(self, matches=(), error: Exception | None = None) -> None:
        self.matches = list(matches)
        self.error = error
        self.searches: list[dict] = []

    def search(self, question, top_k=5, document=None):
        self.searches.append({"question": question, "top_k": top_k, "document": document})
        if self.error:
            raise self.error
        return self.matches

    def list_documents(self) -> list[str]:
        return ["Q2"]


class FakeLLM:
    def __init__(self) -> None:
        self.text_calls: list[dict] = []
        self.structured_calls: list[dict] = []

    def generate_text(self, **kwargs) -> str:
        self.text_calls.append(kwargs)
        return "Revenue grew 12% [1]."

    def generate_structured(self, **kwargs):
        self.structured_calls.append(kwargs)
        return _GroundingJudgment(
            claims=[
                _ClaimJudgment(
                    claim_text="Revenue grew 12%", citation_indices=[1], verdict="supported", reason="ok"
                )
            ]
        )


class FakeSearch:
    def __init__(self, items=()) -> None:
        self.items = list(items)
        self.calls = 0

    def search(self, query: str, max_results: int):
        self.calls += 1
        return self.items


def make_service(matches=(), web=(), arxiv=(), store_error=None, claude_verifier=True) -> ResearchService:
    llm = FakeLLM()
    return ResearchService(
        store=FakeStore(matches, store_error),
        llm=llm,
        verifier=GroundingVerifier(llm if claude_verifier else None),
        web_agent=FakeSearch(web),
        arxiv_agent=FakeSearch(arxiv),
    )


CHUNKS = [match("v1", "Q2", 1, "Revenue grew 12%."), match("v2", "Q2", 2, "Costs fell 3%.")]
WEB = [SourceItem(title="Wiki", url="https://w.example", snippet="web text", kind="web")]
ARXIV = [SourceItem(title="Paper", snippet="abstract", kind="arxiv")]


def test_answer_and_verification_share_one_evidence_numbering():
    service = make_service(CHUNKS, WEB, ARXIV)

    response = service.ask(AskRequest(question="How did revenue change?", selected_document="Q2", top_k=4))

    assert service.store.searches == [{"question": "How did revenue change?", "top_k": 4, "document": "Q2"}]
    assert [h.citation_index for h in response.retrieval] == [1, 2]
    assert [s.citation_index for s in response.sources] == [3, 4]
    assert response.retrieval[0].page_num == 1
    answer_prompt = service.llm.text_calls[0]["prompt"]
    verify_prompt = service.llm.structured_calls[0]["prompt"]
    for n in range(1, 5):
        assert f'<passage index="{n}"' in answer_prompt
        assert f'<passage index="{n}"' in verify_prompt
    assert "Revenue grew 12% [1]." in verify_prompt
    assert response.answer == "Revenue grew 12% [1]."
    assert response.grounding.status == "verified"
    assert response.grounding.mode == "nli"
    assert response.grounding.score == 1.0


def test_the_answer_prompt_includes_a_worked_citation_example():
    service = make_service(CHUNKS)
    service.ask(AskRequest(question="Revenue?"))
    assert service.llm.text_calls[0]["examples"] == [CITATION_EXAMPLE]


def test_without_any_evidence_the_model_is_not_asked():
    service = make_service()

    response = service.ask(AskRequest(question="Anything at all?"))

    assert response.answer == NO_EVIDENCE_ANSWER
    assert service.llm.text_calls == []
    assert service.llm.structured_calls == []
    assert response.grounding.status == "no_context"


def test_disabled_sources_are_not_queried():
    service = make_service(CHUNKS, WEB, ARXIV)

    response = service.ask(
        AskRequest(question="Revenue?", include_web=False, include_arxiv=False, verify_grounding=False)
    )

    assert service.web_agent.calls == 0
    assert service.arxiv_agent.calls == 0
    assert response.sources == []
    assert response.grounding is None


def test_external_sources_alone_are_enough_evidence():
    service = make_service(web=WEB)
    response = service.ask(AskRequest(question="What is a transformer?"))
    assert response.answer == "Revenue grew 12% [1]."
    assert response.sources[0].citation_index == 1


def test_vector_store_errors_propagate():
    service = make_service(store_error=PineconeError("index unavailable"))
    with pytest.raises(PineconeError):
        service.ask(AskRequest(question="Revenue?"))


def test_without_claude_the_default_verification_is_local(monkeypatch):
    service = make_service(CHUNKS, claude_verifier=False)
    calls = []

    class FakeLocal:
        def judge(self, answer, evidence):
            calls.append(answer)
            return []

    service.verifier._local = FakeLocal()

    response = service.ask(AskRequest(question="Revenue?"))

    assert service.default_verification_mode == "local-nli"
    assert response.grounding.mode == "local-nli"
    assert calls == ["Revenue grew 12% [1]."]


def test_an_explicit_verification_mode_wins():
    service = make_service(CHUNKS)
    response = service.ask(AskRequest(question="Revenue?", verification_mode="llm-as-judge"))
    assert response.grounding.mode == "llm-as-judge"


def test_rag_agent_tolerates_sparse_metadata():
    store = FakeStore([SimpleNamespace(id="x", score=None, metadata=None)])
    (hit,) = RagAgent(store, FakeLLM()).retrieve("question")
    assert hit.document == "unknown"
    assert hit.page_num is None
    assert hit.score == 0.0
    assert hit.content == ""


def test_rag_agent_previews_long_chunks():
    store = FakeStore([match("x", "d", 1, "word " * 200)])
    (hit,) = RagAgent(store, FakeLLM()).retrieve("question")
    assert len(hit.preview) <= 241
    assert hit.preview.endswith("…")
    assert hit.content.startswith("word word")


# -- wiring from settings --------------------------------------------------------------------


def test_from_settings_defaults_to_the_local_stack(monkeypatch, tmp_path, embedder):
    built = {}
    monkeypatch.setattr(
        research_module,
        "LocalLLM",
        lambda name, **kwargs: built.setdefault("llm", (name, kwargs)) and FakeLLM(),
    )
    monkeypatch.setattr(research_module, "open_default_store", lambda: LocalVectorStore(tmp_path, embedder))

    service = ResearchService.from_settings()

    assert built["llm"][0] == settings.local_llm_model
    assert isinstance(service.store, LocalVectorStore)
    assert service.verifier.client is None
    assert service.default_verification_mode == "local-nli"


def test_from_settings_uses_claude_and_pinecone_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "claude")
    monkeypatch.setattr(settings, "vector_store", "pinecone")
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    claude, pinecone = object(), FakeStore()
    monkeypatch.setattr(research_module, "ClaudeClient", lambda: claude)
    monkeypatch.setattr(research_module, "PineconeStore", lambda: pinecone)

    service = ResearchService.from_settings()

    assert service.llm is claude
    assert service.store is pinecone
    assert service.verifier.client is claude
    assert service.default_verification_mode == "nli"


def test_claude_provider_without_a_key_is_a_configuration_error(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "claude")
    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
        ResearchService.from_settings()


def test_uploaded_documents_are_indexed_into_the_local_store(tmp_path, embedder):
    store = LocalVectorStore(tmp_path / "index", embedder)
    service = ResearchService(store=store, llm=FakeLLM())
    upload = settings.documents_dir / "uploads" / "memo.md"
    upload.parent.mkdir(parents=True)
    upload.write_text("The offsite is on June 3 in Lisbon.")

    document, chunks = service.add_document(upload)

    assert (document, chunks) == ("uploads/memo", 1)
    assert store.list_documents() == ["uploads/memo"]
    assert (tmp_path / "index" / "veritas_index.npz").exists()


def test_uploads_need_the_local_store(tmp_path):
    service = ResearchService(store=FakeStore(), llm=FakeLLM())
    with pytest.raises(ConfigurationError, match="VECTOR_STORE=local"):
        service.add_document(tmp_path / "memo.md")


def test_warm_up_loads_every_component_that_supports_it():
    loaded = []

    class Warmable:
        def __init__(self, name):
            self.name = name

        def warm_up(self):
            loaded.append(self.name)

    service = ResearchService(store=Warmable("store"), llm=Warmable("llm"), verifier=Warmable("verifier"))
    service.warm_up()
    assert loaded == ["store", "llm", "verifier"]


def test_previews_drop_markdown_markup():
    store = FakeStore(
        [match("x", "d", 1, "# Q2 Results\n\n*Fictional sample.*\n\n## People\n\nHeadcount was 2,150.")]
    )
    (hit,) = RagAgent(store, FakeLLM()).retrieve("question")
    assert hit.preview == "Q2 Results Fictional sample. People Headcount was 2,150."
    assert hit.content.startswith("# Q2 Results")  # the model still gets the original text
