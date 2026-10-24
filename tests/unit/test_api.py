from __future__ import annotations

import anthropic
import httpx2
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pinecone.errors import PineconeError, UnauthorizedError

import veritas.api.main as api_main
from veritas.api.main import ServiceProvider, create_app, resolve_web_dir, safe_upload_name
from veritas.core.errors import ConfigurationError, LLMOutputError, LLMRefusalError
from veritas.core.models import AskResponse, GroundingReport
from veritas.core.settings import settings

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


class StubService:
    def __init__(
        self, response: AskResponse | None = None, error: Exception | None = None, upload_error=None
    ) -> None:
        self.response = response or AskResponse(answer="A [1].", sources=[], retrieval=[])
        self.error = error
        self.upload_error = upload_error
        self.payloads = []
        self.uploads = []

    def add_document(self, path):
        if self.upload_error is not None:
            raise self.upload_error
        self.uploads.append(path)
        return f"uploads/{path.stem}", 3

    def available_documents(self) -> list[str]:
        return ["Q2_Report"]

    def ask(self, payload):
        self.payloads.append(payload)
        if self.error is not None:
            raise self.error
        return self.response


def client_for(service: StubService | None = None, factory=None) -> TestClient:
    return TestClient(create_app(service_factory=factory or (lambda: service)))


def test_health_reports_readiness():
    with client_for(StubService()) as client:
        assert client.get("/health").json() == {"status": "ok", "ready": True, "detail": None}


def test_missing_keys_degrade_to_503_with_the_reason():
    def factory():
        raise ConfigurationError("ANTHROPIC_API_KEY is required")

    with client_for(factory=factory) as client:
        health = client.get("/health").json()
        response = client.post("/v1/ask", json={"question": "What grew?"})

    assert health["status"] == "ok"
    assert health["ready"] is False
    assert "ANTHROPIC_API_KEY" in health["detail"]
    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_service_initialization_is_retried_after_a_cooldown(monkeypatch):
    attempts = []

    def factory():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("pinecone unreachable")
        return "service"

    provider = ServiceProvider(factory)
    with pytest.raises(HTTPException):
        provider.get()
    with pytest.raises(HTTPException):  # still cooling down: not retried
        provider.get()
    assert len(attempts) == 1

    monkeypatch.setattr("veritas.api.main.RETRY_COOLDOWN_SECONDS", 0.0)
    assert provider.get() == "service"
    assert provider.ready
    assert provider.error is None


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "hi"},
        {"question": "   "},
        {"question": "What grew?", "verification_mode": "vibes"},
        {"question": "What grew?", "top_k": 0},
        {"question": "What grew?", "top_k": 500},
    ],
)
def test_invalid_questions_are_rejected(payload):
    with client_for(StubService()) as client:
        assert client.post("/v1/ask", json=payload).status_code == 422


def test_ask_normalizes_the_request_and_returns_the_answer():
    grounding = GroundingReport(status="verified", score=1.0)
    service = StubService(AskResponse(answer="A [1].", sources=[], retrieval=[], grounding=grounding))

    with client_for(service) as client:
        response = client.post("/v1/ask", json={"question": "  What grew?  ", "selected_document": ""})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "A [1]."
    assert body["grounding"]["status"] == "verified"
    assert service.payloads[0].question == "What grew?"
    assert service.payloads[0].selected_document is None


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (LLMRefusalError("cyber"), 422),
        (LLMOutputError("cut off"), 502),
        (
            anthropic.AuthenticationError(
                "bad key", response=httpx2.Response(401, request=REQUEST), body=None
            ),
            503,
        ),
        (anthropic.NotFoundError("no model", response=httpx2.Response(404, request=REQUEST), body=None), 503),
        (
            anthropic.InternalServerError("oops", response=httpx2.Response(500, request=REQUEST), body=None),
            502,
        ),
        (anthropic.APITimeoutError(request=REQUEST), 504),
        (anthropic.APIConnectionError(request=REQUEST), 502),
        (UnauthorizedError(), 503),
        (PineconeError("boom"), 502),
    ],
)
def test_upstream_failures_become_clean_errors(error, status):
    with client_for(StubService(error=error)) as client:
        response = client.post("/v1/ask", json={"question": "What grew?"})
    assert response.status_code == status
    assert response.json()["detail"]
    assert "Traceback" not in response.text


def test_rate_limits_pass_retry_after_through():
    error = anthropic.RateLimitError(
        "slow down", response=httpx2.Response(429, headers={"retry-after": "7"}, request=REQUEST), body=None
    )
    with client_for(StubService(error=error)) as client:
        response = client.post("/v1/ask", json={"question": "What grew?"})
    assert response.status_code == 429
    assert response.headers["retry-after"] == "7"


def test_documents():
    with client_for(StubService()) as client:
        assert client.get("/v1/documents").json() == {"documents": ["Q2_Report"]}


EXPORT = {
    "records": [
        {
            "document": "All documents",
            "question": "What grew?",
            "answer": "Revenue — up 12% [1].",
            "grounding_score": 0.5,
            "flagged_claims": [{"claim_text": "up 12%", "citation_indices": [1], "verdict": "unsupported"}],
        }
    ]
}


def test_csv_export():
    with client_for(StubService()) as client:
        response = client.post("/v1/export/csv", json=EXPORT)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert 'filename="research_session.csv"' in response.headers["content-disposition"]
    assert response.content.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM for spreadsheet apps
    assert "grounding_score" in response.content.decode("utf-8-sig").splitlines()[0]


def test_pdf_export():
    with client_for(StubService()) as client:
        response = client.post("/v1/export/pdf", json=EXPORT)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")


def test_export_rejects_oversized_sessions():
    payload = {"records": [{"document": "d", "question": "q", "answer": "a"}] * 501}
    with client_for(StubService()) as client:
        assert client.post("/v1/export/csv", json=payload).status_code == 422


def test_console_is_served():
    with client_for(StubService()) as client:
        index = client.get("/")
        script = client.get("/static/app.js")
    assert index.status_code == 200
    assert "text/html" in index.headers["content-type"]
    assert script.status_code == 200


def test_web_dir_setting_takes_precedence(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<!doctype html>")
    monkeypatch.setattr(settings, "web_dir", tmp_path)
    assert resolve_web_dir() == tmp_path


def test_without_console_files_the_root_describes_the_api(monkeypatch):
    monkeypatch.setattr("veritas.api.main.resolve_web_dir", lambda: None)
    with client_for(StubService()) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "VERITAS API" in response.json()["message"]


# -- backends and uploads ------------------------------------------------------------------


def test_config_without_an_api_key_offers_only_local_verification(monkeypatch):
    monkeypatch.setattr(api_main, "local_models_installed", lambda: True)
    with client_for(StubService()) as client:
        config = client.get("/v1/config").json()
    assert config["llm_provider"] == "local"
    assert config["llm_model"] == settings.local_llm_model
    assert config["verification_modes"] == ["local-nli"]
    assert config["default_verification_mode"] == "local-nli"
    assert config["uploads_enabled"] is True


def test_config_with_claude_and_pinecone(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_provider", "claude")
    monkeypatch.setattr(settings, "vector_store", "pinecone")
    monkeypatch.setattr(api_main, "local_models_installed", lambda: True)
    with client_for(StubService()) as client:
        config = client.get("/v1/config").json()
    assert config["llm_model"] == settings.chat_model
    assert config["verification_modes"] == ["local-nli", "nli", "llm-as-judge"]
    assert config["default_verification_mode"] == "nli"
    assert config["uploads_enabled"] is False


def test_the_cloud_image_without_local_models_offers_only_claude_modes(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(api_main, "local_models_installed", lambda: False)
    with client_for(StubService()) as client:
        config = client.get("/v1/config").json()
    assert config["verification_modes"] == ["nli", "llm-as-judge"]
    assert config["default_verification_mode"] == "nli"


def test_upload_saves_and_indexes_the_file():
    service = StubService()
    with client_for(service) as client:
        response = client.post("/v1/documents", files={"file": ("memo.md", b"# Offsite\nJune 3 in Lisbon.")})
    assert response.status_code == 201
    assert response.json() == {"document": "uploads/memo", "chunks": 3}
    (saved,) = service.uploads
    assert saved == settings.documents_dir / "uploads" / "memo.md"
    assert saved.read_bytes() == b"# Offsite\nJune 3 in Lisbon."


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("../../etc/passwd.md", "passwd.md"),
        ("..\\..\\secret.txt", "secret.txt"),
        ("Q2 report (final).pdf", "Q2 report _final_.pdf"),
        ("", ""),
    ],
)
def test_upload_names_cannot_escape_the_uploads_folder(filename, expected):
    assert safe_upload_name(filename) == expected


def test_upload_rejects_unsupported_files():
    with client_for(StubService()) as client:
        assert client.post("/v1/documents", files={"file": ("tool.exe", b"MZ")}).status_code == 400
        assert client.post("/v1/documents", files={"file": (".md", b"x")}).status_code == 400


def test_upload_rejects_oversized_files(monkeypatch):
    monkeypatch.setattr("veritas.api.main.MAX_UPLOAD_BYTES", 10)
    with client_for(StubService()) as client:
        assert client.post("/v1/documents", files={"file": ("big.txt", b"x" * 11)}).status_code == 413


def test_upload_needs_the_local_store(monkeypatch):
    monkeypatch.setattr(settings, "vector_store", "pinecone")
    with client_for(StubService()) as client:
        assert client.post("/v1/documents", files={"file": ("memo.md", b"x")}).status_code == 400


def test_an_unreadable_upload_is_reported_and_removed():
    service = StubService(upload_error=ValueError("PDF is password-protected"))
    with client_for(service) as client:
        response = client.post("/v1/documents", files={"file": ("locked.pdf", b"%PDF-1.7")})
    assert response.status_code == 422
    assert "password-protected" in response.json()["detail"]
    assert not (settings.documents_dir / "uploads" / "locked.pdf").exists()


def test_missing_local_dependencies_become_a_503():
    service = StubService(error=ConfigurationError('Local models need: pip install -e ".[local]"'))
    with client_for(service) as client:
        response = client.post("/v1/ask", json={"question": "What grew?"})
    assert response.status_code == 503
    assert ".[local]" in response.json()["detail"]
