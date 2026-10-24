"""FastAPI application: the research API plus the static web console."""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import anthropic
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pinecone.errors import (
    ForbiddenError,
    PineconeError,
    PineconeTimeoutError,
    UnauthorizedError,
)
from pinecone.errors import (
    RateLimitError as PineconeRateLimitError,
)
from starlette.concurrency import run_in_threadpool

from veritas import __version__
from veritas.core.errors import ConfigurationError, LLMOutputError, LLMRefusalError
from veritas.core.exporters import to_csv, to_pdf_bytes
from veritas.core.logging_config import setup_logging
from veritas.core.models import AskRequest, AskResponse, ExportPayload
from veritas.core.settings import settings
from veritas.local.ingest import SUPPORTED_SUFFIXES
from veritas.local.runtime import local_models_installed
from veritas.services.research_service import ResearchService

setup_logging(settings.log_level)
logger = logging.getLogger(__name__)

RETRY_COOLDOWN_SECONDS = 30.0
MAX_UPLOAD_BYTES = 20_000_000
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._ -]+")


def safe_upload_name(filename: str | None) -> str:
    """Keep only the final path component and plain characters, so an upload can't escape
    the uploads folder ("../../etc/x.md" becomes "x.md")."""
    name = Path((filename or "").replace("\\", "/")).name
    return _UNSAFE_FILENAME_CHARS.sub("_", name).strip(" .")


def resolve_web_dir() -> Path | None:
    """Locate the web console: WEB_DIR if set, else the source tree, else ./ui/web."""
    candidates = [settings.web_dir] if settings.web_dir else []
    candidates += [
        Path(__file__).resolve().parents[3] / "ui" / "web",  # source checkout / editable install
        Path.cwd() / "ui" / "web",  # installed package started from the repo or image root
    ]
    return next((path for path in candidates if (path / "index.html").is_file()), None)


class ServiceProvider:
    """Builds the ResearchService on first use and retries after a failure, so a missing
    key or a transient outage at boot doesn't leave the API broken until a restart."""

    def __init__(self, factory: Callable[[], ResearchService]) -> None:
        self._factory = factory
        self._service: ResearchService | None = None
        self._lock = threading.Lock()
        self._failed_at = 0.0
        self.error: str | None = None

    @property
    def ready(self) -> bool:
        return self._service is not None

    def get(self) -> ResearchService:
        if self._service is not None:
            return self._service
        with self._lock:
            if self._service is not None:
                return self._service
            if self.error and time.monotonic() - self._failed_at < RETRY_COOLDOWN_SECONDS:
                raise HTTPException(status_code=503, detail=self.error)
            try:
                service = self._factory()
            except ConfigurationError as exc:
                raise self._failure(f"Service unavailable: {exc}") from exc
            except Exception as exc:
                logger.exception("Could not initialize the research service")
                raise self._failure(
                    "Service unavailable: initialization failed; see the server logs."
                ) from exc
            self._service, self.error = service, None
            return service

    def _failure(self, message: str) -> HTTPException:
        self.error = message
        self._failed_at = time.monotonic()
        return HTTPException(status_code=503, detail=message)


def get_service(request: Request) -> ResearchService:
    return request.app.state.services.get()


ServiceDep = Annotated[ResearchService, Depends(get_service)]


def _error(status: int, detail: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": detail}, headers=headers)


async def _claude_error(_: Request, exc: Exception) -> JSONResponse:
    logger.error("Claude API error: %s", exc)
    # Most specific first: APITimeoutError subclasses APIConnectionError.
    if isinstance(exc, anthropic.RateLimitError):
        retry_after = exc.response.headers.get("retry-after")
        headers = {"Retry-After": retry_after} if retry_after else None
        return _error(429, "The Claude API rate limit was reached; retry shortly.", headers)
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return _error(503, "The Claude API rejected the configured ANTHROPIC_API_KEY.")
    if isinstance(exc, anthropic.NotFoundError):
        return _error(503, "The configured Claude model was not found; check CHAT_MODEL and VERIFIER_MODEL.")
    if isinstance(exc, anthropic.APITimeoutError):
        return _error(504, "The Claude API timed out; retry shortly.")
    if isinstance(exc, anthropic.APIConnectionError):
        return _error(502, "Could not reach the Claude API.")
    if isinstance(exc, anthropic.APIStatusError):
        return _error(502, f"The Claude API returned an error (HTTP {exc.status_code}).")
    return _error(502, "The Claude API request failed.")


async def _pinecone_error(_: Request, exc: Exception) -> JSONResponse:
    logger.error("Pinecone error: %s", exc)
    if isinstance(exc, (UnauthorizedError, ForbiddenError)):
        return _error(503, "Pinecone rejected the configured PINECONE_API_KEY.")
    if isinstance(exc, PineconeRateLimitError):
        return _error(429, "The Pinecone rate limit was reached; retry shortly.")
    if isinstance(exc, PineconeTimeoutError):
        return _error(504, "The vector store timed out; retry shortly.")
    return _error(502, "The vector store request failed.")


async def _refusal_error(_: Request, exc: Exception) -> JSONResponse:
    logger.warning("Model declined a request: %s", exc)
    return _error(422, str(exc))


async def _llm_output_error(_: Request, exc: Exception) -> JSONResponse:
    logger.error("Unusable model output: %s", exc)
    return _error(502, "The language model returned an unusable response; please retry.")


async def _configuration_error(_: Request, exc: Exception) -> JSONResponse:
    # e.g. local models requested but the [local] extra isn't installed, or a model download failed
    logger.error("Configuration problem: %s", exc)
    return _error(503, str(exc))


def _warm_up(warm: Callable[[], None]) -> None:
    try:
        warm()
        logger.info("Models loaded and ready")
    except Exception:
        logger.warning("Model warm-up failed; models will load on first use", exc_info=True)


def _attachment(filename: str) -> dict[str, str]:
    return {"Content-Disposition": f'attachment; filename="{filename}"'}


def create_app(service_factory: Callable[[], ResearchService] = ResearchService.from_settings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        try:  # build the service now so the first request doesn't pay for client setup
            service = await run_in_threadpool(application.state.services.get)
        except HTTPException:
            logger.warning("Starting in degraded mode. %s", application.state.services.error)
        else:
            warm = getattr(service, "warm_up", None)
            if warm is not None:  # load local models in the background; the API answers meanwhile
                threading.Thread(target=_warm_up, args=(warm,), name="veritas-warm-up", daemon=True).start()
        yield

    app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)
    app.state.services = ServiceProvider(service_factory)
    app.add_exception_handler(anthropic.APIError, _claude_error)
    app.add_exception_handler(PineconeError, _pinecone_error)
    app.add_exception_handler(LLMRefusalError, _refusal_error)
    app.add_exception_handler(LLMOutputError, _llm_output_error)
    app.add_exception_handler(ConfigurationError, _configuration_error)

    web_dir = resolve_web_dir()
    if web_dir is not None:
        app.mount("/static", StaticFiles(directory=web_dir), name="static")
    else:
        logger.warning("Web console files not found; set WEB_DIR to serve the UI")

    @app.get("/", include_in_schema=False)
    def index() -> Response:
        if web_dir is not None:
            return FileResponse(web_dir / "index.html")
        return JSONResponse({"message": "VERITAS API is running. API docs: /docs"})

    @app.get("/health")
    def health(request: Request) -> dict[str, object]:
        services: ServiceProvider = request.app.state.services
        return {"status": "ok", "ready": services.ready, "detail": services.error}

    @app.get("/v1/config")
    def config() -> dict[str, object]:
        """Which backends are active, so the console can offer only what works."""
        claude = settings.claude_available
        return {
            "llm_provider": settings.llm_provider,
            "llm_model": settings.local_llm_model
            if settings.llm_provider == "local"
            else settings.chat_model,
            "vector_store": settings.vector_store,
            "verification_modes": [
                *(["local-nli"] if local_models_installed() else []),  # the cloud image omits them
                *(["nli", "llm-as-judge"] if claude else []),
            ],
            "default_verification_mode": "nli" if claude else "local-nli",
            "uploads_enabled": settings.vector_store == "local",
        }

    @app.get("/v1/documents")
    def documents(service: ServiceDep) -> dict[str, list[str]]:
        return {"documents": service.available_documents()}

    @app.post("/v1/documents", status_code=201)
    async def upload_document(file: UploadFile, service: ServiceDep) -> dict[str, object]:
        if settings.vector_store != "local":
            raise HTTPException(
                400, "Uploads need VECTOR_STORE=local; with Pinecone, ingest through the S3 pipeline."
            )
        name = safe_upload_name(file.filename)
        if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES or not Path(name).stem:
            raise HTTPException(
                400, f"Upload a PDF, Markdown or text file ({', '.join(SUPPORTED_SUFFIXES)})."
            )
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"Files are limited to {MAX_UPLOAD_BYTES // 1_000_000} MB.")
        target = settings.documents_dir / "uploads" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        try:
            document, chunks = await run_in_threadpool(service.add_document, target)
        except ConfigurationError:
            raise
        except Exception as exc:  # untrusted input: report unreadable files instead of failing
            logger.warning("Could not index upload %s", name, exc_info=True)
            target.unlink(missing_ok=True)
            raise HTTPException(422, f"Could not read {name}: {exc}") from exc
        return {"document": document, "chunks": chunks}

    @app.post("/v1/ask")
    def ask(payload: AskRequest, service: ServiceDep) -> AskResponse:
        return service.ask(payload)

    @app.post("/v1/export/csv")
    def export_csv(payload: ExportPayload) -> Response:
        # A byte-order mark lets spreadsheet apps detect UTF-8 instead of garbling non-ASCII text.
        content = ("\N{BYTE ORDER MARK}" + to_csv(payload.records)).encode("utf-8")
        return Response(
            content, media_type="text/csv; charset=utf-8", headers=_attachment("research_session.csv")
        )

    @app.post("/v1/export/pdf")
    def export_pdf(payload: ExportPayload) -> Response:
        return Response(
            to_pdf_bytes(payload.records),
            media_type="application/pdf",
            headers=_attachment("research_session.pdf"),
        )

    return app


app = create_app()
