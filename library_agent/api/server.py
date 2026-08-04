"""HTTP API for chatting with and ingesting books.

The module intentionally keeps the API boundary thin: the agent graph, vector
store and ingestion pipeline remain the source of truth for domain behavior.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from time import perf_counter
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field, field_validator

from library_agent.agent.graph import BookAgent
from library_agent.api.logging_config import configure_logging, request_id_context
from library_agent.api.storage import LibraryStorage
from library_agent.infrastructure.vector_store import get_global_vector_store
from library_agent.rag_engine.document_loader import get_pdf_page_count
from library_agent.rag_engine.vector_pipeline import VectorPipeline


configure_logging()
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOOK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,62}$")
SUPPORTED_SUFFIXES = {".pdf", ".txt"}
INTERNAL_OUTPUT_MARKERS = (
    "更新后的聊天摘要",
    "【更新后的聊天摘要】",
    "之前对话的背景摘要",
    "【之前对话的背景摘要】",
    "用户起初问",
    "用户随后问",
    "用户接着说",
    "助手说明",
    "我再帮你继续查",
)


def _resolve_path(value: str | None, default: Path) -> Path:
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class RuntimeSettings:
    vector_db_path: Path
    checkpoint_db_path: Path
    library_db_path: Path
    upload_dir: Path
    cors_origins: tuple[str, ...]

    @classmethod
    def from_environment(cls) -> "RuntimeSettings":
        raw_origins = os.getenv("CORS_ORIGINS", "*")
        origins = tuple(origin.strip() for origin in raw_origins.split(",") if origin.strip()) or ("*",)
        return cls(
            vector_db_path=_resolve_path(
                os.getenv("VECTOR_DB_PATH"),
                PROJECT_ROOT / "rag_engine" / "db_vector_data",
            ),
            checkpoint_db_path=_resolve_path(
                os.getenv("CHECKPOINT_DB_PATH"),
                PROJECT_ROOT / "agent" / "checkpoints.sqlite",
            ),
            library_db_path=_resolve_path(
                os.getenv("LIBRARY_DB_PATH"),
                PROJECT_ROOT / "data" / "library.sqlite3",
            ),
            upload_dir=_resolve_path(
                os.getenv("UPLOAD_DIR"),
                PROJECT_ROOT / "uploads",
            ),
            cors_origins=origins,
        )


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=20_000)
    book_id: str = Field(..., min_length=3, max_length=63)
    thread_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message 不能为空")
        return value

    @field_validator("book_id")
    @classmethod
    def book_id_must_be_chroma_safe(cls, value: str) -> str:
        return validate_book_id(value)


class UploadResponse(BaseModel):
    book_id: str
    filename: str
    chunks: int
    pages: int | None = None
    message: str


class BookResponse(BaseModel):
    id: str
    title: str
    meta: str
    indexed: bool


class HistoryMessageResponse(BaseModel):
    id: str
    role: str
    content: str


def validate_book_id(value: str) -> str:
    value = value.strip()
    if not BOOK_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "book_id 必须是 3-63 位，以字母或数字开头和结尾，只能包含字母、数字、下划线和连字符"
        )
    return value


def _sse(data: dict, event: str | None = None) -> str:
    """Encode one JSON payload as an SSE frame."""
    lines = []
    if event:
        lines.append(f"event: {event}")
    payload = json.dumps(data, ensure_ascii=False)
    lines.extend(f"data: {line}" for line in payload.splitlines() or [""])
    return "\n".join(lines) + "\n\n"


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
    return "".join(text_parts)


class _AssistantOutputFilter:
    """Prevent internal memory headings from leaking into user-visible output."""

    def __init__(self) -> None:
        self._pending = ""
        self._blocked = False

    def feed(self, text: str) -> str:
        if not text or self._blocked:
            return ""

        self._pending += text
        marker_index = min(
            (
                self._pending.find(marker)
                for marker in INTERNAL_OUTPUT_MARKERS
                if self._pending.find(marker) >= 0
            ),
            default=-1,
        )
        if marker_index >= 0:
            visible = self._pending[:marker_index]
            self._pending = ""
            self._blocked = True
            return visible

        # Keep a possible partial marker buffered because streaming chunks can
        # split a Chinese heading across multiple model events.
        keep = 0
        for marker in INTERNAL_OUTPUT_MARKERS:
            for prefix_length in range(1, min(len(marker), len(self._pending) + 1)):
                if self._pending.endswith(marker[:prefix_length]):
                    keep = max(keep, prefix_length)
        if keep == 0:
            visible, self._pending = self._pending, ""
        else:
            visible = self._pending[:-keep]
            self._pending = self._pending[-keep:]
        return visible

    def flush(self) -> str:
        if self._blocked:
            return ""
        visible, self._pending = self._pending, ""
        return visible


def _sanitize_assistant_output(content: str) -> str:
    output_filter = _AssistantOutputFilter()
    return output_filter.feed(content) + output_filter.flush()


def _agent_state(request: ChatRequest) -> tuple[dict, str]:
    thread_id = request.thread_id or uuid4().hex
    config = {
        "configurable": {
            "thread_id": thread_id,
            "book_id": request.book_id,
        }
    }
    return config, thread_id


async def _chat_events(request: Request, payload: ChatRequest) -> AsyncGenerator[str, None]:
    request_id = getattr(request.state, "request_id", uuid4().hex)
    request_token = request_id_context.set(request_id)
    started_at = perf_counter()
    config, thread_id = _agent_state(payload)
    config["configurable"]["request_id"] = request_id
    emitted_text = False
    status_emitted = False
    token_events = 0
    answer = ""
    answer_filter = _AssistantOutputFilter()

    try:
        agent_app = getattr(request.app.state, "agent_app", None)
        logger.info(
            "chat_started",
            extra={
                "event": "chat_started",
                "book_id": payload.book_id,
                "thread_id": thread_id,
                "message_chars": len(payload.message),
            },
        )
        if agent_app is None:
            logger.error(
                "chat_rejected_agent_not_ready",
                extra={"event": "chat_rejected", "book_id": payload.book_id, "thread_id": thread_id},
            )
            yield _sse({"type": "error", "message": "Agent 尚未完成初始化"}, event="error")
            return

        yield _sse({"type": "start", "thread_id": thread_id, "book_id": payload.book_id}, event="start")

        library_storage = getattr(request.app.state, "library_storage", None)
        book_title = payload.book_id
        if library_storage is not None:
            book_title = await asyncio.to_thread(library_storage.get_book_title, payload.book_id) or payload.book_id
            await asyncio.to_thread(library_storage.ensure_thread, thread_id, payload.book_id)
            await asyncio.to_thread(
                library_storage.add_message,
                thread_id,
                payload.book_id,
                "user",
                payload.message,
            )

        async for part in agent_app.astream(
            {
                "messages": [HumanMessage(content=payload.message)],
                "book_title": book_title,
            },
            config=config,
            stream_mode=["messages", "updates"],
            version="v2",
        ):
            if await request.is_disconnected():
                logger.warning(
                    "chat_client_disconnected",
                    extra={
                        "event": "chat_client_disconnected",
                        "book_id": payload.book_id,
                        "thread_id": thread_id,
                        "answer_chars": len(answer),
                        "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                    },
                )
                return

            part_type = part.get("type")
            part_data = part.get("data") or {}

            if part_type == "messages":
                # v2 messages stream data is (message_chunk, metadata).
                message_chunk = part_data[0] if isinstance(part_data, (tuple, list)) else None
                if message_chunk is None:
                    continue

                metadata = part_data[1] if isinstance(part_data, (tuple, list)) and len(part_data) > 1 else {}
                if isinstance(message_chunk, ToolMessage) or (
                    isinstance(metadata, dict) and metadata.get("langgraph_node") == "action_tools"
                ):
                    # Tool results contain the raw retrieved context. They
                    # are internal state, not user-visible answer tokens.
                    continue

                tool_chunks = (
                    getattr(message_chunk, "tool_call_chunks", None)
                    or getattr(message_chunk, "tool_calls", None)
                )
                if tool_chunks and not status_emitted:
                    status_emitted = True
                    logger.info(
                        "chat_retrieval_started",
                        extra={
                            "event": "chat_retrieval_started",
                            "book_id": payload.book_id,
                            "thread_id": thread_id,
                            "tool": "search_keyword_tool",
                        },
                    )
                    yield _sse(
                        {"type": "status", "message": "正在检索书本内容…", "tool": "search_keyword_tool"},
                        event="status",
                    )

                token = _content_to_text(getattr(message_chunk, "content", ""))
                if token:
                    visible_token = answer_filter.feed(token)
                    answer += visible_token
                    token_events += 1
                    emitted_text = True
                    if visible_token:
                        yield _sse({"type": "token", "content": visible_token}, event="token")

            elif part_type == "updates":
                # If a provider does not expose message chunks, the completed
                # node update still contains the final AIMessage.
                for node_name, node_state in part_data.items():
                    # ToolNode updates contain the raw retrieved context. It
                    # is internal state and must not be sent as user-visible
                    # answer text; only the agent's final response is public.
                    if node_name != "agent_brain":
                        continue
                    if not isinstance(node_state, dict):
                        continue
                    for message in node_state.get("messages", []):
                        if getattr(message, "tool_calls", None) and not status_emitted:
                            status_emitted = True
                            logger.info(
                                "chat_retrieval_started",
                                extra={
                                    "event": "chat_retrieval_started",
                                    "book_id": payload.book_id,
                                    "thread_id": thread_id,
                                    "tool": "search_keyword_tool",
                                },
                            )
                            yield _sse(
                                {"type": "status", "message": "正在检索书本内容…", "tool": "search_keyword_tool"},
                                event="status",
                            )

                        if not emitted_text:
                            text = _content_to_text(getattr(message, "content", ""))
                            if text:
                                visible_text = answer_filter.feed(text)
                                answer += visible_text
                                token_events += 1
                                emitted_text = True
                                if visible_text:
                                    yield _sse({"type": "token", "content": visible_text}, event="token")

        trailing_text = answer_filter.flush()
        if trailing_text:
            answer += trailing_text
            token_events += 1
            yield _sse({"type": "token", "content": trailing_text}, event="token")

        library_storage = getattr(request.app.state, "library_storage", None)
        if library_storage is not None and answer:
            await asyncio.to_thread(
                library_storage.add_message,
                thread_id,
                payload.book_id,
                "assistant",
                answer,
            )
        logger.info(
            "chat_completed",
            extra={
                "event": "chat_completed",
                "book_id": payload.book_id,
                "thread_id": thread_id,
                "answer_chars": len(answer),
                "token_events": token_events,
                "duration_ms": round((perf_counter() - started_at) * 1000, 2),
            },
        )
        yield _sse({"type": "done", "thread_id": thread_id}, event="done")

    except asyncio.CancelledError:
        logger.warning(
            "chat_cancelled",
            extra={
                "event": "chat_cancelled",
                "book_id": payload.book_id,
                "thread_id": thread_id,
                "answer_chars": len(answer),
                "duration_ms": round((perf_counter() - started_at) * 1000, 2),
            },
        )
        raise
    except Exception:
        logger.exception(
            "chat_failed",
            extra={
                "event": "chat_failed",
                "book_id": payload.book_id,
                "thread_id": thread_id,
                "answer_chars": len(answer),
                "duration_ms": round((perf_counter() - started_at) * 1000, 2),
            },
        )
        yield _sse(
            {"type": "error", "message": "对话处理失败，请稍后重试"},
            event="error",
        )
    finally:
        request_id_context.reset(request_token)


def _store_uploaded_file(upload_dir: Path, book_id: str, upload: UploadFile) -> tuple[Path, str]:
    original_name = Path(upload.filename or "document").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("目前只支持 PDF 或 TXT 文件")

    tenant_dir = upload_dir / book_id
    tenant_dir.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid4().hex}{suffix}"
    destination = tenant_dir / stored_name

    with destination.open("wb") as output:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    return destination, original_name


def _ingest_file(pipeline: VectorPipeline, vector_store, path: Path, book_id: str) -> int:
    pipeline.book_to_store_by_stream(file_path=str(path), book_id=book_id)
    return vector_store.get_collection_by_tenant_id(book_id).count()


def _display_meta(path: Path, pages: int | None) -> str:
    suffix = path.suffix.upper().lstrip(".") or "FILE"
    return f"{suffix} · {pages} pages" if pages is not None else f"{suffix} · indexed"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = RuntimeSettings.from_environment()
    settings.vector_db_path.mkdir(parents=True, exist_ok=True)
    settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.library_db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    vector_store = get_global_vector_store(db_path=settings.vector_db_path)
    agent_builder = BookAgent()
    pipeline = VectorPipeline(vector_store=vector_store)
    library_storage = LibraryStorage(settings.library_db_path)

    known_books = {book["id"] for book in library_storage.list_books()}

    # Recover metadata for files uploaded before the metadata database was
    # introduced. The original filename is not available in that case, so the
    # stored book_id remains the safe fallback title.
    for book_dir in settings.upload_dir.iterdir():
        if not book_dir.is_dir() or book_dir.name in known_books:
            continue
        files = [path for path in book_dir.iterdir() if path.is_file()]
        if not files:
            continue
        source = files[0]
        pages = get_pdf_page_count(str(source)) if source.suffix.lower() == ".pdf" else None
        library_storage.upsert_book(
            book_id=book_dir.name,
            title=book_dir.name.replace("_", " ").replace("-", " ").title(),
            meta=_display_meta(source, pages),
            filename=source.name,
            chunks=0,
        )

    # Normalize metadata for books that were already in the database before
    # page counts became the shelf display format.
    bundled_source = PROJECT_ROOT / "rag_engine" / "敏捷项目管理.pdf"
    if bundled_source.exists() and "agile_project_management" in known_books:
        library_storage.update_book_meta(
            "agile_project_management",
            _display_meta(bundled_source, get_pdf_page_count(str(bundled_source))),
        )
    for book_dir in settings.upload_dir.iterdir():
        if not book_dir.is_dir() or book_dir.name not in known_books:
            continue
        sources = [path for path in book_dir.iterdir() if path.is_file()]
        if not sources:
            continue
        source = sources[0]
        pages = get_pdf_page_count(str(source)) if source.suffix.lower() == ".pdf" else None
        library_storage.update_book_meta(book_dir.name, _display_meta(source, pages))

    # The async checkpointer is required by astream_events/astream so the API
    # does not block on SQLite while a client is receiving tokens.
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    async with AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_db_path)) as checkpointer:
        app.state.settings = settings
        app.state.vector_store = vector_store
        app.state.agent_builder = agent_builder
        app.state.pipeline = pipeline
        app.state.library_storage = library_storage
        app.state.agent_app = agent_builder.build_graph(
            checkpointer=checkpointer,
            streaming=True,
        )
        yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Library Agent API",
        version="0.3.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        incoming_request_id = request.headers.get("X-Request-ID", "").strip()
        request_id = (incoming_request_id[:128] or uuid4().hex)
        request.state.request_id = request_id
        request_token = request_id_context.set(request_id)
        started_at = perf_counter()
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            logger.info(
                "http_response_started",
                extra={
                    "event": "http_response_started",
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                },
            )
            return response
        except Exception:
            logger.exception(
                "http_request_failed",
                extra={
                    "event": "http_request_failed",
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                },
            )
            raise
        finally:
            request_id_context.reset(request_token)

    settings = RuntimeSettings.from_environment()
    allow_all_origins = settings.cors_origins == ("*",)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=not allow_all_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "agent_ready": hasattr(app.state, "agent_app")}

    @app.get("/api/books", response_model=list[BookResponse])
    async def books(request: Request):
        library_storage = getattr(request.app.state, "library_storage", None)
        if library_storage is None:
            raise HTTPException(status_code=503, detail="书架服务尚未完成初始化")
        return await asyncio.to_thread(library_storage.list_books)

    @app.get(
        "/api/books/{book_id}/threads/{thread_id}/messages",
        response_model=list[HistoryMessageResponse],
    )
    async def history(book_id: str, thread_id: str, request: Request):
        try:
            book_id = validate_book_id(book_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not 1 <= len(thread_id) <= 128:
            raise HTTPException(status_code=422, detail="thread_id 长度必须为 1-128 位")

        library_storage = getattr(request.app.state, "library_storage", None)
        if library_storage is None:
            raise HTTPException(status_code=503, detail="历史记录服务尚未完成初始化")
        messages = await asyncio.to_thread(library_storage.list_messages, book_id, thread_id)
        for message in messages:
            if message.get("role") == "assistant":
                message["content"] = _sanitize_assistant_output(message["content"])
        return messages

    @app.post("/api/chat")
    async def chat(payload: ChatRequest, request: Request):
        return StreamingResponse(
            _chat_events(request, payload),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/upload", response_model=UploadResponse, status_code=201)
    async def upload(
        request: Request,
        book_id: str = Form(...),
        file: UploadFile = File(...),
    ):
        try:
            book_id = validate_book_id(book_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        pipeline = getattr(request.app.state, "pipeline", None)
        vector_store = getattr(request.app.state, "vector_store", None)
        settings = getattr(request.app.state, "settings", RuntimeSettings.from_environment())
        if pipeline is None or vector_store is None:
            raise HTTPException(status_code=503, detail="服务尚未完成初始化")

        upload_started_at = perf_counter()
        logger.info(
            "upload_started",
            extra={
                "event": "upload_started",
                "book_id": book_id,
                "upload_filename": Path(file.filename or "document").name,
            },
        )
        try:
            path, original_name = await asyncio.to_thread(
                _store_uploaded_file,
                settings.upload_dir,
                book_id,
                file,
            )
            pages = (
                await asyncio.to_thread(get_pdf_page_count, str(path))
                if path.suffix.lower() == ".pdf"
                else None
            )
            chunks = await asyncio.to_thread(_ingest_file, pipeline, vector_store, path, book_id)
            library_storage = getattr(request.app.state, "library_storage", None)
            if library_storage is not None:
                await asyncio.to_thread(
                    library_storage.upsert_book,
                    book_id,
                    Path(original_name).stem,
                    _display_meta(path, pages),
                    original_name,
                    chunks,
                )
        except ValueError as exc:
            logger.warning(
                "upload_rejected",
                extra={"event": "upload_rejected", "book_id": book_id, "reason": str(exc)},
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception(
                "upload_failed",
                extra={
                    "event": "upload_failed",
                    "book_id": book_id,
                    "duration_ms": round((perf_counter() - upload_started_at) * 1000, 2),
                },
            )
            raise HTTPException(status_code=500, detail="文件解析或向量入库失败") from exc
        finally:
            await file.close()

        logger.info(
            "upload_completed",
            extra={
                "event": "upload_completed",
                "book_id": book_id,
                "upload_filename": original_name,
                "chunks": chunks,
                "pages": pages,
                "duration_ms": round((perf_counter() - upload_started_at) * 1000, 2),
            },
        )

        return UploadResponse(
            book_id=book_id,
            filename=original_name,
            chunks=chunks,
            pages=pages,
            message="文件上传并建库成功，现在可以开始对话",
        )

    return app


app = create_app()
