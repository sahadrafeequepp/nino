import asyncio
import io
import ipaddress
import json
import logging
from pathlib import Path
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal
import wave

# Suppress noisy HTTP access logs (e.g. POST /chat/stream HTTP/1.1 200 OK)
logging.getLogger("uvicorn.access").disabled = True
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

logger = logging.getLogger("nino.main")

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from nino.api.automations import router as automations_router
from nino.api.avatar_ws import router as avatar_router
from nino.api.confirmations import router as confirmations_router
from nino.api.memory import router as memory_router
from nino.api.vision import vision_router
from nino.api.voice_ws import router as voice_router
from nino.api.agents import router as agents_router, set_agent_registry
from nino.api.dashboard import router as dashboard_router
from nino.api.auth import router as auth_router
from nino.agents import AgentRegistry, SubAgentRunner, get_run_store
from nino.tools.delegation import DelegateTaskTool
from nino.auth import get_current_user, verify_token
from nino.automation.briefing import DailyBriefingGenerator
from nino.automation.notifications import (
    ConsoleNotificationChannel,
    NotificationDispatcher,
    NtfyNotificationChannel,
)
from nino.automation.policy import ProactivePolicyEngine
from nino.automation.scheduler import SQLiteJobScheduler
from nino.config import Settings, get_settings
from nino.memory.conversation import ConversationStore
from nino.memory.db import MemoryDB
from nino.memory.facts import FactManager
from nino.memory.retrieval import RetrievalPipeline
from nino.memory.vector.ollama_embedder import OllamaEmbedder
from nino.memory.vector.sqlite_store import SQLiteVectorStore
from nino.models.base import LLMAdapter
from nino.models.factory import get_llm_adapter
from nino.persona.manager import PersonaManager
from nino.security.audit import AuditLogger
from nino.security.lockout import AuthLockoutManager, get_lockout_manager
from nino.tools import (
    CentralExecutor,
    Confirmation,
    ConfirmationManager,
    ToolOrchestrator,
    ToolRegistry,
    build_default_registry,
)
from nino.tools.weather import get_stored_location


# Schemas
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"] = Field(
        default="user", description="Role of the message sender"
    )
    content: str = Field(default="", description="Text message content")


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(
        ..., description="Conversation history"
    )
    session_id: str | None = Field(
        default=None, description="Optional conversational session ID"
    )
    model: str | None = Field(
        default=None, description="Optional model override"
    )
    confirmation_id: str | None = Field(
        default=None, description="Optional confirmation ID to execute an approved tool"
    )


class ChatResponse(BaseModel):
    reply: str
    model: str
    session_id: str
    citations: list[str] = Field(default_factory=list)
    pending_confirmation: Confirmation | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    debug_retrieval: dict[str, Any] | None = None



class HealthResponse(BaseModel):
    status: str
    assistant: str = "Nino"
    adapter: str
    model: str
    ollama_reachable: bool | None = None


# Lifespan Management
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.audit_logger = AuditLogger(
        log_path=settings.audit_log_path,
        log_content=settings.log_content,
    )
    app.state.memory_db = MemoryDB(
        db_path=settings.db_path,
        encryption_key=settings.memory_encryption_key,
    )
    app.state.vector_store = SQLiteVectorStore(memory_db=app.state.memory_db)
    app.state.embedder = OllamaEmbedder(
        base_url=settings.ollama_base_url,
        model_name=settings.embed_model,
    )
    app.state.conv_store = ConversationStore(memory_db=app.state.memory_db)
    app.state.fact_manager = FactManager(
        memory_db=app.state.memory_db,
        vector_store=app.state.vector_store,
        embedder=app.state.embedder,
        sensitive_memory=settings.sensitive_memory,
    )
    app.state.retrieval = RetrievalPipeline(
        memory_db=app.state.memory_db,
        vector_store=app.state.vector_store,
        embedder=app.state.embedder,
        token_budget_memory=settings.token_budget_memory,
        debug_retrieval=settings.debug_retrieval,
    )
    app.state.adapter = get_llm_adapter(settings)
    app.state.confirmation_manager = ConfirmationManager(
        memory_db=app.state.memory_db,
        default_expiry_minutes=settings.confirmation_expiry_minutes,
    )
    app.state.tool_registry = build_default_registry(
        memory_db=app.state.memory_db,
        settings=settings,
        fact_manager=app.state.fact_manager,
    )
    app.state.central_executor = CentralExecutor(
        registry=app.state.tool_registry,
        confirmation_manager=app.state.confirmation_manager,
        audit_logger=app.state.audit_logger,
        allow_restricted_tools=settings.allow_restricted_tools,
        tool_timeout_seconds=settings.tool_timeout_seconds,
        output_max_chars=settings.tool_output_max_chars,
        log_content=settings.log_content,
    )
    app.state.tool_orchestrator = ToolOrchestrator(
        registry=app.state.tool_registry,
        executor=app.state.central_executor,
        max_iterations=settings.max_tool_iterations,
        total_timeout=settings.total_tool_timeout_seconds,
    )
    app.state.persona_manager = PersonaManager()

    # Security Lockout & Proactive Automations (Phase 5)
    app.state.lockout_manager = AuthLockoutManager(
        max_failed_attempts=settings.auth_max_failed_attempts,
        lockout_seconds=settings.auth_lockout_seconds,
    )
    app.state.proactive_policy = ProactivePolicyEngine(
        memory_db=app.state.memory_db,
        settings=settings,
    )
    dispatcher = NotificationDispatcher([ConsoleNotificationChannel()])
    if settings.ntfy_url and settings.ntfy_topic:
        dispatcher.add_channel(NtfyNotificationChannel(settings))
    app.state.notification_dispatcher = dispatcher

    app.state.briefing_generator = DailyBriefingGenerator(
        memory_db=app.state.memory_db,
        settings=settings,
        adapter=app.state.adapter,
    )
    app.state.scheduler = SQLiteJobScheduler(
        memory_db=app.state.memory_db,
        settings=settings,
        policy_engine=app.state.proactive_policy,
        dispatcher=dispatcher,
        briefing_gen=app.state.briefing_generator,
    )
    await app.state.scheduler.start()

    # Vision Subsystem (Phase 6A)
    from nino.vision.ollama import OllamaVisionAdapter
    from nino.vision.storage import TempImageManager
    app.state.temp_image_manager = TempImageManager(settings.temp_image_dir)
    if not getattr(app.state, "vision_adapter", None):
        app.state.vision_adapter = OllamaVisionAdapter(
            base_url=settings.ollama_base_url,
            model=settings.vision_model,
        )
    app.state.executor = app.state.central_executor

    # Sub-Agents Subsystem (Phase 6C)
    app.state.agent_registry = AgentRegistry()
    set_agent_registry(app.state.agent_registry)
    app.state.subagent_runner = SubAgentRunner(
        agent_registry=app.state.agent_registry,
        global_tool_registry=app.state.tool_registry,
        executor=app.state.central_executor,
        audit_logger=app.state.audit_logger,
        run_store=get_run_store(),
    )
    # Register DelegateTaskTool in the central tool registry for orchestrator access
    app.state.tool_registry.register(
        DelegateTaskTool(
            runner=app.state.subagent_runner,
            adapter=app.state.adapter,
        )
    )

    # Reset lockout manager on startup so stale test runs don't lock local dev
    get_lockout_manager().reset()

    yield

    await app.state.scheduler.stop()


app = FastAPI(
    title="Nino - Local AI Assistant Gateway",
    description="Local-first, security-minded personal AI assistant backend with persistent memory and tools",
    version="0.3.0",
    lifespan=lifespan,
)

# Allow origins for localhost and dev clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(auth_router)
app.include_router(memory_router)
app.include_router(confirmations_router)
app.include_router(voice_router)
app.include_router(avatar_router)
app.include_router(automations_router)
app.include_router(vision_router)
app.include_router(agents_router)
app.include_router(dashboard_router)



@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    """Audit every incoming HTTP request and record to JSONL."""
    start_time = time.perf_counter()
    client_ip = request.client.host if request.client else "127.0.0.1"
    endpoint = request.url.path
    method = request.method

    settings_getter = request.app.dependency_overrides.get(get_settings, None)
    if settings_getter:
        settings = settings_getter()
        audit_logger = AuditLogger(
            log_path=settings.audit_log_path,
            log_content=settings.log_content,
        )
    else:
        audit_logger = getattr(request.app.state, "audit_logger", None)
        if not audit_logger:
            settings = get_settings()
            audit_logger = AuditLogger(
                log_path=settings.audit_log_path,
                log_content=settings.log_content,
            )

    try:
        response: Response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000
        audit_logger.log(
            endpoint=endpoint,
            method=method,
            status_code=response.status_code,
            duration_ms=duration_ms,
            client_ip=client_ip,
        )
        return response
    except Exception as exc:
        duration_ms = (time.perf_counter() - start_time) * 1000
        audit_logger.log(
            endpoint=endpoint,
            method=method,
            status_code=500,
            duration_ms=duration_ms,
            client_ip=client_ip,
        )
        raise exc


# Dependencies
def get_adapter(request: Request) -> LLMAdapter:
    adapter = getattr(request.app.state, "adapter", None)
    if adapter is None:
        settings = get_settings()
        return get_llm_adapter(settings)
    return adapter


def get_conv_store(request: Request) -> ConversationStore:
    return getattr(request.app.state, "conv_store", None) or ConversationStore(
        memory_db=MemoryDB(get_settings().db_path)
    )


def get_retrieval(request: Request) -> RetrievalPipeline:
    return getattr(request.app.state, "retrieval", None)


def get_fact_mgr(request: Request) -> FactManager:
    return getattr(request.app.state, "fact_manager", None)


def get_orchestrator(request: Request) -> ToolOrchestrator:
    orch = getattr(request.app.state, "tool_orchestrator", None)
    if orch is None:
        settings = get_settings()
        db = getattr(request.app.state, "memory_db", None) or MemoryDB(settings.db_path)
        conf_mgr = getattr(request.app.state, "confirmation_manager", None) or ConfirmationManager(
            db, default_expiry_minutes=settings.confirmation_expiry_minutes
        )
        audit_log = getattr(request.app.state, "audit_logger", None) or AuditLogger(
            settings.audit_log_path, log_content=settings.log_content
        )
        registry = getattr(request.app.state, "tool_registry", None) or build_default_registry(db, settings)
        executor = getattr(request.app.state, "central_executor", None) or CentralExecutor(
            registry=registry,
            confirmation_manager=conf_mgr,
            audit_logger=audit_log,
            allow_restricted_tools=settings.allow_restricted_tools,
            tool_timeout_seconds=settings.tool_timeout_seconds,
            output_max_chars=settings.tool_output_max_chars,
            log_content=settings.log_content,
        )
        orch = ToolOrchestrator(
            registry=registry,
            executor=executor,
            max_iterations=settings.max_tool_iterations,
            total_timeout=settings.total_tool_timeout_seconds,
        )
        request.app.state.tool_orchestrator = orch
    return orch


def get_persona_mgr(request: Request) -> PersonaManager:
    return getattr(request.app.state, "persona_manager", None) or PersonaManager()



STATIC_DIR = Path(__file__).parent / "static"


def is_trusted_network_client(ip_str: str) -> bool:
    """Check if incoming client is on localhost, local private network (Wi-Fi/LAN), or Tailscale."""
    if not ip_str or ip_str in ("127.0.0.1", "::1", "localhost", "testclient"):
        return True
    try:
        ip = ipaddress.ip_address(ip_str)
        return (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip in ipaddress.ip_network("100.64.0.0/10")
        )
    except ValueError:
        return False


@app.get("/api/client-config", summary="Safe client configuration for local and LAN web UI")
async def client_config(request: Request, settings: Settings = Depends(get_settings)):
    """Return local connection configuration for localhost and private LAN clients."""
    client_ip = request.client.host if request.client else "127.0.0.1"
    is_trusted = is_trusted_network_client(client_ip)
    return {
        "assistant": "Nino",
        "auth_token": settings.auth_token if is_trusted else "",
        "model": settings.model_name,
        "adapter": settings.model_adapter,
        "is_local": is_trusted,
    }


@app.get("/", response_class=HTMLResponse, summary="Nino Web Chat Interface")
@app.get("/chat-ui", response_class=HTMLResponse, summary="Nino Web Chat Interface")
async def chat_ui():
    """Serve the Nino Web Chat single-page interface."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>Nino Chat UI not found</h1>", status_code=404)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


_tts_adapter = None


def get_tts_adapter():
    global _tts_adapter
    if _tts_adapter is None:
        from nino.voice.tts.piper_adapter import PiperTTSAdapter
        settings = get_settings()
        _tts_adapter = PiperTTSAdapter(voice=settings.tts_voice, speed=settings.tts_speed)
    return _tts_adapter


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 22050, channels: int = 1, sample_width: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


@app.post("/api/tts", summary="Synthesize speech audio using local Piper TTS neural voice")
async def api_tts_post(payload: TTSRequest):
    """Generate WAV audio using Nino's local neural Piper voice."""
    tts = get_tts_adapter()
    pcm = await tts.synthesize(payload.text)
    if not pcm:
        raise HTTPException(status_code=400, detail="Failed to synthesize speech")
    wav_bytes = pcm_to_wav(pcm, sample_rate=tts.sample_rate)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/api/tts", summary="Synthesize speech audio using local Piper TTS neural voice (GET)")
async def api_tts_get(text: str = Query(..., min_length=1, max_length=4000)):
    """Generate WAV audio using Nino's local neural Piper voice."""
    tts = get_tts_adapter()
    pcm = await tts.synthesize(text)
    if not pcm:
        raise HTTPException(status_code=400, detail="Failed to synthesize speech")
    wav_bytes = pcm_to_wav(pcm, sample_rate=tts.sample_rate)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/api/parse-document", summary="Extract text from uploaded document (PDF, code, text)")
async def parse_document(file: UploadFile = File(...)):
    """Extract readable text from uploaded documents (PDF, Markdown, Source Code, Text, CSV, JSON)."""
    filename = file.filename or "uploaded_file"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    extracted_text = ""
    pages = 1

    # PDF extraction
    if filename.lower().endswith(".pdf") or (file.content_type and "pdf" in file.content_type):
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    raise HTTPException(status_code=400, detail="PDF is password-protected or encrypted.")
            pages = len(reader.pages)
            text_parts = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(f"--- Page {i + 1} ---\n{page_text.strip()}")
            extracted_text = "\n\n".join(text_parts)
            if not extracted_text.strip():
                extracted_text = (
                    f"[Note: '{filename}' contains {pages} page(s), but no extractable text was found. "
                    "The PDF may be scanned images. Consider OCR or attaching key pages as images.]"
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {e}")
    else:
        # Text/Code/Data file decoding
        try:
            extracted_text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                extracted_text = content.decode("latin-1")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Could not decode text file: {e}")

    # Enforce safe upper length limit so prompt token budget isn't blown
    truncated = False
    max_doc_chars = 45000
    if len(extracted_text) > max_doc_chars:
        extracted_text = (
            extracted_text[:max_doc_chars]
            + f"\n\n... [Content truncated at {max_doc_chars} chars to fit LLM memory budget. Total document pages: {pages}] ..."
        )
        truncated = True

    return {
        "status": "success",
        "filename": filename,
        "pages": pages,
        "characters": len(extracted_text),
        "truncated": truncated,
        "text": extracted_text,
    }


@app.get("/health", response_model=HealthResponse, summary="Public health check")
async def health(settings: Settings = Depends(get_settings)):
    """Health check returning Nino's service and model status."""
    ollama_ok = None
    if settings.model_adapter == "ollama":
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
                ollama_ok = res.status_code == 200
        except Exception:
            ollama_ok = False

    return HealthResponse(
        status="ok",
        assistant="Nino",
        adapter=settings.model_adapter,
        model=settings.model_name,
        ollama_reachable=ollama_ok,
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat completion with persistent memory, document RAG, and tools",
    dependencies=[Depends(verify_token)],
)
async def chat(
    payload: ChatRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
    current_user: dict[str, Any] = Depends(get_current_user),
    adapter: LLMAdapter = Depends(get_adapter),
    conv_store: ConversationStore = Depends(get_conv_store),
    retrieval: RetrievalPipeline = Depends(get_retrieval),
    fact_manager: FactManager = Depends(get_fact_mgr),
    orchestrator: ToolOrchestrator = Depends(get_orchestrator),
    persona_mgr: PersonaManager = Depends(get_persona_mgr),
):
    """Complete a chat conversation with RAG retrieval, tool execution, session logging, and fact extraction."""
    valid_messages = [m for m in payload.messages if m.content and m.content.strip()]
    if not valid_messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot complete chat with empty message.",
        )
    user_id = current_user.get("id", "admin")
    session_id = conv_store.ensure_session(payload.session_id, user_id=user_id)
    user_message = valid_messages[-1].content
    model_override = payload.model or settings.model_name

    # 1. Retrieve relevant facts and documents
    retrieved_ctx = None
    if retrieval:
        retrieved_ctx = await retrieval.retrieve_context(user_message)

    # 2. Assemble prompt within token budget
    assembled_messages = []
    stored_loc = get_stored_location(conv_store.memory_db) if conv_store else None
    loc_str = "Vadakara, Calicut, Kerala, India"
    if stored_loc:
        parts = [p for p in [stored_loc.get("city"), stored_loc.get("region"), stored_loc.get("country")] if p]
        loc_str = ", ".join(parts) if parts else loc_str

    base_system = persona_mgr.build_system_prompt(location_info=loc_str)
    system_parts = [base_system]

    if retrieved_ctx and retrieved_ctx.system_prompt:
        system_parts.append(retrieved_ctx.system_prompt)

    if orchestrator:
        system_parts.append(orchestrator.build_tool_system_prompt(user_message=user_message))

    system_content = "\n\n".join(system_parts)
    assembled_messages.append({"role": "system", "content": system_content})

    # Retrieve prior conversation messages from session store if populated
    history_messages = await conv_store.get_budgeted_messages(
        session_id=session_id,
        token_budget=settings.token_budget_total - settings.token_budget_memory,
        adapter=adapter,
    )
    if history_messages:
        assembled_messages.extend(history_messages)
        history_tuples = [(m["role"], m["content"]) for m in history_messages]
        for m in valid_messages:
            if (m.role, m.content) not in history_tuples:
                assembled_messages.append({"role": m.role, "content": m.content})
    else:
        for m in valid_messages:
            assembled_messages.append({"role": m.role, "content": m.content})

    # Enforce external provider privacy perimeter
    if settings.model_adapter == "external" and retrieval:
        assembled_messages = retrieval.strip_memory_for_external(assembled_messages)

    # 3. Call Orchestrator / LLM
    try:
        if orchestrator:
            orch_res = await orchestrator.run(
                messages=assembled_messages,
                adapter=adapter,
                confirmation_id=payload.confirmation_id,
                context_data=assembled_messages,
                model_override=model_override,
            )
            reply = orch_res.reply
            pending_conf = orch_res.pending_confirmation
            tool_calls_data = [t.model_dump() for t in orch_res.tool_calls]
        else:
            reply = await adapter.chat(assembled_messages, model=model_override)
            pending_conf = None
            tool_calls_data = []
    except RuntimeError as e:
        logger.warning("Upstream model/tool runtime error: %s", e)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))
    except Exception as e:
        logger.exception("Chat completion error: %s", e)
        err_msg = str(e).strip() or repr(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat completion error: {err_msg}",
        )


    # 4. Persist messages in session
    user_msg_id = conv_store.add_message(session_id, "user", user_message)
    conv_store.add_message(session_id, "assistant", reply)

    # Auto-title session if currently default placeholder
    try:
        sess_info = conv_store.get_session(session_id)
        if sess_info and sess_info.get("title", "").startswith("Session 20"):
            snippet = user_message.strip().replace("\n", " ")
            if len(snippet) > 40:
                snippet = snippet[:40].rstrip() + "..."
            if snippet:
                conv_store.update_session_title(session_id, snippet)
    except Exception:
        pass

    # 5. Background task for durable fact extraction
    if fact_manager:
        background_tasks.add_task(
            fact_manager.extract_facts_from_turn,
            user_message=user_message,
            source_message_id=user_msg_id,
            adapter=adapter,
        )

    debug_info = None
    if settings.debug_retrieval and retrieved_ctx:
        debug_info = {
            "facts_used": retrieved_ctx.retrieved_facts,
            "chunks_used": retrieved_ctx.retrieved_chunks,
        }

    return ChatResponse(
        reply=reply,
        model=model_override,
        session_id=session_id,
        citations=retrieved_ctx.cited_sources if retrieved_ctx else [],
        pending_confirmation=pending_conf,
        tool_calls=tool_calls_data,
        debug_retrieval=debug_info,
    )



@app.post(
    "/chat/stream",
    summary="Streaming chat completion via Server-Sent Events",
    dependencies=[Depends(verify_token)],
)
async def chat_stream(
    payload: ChatRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
    current_user: dict[str, Any] = Depends(get_current_user),
    adapter: LLMAdapter = Depends(get_adapter),
    conv_store: ConversationStore = Depends(get_conv_store),
    retrieval: RetrievalPipeline = Depends(get_retrieval),
    fact_manager: FactManager = Depends(get_fact_mgr),
    persona_mgr: PersonaManager = Depends(get_persona_mgr),
):
    """Stream chat response tokens using SSE with memory retrieval and background fact extraction."""
    valid_messages = [m for m in payload.messages if m.content and m.content.strip()]
    if not valid_messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot stream chat with empty message.",
        )
    user_id = current_user.get("id", "admin")
    session_id = conv_store.ensure_session(payload.session_id, user_id=user_id)
    user_message = valid_messages[-1].content
    model_override = payload.model or settings.model_name

    retrieved_ctx = None
    if retrieval:
        retrieved_ctx = await retrieval.retrieve_context(user_message)

    assembled_messages = []
    stored_loc = get_stored_location(conv_store.memory_db) if conv_store else None
    loc_str = "Vadakara, Calicut, Kerala, India"
    if stored_loc:
        parts = [p for p in [stored_loc.get("city"), stored_loc.get("region"), stored_loc.get("country")] if p]
        loc_str = ", ".join(parts) if parts else loc_str

    base_system = persona_mgr.build_system_prompt(location_info=loc_str)
    if retrieved_ctx and retrieved_ctx.system_prompt:
        system_content = f"{base_system}\n\n{retrieved_ctx.system_prompt}"
    else:
        system_content = base_system

    assembled_messages.append({"role": "system", "content": system_content})

    history_messages = await conv_store.get_budgeted_messages(
        session_id=session_id,
        token_budget=settings.token_budget_total - settings.token_budget_memory,
        adapter=adapter,
    )
    if history_messages:
        assembled_messages.extend(history_messages)
        history_tuples = [(m["role"], m["content"]) for m in history_messages]
        for m in valid_messages:
            if (m.role, m.content) not in history_tuples:
                assembled_messages.append({"role": m.role, "content": m.content})
    else:
        for m in valid_messages:
            assembled_messages.append({"role": m.role, "content": m.content})

    if settings.model_adapter == "external" and retrieval:
        assembled_messages = retrieval.strip_memory_for_external(assembled_messages)

    user_msg_id = conv_store.add_message(session_id, "user", user_message)

    # Auto-title session if currently default placeholder
    try:
        sess_info = conv_store.get_session(session_id)
        if sess_info and sess_info.get("title", "").startswith("Session 20"):
            snippet = user_message.strip().replace("\n", " ")
            if len(snippet) > 40:
                snippet = snippet[:40].rstrip() + "..."
            if snippet:
                conv_store.update_session_title(session_id, snippet)
    except Exception:
        pass

    async def event_generator() -> AsyncIterator[str]:
        # Emit initial metadata
        init_data = json.dumps({
            "content": "",
            "session_id": session_id,
            "citations": retrieved_ctx.cited_sources if retrieved_ctx else [],
        })
        yield f"event: meta\ndata: {init_data}\n\n"

        full_reply = []
        try:
            async for chunk in adapter.stream_chat(assembled_messages, model=model_override):
                full_reply.append(chunk)
                data = json.dumps({"content": chunk})
                yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
        except RuntimeError as e:
            err_data = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {err_data}\n\n"
        except Exception as e:
            err_data = json.dumps({"error": f"Stream error: {e}"})
            yield f"event: error\ndata: {err_data}\n\n"

        # Save assistant reply and trigger background fact extraction
        final_text = "".join(full_reply)
        if final_text:
            conv_store.add_message(session_id, "assistant", final_text)
            if fact_manager:
                asyncio.create_task(
                    fact_manager.extract_facts_from_turn(
                        user_message=user_message,
                        source_message_id=user_msg_id,
                        adapter=adapter,
                    )
                )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        background=background_tasks,
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
