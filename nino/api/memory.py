"""Memory management API endpoints for Nino."""

import time
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from nino.auth import get_current_user, verify_token
from nino.config import get_settings
from nino.memory.db import MemoryDB
from nino.memory.facts import FactManager
from nino.memory.vector.base import Embedder, VectorStore
from nino.memory.vector.ollama_embedder import OllamaEmbedder
from nino.memory.vector.sqlite_store import SQLiteVectorStore
from nino.memory.conversation import ConversationStore
from nino.security.audit import AuditLogger

router = APIRouter(prefix="/memory", tags=["memory"], dependencies=[Depends(verify_token)])


# Schemas
class FactPatchRequest(BaseModel):
    text: str | None = Field(default=None, description="Updated fact text")
    category: str | None = Field(default=None, description="Updated fact category")


class FactCreateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Fact text to store")
    category: str = Field(default="general", description="Category of the fact")


class SessionTitleRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="New title for the session")


class SessionCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200, description="Optional initial session title")


class ForgetRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Semantic search query of what to forget")
    confirm: bool = Field(default=False, description="Set to true to confirm deletion of matched memories")


def get_memory_db(request: Request) -> MemoryDB:
    db = getattr(request.app.state, "memory_db", None)
    if not db:
        settings_getter = request.app.dependency_overrides.get(get_settings, get_settings)
        settings = settings_getter()
        db = MemoryDB(db_path=settings.db_path, encryption_key=settings.memory_encryption_key)
        request.app.state.memory_db = db
    return db


def get_vector_store(request: Request) -> VectorStore:
    vstore = getattr(request.app.state, "vector_store", None)
    if not vstore:
        vstore = SQLiteVectorStore(get_memory_db(request))
        request.app.state.vector_store = vstore
    return vstore


def get_embedder(request: Request) -> Embedder:
    embedder = getattr(request.app.state, "embedder", None)
    if not embedder:
        settings_getter = request.app.dependency_overrides.get(get_settings, get_settings)
        settings = settings_getter()
        embedder = OllamaEmbedder(base_url=settings.ollama_base_url, model_name=settings.embed_model)
        request.app.state.embedder = embedder
    return embedder


def get_fact_manager(request: Request) -> FactManager:
    mgr = getattr(request.app.state, "fact_manager", None)
    if not mgr:
        settings_getter = request.app.dependency_overrides.get(get_settings, get_settings)
        settings = settings_getter()
        mgr = FactManager(
            memory_db=get_memory_db(request),
            vector_store=get_vector_store(request),
            embedder=get_embedder(request),
            sensitive_memory=settings.sensitive_memory,
        )
        request.app.state.fact_manager = mgr
    return mgr


def get_audit_logger(request: Request) -> AuditLogger:
    logger = getattr(request.app.state, "audit_logger", None)
    if not logger:
        settings_getter = request.app.dependency_overrides.get(get_settings, get_settings)
        settings = settings_getter()
        logger = AuditLogger(log_path=settings.audit_log_path, log_content=settings.log_content)
        request.app.state.audit_logger = logger
    return logger


def get_conv_store(request: Request) -> ConversationStore:
    store = getattr(request.app.state, "conv_store", None)
    if not store:
        db = get_memory_db(request)
        store = ConversationStore(memory_db=db)
        request.app.state.conv_store = store
    return store


@router.get("/sessions", summary="List saved conversation sessions")
async def list_sessions(
    limit: int = Query(default=50, ge=1, le=200, description="Max sessions to return"),
    conv_store: ConversationStore = Depends(get_conv_store),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Retrieve saved local chat sessions with message counts and timestamps."""
    user_id = current_user.get("id", "admin")
    return {"sessions": conv_store.list_sessions(limit=limit, user_id=user_id)}


@router.get("/sessions/latest", summary="Get the latest conversation session with messages")
async def get_latest_session(
    conv_store: ConversationStore = Depends(get_conv_store),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Fetch the most recent session and its full message history."""
    user_id = current_user.get("id", "admin")
    session = conv_store.get_latest_session(user_id=user_id)
    if not session:
        return {"session": None, "messages": []}
    messages = conv_store.get_messages(session["id"], user_id=user_id)
    return {"session": session, "messages": messages}


@router.get("/sessions/{session_id}", summary="Get a conversation session and its messages")
async def get_session(
    session_id: str,
    conv_store: ConversationStore = Depends(get_conv_store),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Fetch session details and its decrypted message history by session ID."""
    user_id = current_user.get("id", "admin")
    session = conv_store.get_session(session_id, user_id=user_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    messages = conv_store.get_messages(session_id, user_id=user_id)
    return {"session": session, "messages": messages}


@router.post("/sessions/{session_id}/title", summary="Update or save a session title")
async def update_session_title(
    session_id: str,
    payload: SessionTitleRequest,
    conv_store: ConversationStore = Depends(get_conv_store),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Name or rename a session to store it with a meaningful label."""
    user_id = current_user.get("id", "admin")
    session = conv_store.get_session(session_id, user_id=user_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    conv_store.update_session_title(session_id, payload.title)
    return {"status": "ok", "session_id": session_id, "title": payload.title}


@router.post("/sessions", summary="Create a new conversation session")
async def create_session(
    payload: SessionCreateRequest | None = None,
    conv_store: ConversationStore = Depends(get_conv_store),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Explicitly create a new conversation session."""
    title = payload.title if payload else None
    user_id = current_user.get("id", "admin")
    session_id = conv_store.create_session(title=title, user_id=user_id)
    return {"status": "created", "session_id": session_id, "title": title}


@router.delete("/sessions/{session_id}", summary="Delete a conversation session")
async def delete_session(
    session_id: str,
    conv_store: ConversationStore = Depends(get_conv_store),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Delete a conversation session and all its messages."""
    user_id = current_user.get("id", "admin")
    deleted = conv_store.delete_session(session_id, user_id=user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"status": "deleted", "session_id": session_id}


@router.post("/facts/store", summary="Explicitly store a durable personal fact")
async def store_fact(
    payload: FactCreateRequest,
    fact_manager: FactManager = Depends(get_fact_manager),
    audit_logger: AuditLogger = Depends(get_audit_logger),
):
    """Explicitly save a personal fact into local SQLite and vector store."""
    start = time.perf_counter()
    saved = await fact_manager.add_or_update_fact(text=payload.text, category=payload.category)
    dur = (time.perf_counter() - start) * 1000
    if not saved:
        return {"status": "already_exists_or_sensitive", "fact": None}

    audit_logger.log_memory_operation(
        action="store",
        entity_type="fact",
        count=1,
        duration_ms=dur,
    )
    return {"status": "saved", "fact": saved}


@router.get("/facts", summary="List stored personal facts")
async def list_facts(
    category: str | None = Query(default=None, description="Filter by category"),
    current_only: bool = Query(default=True, description="Filter for current facts only"),
    fact_manager: FactManager = Depends(get_fact_manager),
    audit_logger: AuditLogger = Depends(get_audit_logger),
):
    """Retrieve personal facts with category and current_only filters."""
    start = time.perf_counter()
    facts = fact_manager.list_facts(category=category, current_only=current_only)
    dur = (time.perf_counter() - start) * 1000
    audit_logger.log_memory_operation(
        action="list",
        entity_type="fact",
        count=len(facts),
        duration_ms=dur,
    )
    return {"facts": facts, "count": len(facts)}


@router.patch("/facts/{fact_id}", summary="Update a stored fact")
async def update_fact(
    fact_id: str,
    payload: FactPatchRequest,
    fact_manager: FactManager = Depends(get_fact_manager),
    audit_logger: AuditLogger = Depends(get_audit_logger),
):
    """Edit text or category of an existing fact."""
    start = time.perf_counter()
    success = await fact_manager.update_fact(
        fact_id=fact_id,
        new_text=payload.text,
        new_category=payload.category,
    )
    dur = (time.perf_counter() - start) * 1000
    if not success:
        raise HTTPException(status_code=404, detail=f"Fact '{fact_id}' not found.")

    audit_logger.log_memory_operation(
        action="update",
        entity_type="fact",
        entity_id=fact_id,
        duration_ms=dur,
    )
    return {"status": "updated", "id": fact_id}


@router.delete("/facts/{fact_id}", summary="Delete a fact from SQLite and vector store")
async def delete_fact(
    fact_id: str,
    fact_manager: FactManager = Depends(get_fact_manager),
    audit_logger: AuditLogger = Depends(get_audit_logger),
):
    """Delete a single fact permanently from both SQLite and vector search."""
    start = time.perf_counter()
    success = await fact_manager.delete_fact(fact_id)
    dur = (time.perf_counter() - start) * 1000
    if not success:
        raise HTTPException(status_code=404, detail=f"Fact '{fact_id}' not found.")

    audit_logger.log_memory_operation(
        action="delete",
        entity_type="fact",
        entity_id=fact_id,
        duration_ms=dur,
    )
    return {"status": "deleted", "id": fact_id}


@router.post("/forget", summary="Find and forget memories matching query")
async def forget_memory(
    payload: ForgetRequest,
    embedder: Embedder = Depends(get_embedder),
    vector_store: VectorStore = Depends(get_vector_store),
    fact_manager: FactManager = Depends(get_fact_manager),
    audit_logger: AuditLogger = Depends(get_audit_logger),
):
    """Perform semantic search for memories matching query.

    If confirm=false, returns matched candidate memories for preview.
    If confirm=true, deletes matched candidates from both SQLite and the vector store.
    """
    start = time.perf_counter()
    query_vec = await embedder.embed(payload.query)
    matches = await vector_store.search(query_vector=query_vec, top_k=5)

    # Filter relevant matches
    relevant = [m for m in matches if m.score >= 0.50]

    deleted_count = 0
    if payload.confirm and relevant:
        for m in relevant:
            if m.entity_type == "fact":
                await fact_manager.delete_fact(m.entity_id)
                deleted_count += 1
            elif m.entity_type == "document_chunk":
                await vector_store.delete(m.id)
                deleted_count += 1

    dur = (time.perf_counter() - start) * 1000
    audit_logger.log_memory_operation(
        action="forget" if payload.confirm else "preview_forget",
        entity_type="memory",
        count=deleted_count if payload.confirm else len(relevant),
        duration_ms=dur,
    )

    candidates = [
        {
            "id": m.id,
            "entity_type": m.entity_type,
            "entity_id": m.entity_id,
            "similarity": round(m.score, 3),
            "preview": m.metadata.get("text", "")[:120],
        }
        for m in relevant
    ]

    return {
        "query": payload.query,
        "confirmed": payload.confirm,
        "deleted_count": deleted_count,
        "matches": candidates,
    }


@router.get("/export", summary="Export all memory data")
async def export_memory(
    memory_db: MemoryDB = Depends(get_memory_db),
    fact_manager: FactManager = Depends(get_fact_manager),
    audit_logger: AuditLogger = Depends(get_audit_logger),
):
    """Export complete snapshot of facts, sessions, and indexed documents."""
    start = time.perf_counter()
    facts = fact_manager.list_facts(current_only=False)

    with memory_db.get_connection() as conn:
        sessions_cur = conn.execute("SELECT id, title, created_at, updated_at FROM sessions")
        sessions = [dict(r) for r in sessions_cur.fetchall()]

        docs_cur = conn.execute("SELECT id, file_path, file_hash, indexed_at, chunk_count FROM documents")
        docs = [dict(r) for r in docs_cur.fetchall()]

    dur = (time.perf_counter() - start) * 1000
    audit_logger.log_memory_operation(
        action="export",
        entity_type="all",
        count=len(facts) + len(sessions) + len(docs),
        duration_ms=dur,
    )

    return {
        "facts": facts,
        "sessions": sessions,
        "documents": docs,
    }


@router.delete("/all", summary="Purge all memory data (requires confirm=true)")
async def delete_all_memory(
    confirm: bool = Query(default=False, description="Set confirm=true to wipe memory permanently"),
    memory_db: MemoryDB = Depends(get_memory_db),
    vector_store: VectorStore = Depends(get_vector_store),
    audit_logger: AuditLogger = Depends(get_audit_logger),
):
    """Completely wipe all stored facts, sessions, documents, and vector entries."""
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Confirmation required to wipe memory. Add query parameter '?confirm=true'.",
        )

    start = time.perf_counter()
    with memory_db.get_connection() as conn:
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM sessions")
        conn.execute("DELETE FROM facts")
        conn.execute("DELETE FROM document_chunks")
        conn.execute("DELETE FROM documents")
        conn.execute("DELETE FROM vector_entries")

    await vector_store.delete_all()
    dur = (time.perf_counter() - start) * 1000

    audit_logger.log_memory_operation(
        action="delete_all",
        entity_type="all",
        duration_ms=dur,
    )

    return {"status": "all_memory_purged", "confirmed": True}
